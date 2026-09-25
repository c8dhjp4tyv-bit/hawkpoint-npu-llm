"""Host orchestration for the supported NPU-only decode graphs."""

from collections import OrderedDict
import gc
import logging
import os
import resource
import time

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device

from designs.attention_block import attention_block
from designs.decoder_layer import decoder_layer
from designs.elementwise import (
    residual_add,
    swiglu,
)
from designs.project import norm_project, project, project_residual
from designs.project_bf16 import project_bf16, project_bf16_residual
from designs.qkv_rope import qkv_rope
from designs.qwen_decoder import qwen_decoder
from designs.rmsnorm import rmsnorm
from designs.tensor_copy import slice_bf16
from runtime.model import XDNA1Model
from runtime.cpu_backend import CPUDecoderStage, _bf16 as _bf16_cpu
from runtime.sampling import GREEDY, Sampler, SamplingParams
from runtime.stopping import (
    StopMatcher,
    log_softmax,
    normalize_logprobs,
    normalize_stop,
)
from runtime.tokenizer import SmolLMTokenizer, StreamDetokenizer


class NPUDecoder:
    def __init__(self, model_dir, context_length=64, npu_layers=None):
        if context_length != 64:
            raise ValueError("the current NPU attention kernel has a fixed 64-token cache")
        self._closed = False
        self.model = XDNA1Model(model_dir)
        self.model_family = self.model.metadata.get("model_family", "llama")
        default_system = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            if self.model_family == "qwen2"
            else "You are a helpful AI assistant named SmolLM."
        )
        self.tokenizer = SmolLMTokenizer(model_dir, default_system=default_system)
        self.context_length = context_length
        self.hidden_size = self.model.metadata["hidden_size"]
        self.intermediate_size = self.model.metadata["intermediate_size"]
        self.q_heads = self.model.metadata["attention_heads"]
        self.kv_heads = self.model.metadata["kv_heads"]
        self.head_dim = self.model.metadata["head_dim"]
        self.q_per_kv = self.q_heads // self.kv_heads
        self._rope_inv_freq = 1.0 / (
            float(self.model.metadata.get("rope_theta", 10000.0))
            ** (
                np.arange(0, self.head_dim, 2, dtype=np.float32)
                / np.float32(self.head_dim)
            )
        )
        self._rope_lut_cache = {}
        self._qwen_embedding_cache = OrderedDict()
        self._qwen_embedding_cache_limit = 128
        self._profile_enabled = os.environ.get("HAWKPOINT_PROFILE") == "1"
        # Token ids whose keys and values are currently valid at cache
        # positions 0..n-1. Attention reads only positions up to the current
        # one, so a later prompt that starts with the same tokens can resume
        # after them instead of recomputing the shared prefix.
        self._cached_prefix = []
        self._prefix_cache_enabled = (
            os.environ.get("HAWKPOINT_PREFIX_CACHE", "1") != "0"
        )
        self._phase_totals = {}
        self.rms_norm_eps = float(
            self.model.metadata.get("rms_norm_eps", 1e-5)
        )
        self.layers = self.model.metadata["layers"]
        self.npu_layers = (
            self.layers if npu_layers is None else int(npu_layers)
        )
        if not 0 <= self.npu_layers <= self.layers:
            raise ValueError(
                f"npu_layers must be between 0 and {self.layers}"
            )
        # The SmolLM graph is parameterised by the number of layers; the host
        # uses firmware-safe two-layer chunks. Keep Qwen's graph path separate
        # because it uses a different packed layout and cache contract.
        self._fused_smollm = (
            self.npu_layers > 0 and self.model_family != "qwen2"
        )
        self._use_quantized_decoder = (
            self._fused_smollm
            and os.environ.get("HAWKPOINT_DECODER_W8") == "1"
        )
        # The NPU matmul is convenient for Qwen's BF16 head, but on Phoenix
        # the host BLAS path is faster for SmolLM's 49k x 576 output matrix.
        # Keep an escape hatch for hardware where the NPU head wins.
        self._use_cpu_lm_head = (
            self.model_family != "qwen2"
            and os.environ.get("HAWKPOINT_NPU_LM_HEAD") != "1"
        )
        self._use_cpu_final_norm = (
            self._use_cpu_lm_head
            and self.model_family != "qwen2"
            and os.environ.get("HAWKPOINT_CPU_FINAL_NORM") == "1"
        )
        # Phoenix's command watchdog reliably completes the shipped graph for
        # one or two layers, while larger compile-time values may deadlock in
        # current firmware.  Keep two as the safe default, but allow a
        # hardware-specific override for benchmarking newer runtimes.
        try:
            requested_chunk = int(os.environ.get("HAWKPOINT_SMOLLM_CHUNK", "2"))
        except ValueError:
            requested_chunk = 2
        self._smollm_chunk_size = max(1, min(self.layers, requested_chunk))
        self._smollm_chunks = (
            [
                (first, min(self._smollm_chunk_size, self.npu_layers - first))
                for first in range(0, self.npu_layers, self._smollm_chunk_size)
            ]
            if self._fused_smollm
            else []
        )
        if self.npu_layers:
            set_current_device(from_name("npu", n_cols=4))
        self._final_norm_buffer = (
            iron.zeros(self.hidden_size, dtype=bfloat16, device="npu")
            if self.npu_layers
            else None
        )
        self._lm_head_buffer = (
            iron.zeros(
                self.model.metadata["vocab_size"],
                dtype=bfloat16,
                device="npu",
            )
            if self.npu_layers and not self._use_cpu_lm_head
            else None
        )
        self._device_weights = {}
        self._bf16_weights = {}
        self._raw_weights = {}
        self._packed_layer_weights = {}
        self._packed_layer_gammas = {}
        self._decoder_weights = {}
        self._decoder_gammas = {}
        self._cpu_lm_head_f32 = None
        self._cpu_final_norm_f32 = None
        self._qwen_chunks = {}
        self._qwen_host_buffers = {}
        # Upload only the selected row for each token.  The previous SmolLM
        # path kept the complete embedding table on the NPU and launched a
        # slice kernel per token; that dispatch dominated the measured decode
        # latency despite the row being only a few KiB.
        self._embedding = None
        self._host_embedding = (
            self.model.raw("token_embedding")
            if self.npu_layers
            else None
        )
        # One persistent device buffer receives each token's embedding row.
        # Creating a fresh tensor per token opened a new XRT device handle and
        # buffer object on every decode step.
        self._embedding_buffer = (
            iron.zeros(self.hidden_size, dtype=bfloat16, device="npu")
            if self.npu_layers
            else None
        )
        # Row-split engine: every SmolLM decoder layer and the final RMSNorm in
        # one dispatch with six parallel weight streams (designs/engine.py).
        # It computes the same BF16 results as the chunked layer graphs.
        self._engine = None
        if (
            self._fused_smollm
            and self.npu_layers == self.layers
            and not self._use_quantized_decoder
            and self._use_cpu_lm_head
            and os.environ.get("HAWKPOINT_ENGINE", "1") != "0"
        ):
            # Imported lazily: the engine module builds its layout constants
            # from designs.engine, which is only meaningful with MLIR-AIE.
            from runtime import engine as engine_module

            if engine_module.supports(self.model.metadata):
                self._engine = engine_module.DecoderEngine(
                    self.model, self._rope_inv_freq
                )
        self.cpu_stage = (
            CPUDecoderStage(self.model, self.npu_layers, context_length)
            if (
                self.npu_layers < self.layers
                or (
                    self.model_family == "qwen2"
                    and self.npu_layers == self.layers
                )
            )
            else None
        )
        self._cpu_embedding = (
            np.asarray(
                self.model.raw("token_embedding"), dtype=np.float32
            ).reshape(
                self.model.metadata["vocab_size"], self.hidden_size
            )
            if self.npu_layers == 0
            else None
        )
        self.kv_cache = {}
        for first, count in self._smollm_chunks:
            self.kv_cache[first] = iron.zeros(
                (
                    count,
                    self.kv_heads,
                    2 * self.context_length * self.head_dim,
                ),
                dtype=bfloat16,
                device="npu",
            )
        self._qwen_cache = {}
        if self.model_family == "qwen2":
            for first in range(0, self.npu_layers, 2):
                count = min(2, self.npu_layers - first)
                self._qwen_cache[first] = iron.zeros(
                    (
                        count,
                        self.kv_heads,
                        2 * self.context_length * self.head_dim,
                    ),
                    dtype=bfloat16,
                    device="npu",
                )

    def _raw_bf16(self, name):
        if name not in self._raw_weights:
            array = np.asarray(self.model.raw(name), dtype=np.float32).astype(bfloat16)
            self._raw_weights[name] = iron.tensor(
                array, dtype=bfloat16, device="npu"
            )
        return self._raw_weights[name]

    def _quantized(self, name):
        if name not in self._device_weights:
            tensor = self.model.quantized(name)
            self._device_weights[name] = (
                iron.tensor(tensor.weight, dtype=np.int8, device="npu"),
                iron.tensor(tensor.scale, dtype=np.float32, device="npu"),
                tensor.shape,
            )
        return self._device_weights[name]

    def _bf16_projection(self, name):
        if name not in self._bf16_weights:
            self._bf16_weights[name] = iron.tensor(
                self.model.bf16_projection(name),
                dtype=bfloat16,
                device="npu",
            )
        return self._bf16_weights[name]

    def _project(self, activation, name, output=None):
        weight, scale, shape = self._quantized(name)
        rows, cols = shape
        if output is None:
            output = iron.zeros(rows, dtype=bfloat16, device="npu")
        project(
            weight,
            scale,
            activation,
            output,
            M=rows,
            K=cols,
            rows=self._projection_rows(cols),
        )
        return output

    def _project_bf16(self, activation, name, output=None):
        shape = self.model.tensors[name]["shape"]
        rows, cols = shape
        if output is None:
            output = iron.zeros(rows, dtype=bfloat16, device="npu")
        project_bf16(
            self._bf16_projection(name),
            activation,
            output,
            M=rows,
            K=cols,
            rows=self._bf16_projection_rows(cols),
        )
        return output

    def _norm_project(self, activation, gamma_name, projection_name):
        weight, scale, shape = self._quantized(projection_name)
        rows, cols = shape
        output = iron.zeros(rows, dtype=bfloat16, device="npu")
        norm_project(
            weight,
            scale,
            activation,
            self._raw_bf16(gamma_name),
            output,
            M=rows,
            K=cols,
            rows=self._projection_rows(cols),
        )
        return output

    def _project_residual(self, activation, residual, name):
        weight, scale, shape = self._quantized(name)
        rows, cols = shape
        output = iron.zeros(rows, dtype=bfloat16, device="npu")
        project_residual(
            weight,
            scale,
            activation,
            residual,
            output,
            M=rows,
            K=cols,
            rows=self._projection_rows(cols),
        )
        return output

    def _project_bf16_residual(self, activation, residual, name):
        shape = self.model.tensors[name]["shape"]
        rows, cols = shape
        output = iron.zeros(rows, dtype=bfloat16, device="npu")
        project_bf16_residual(
            self._bf16_projection(name),
            activation,
            residual,
            output,
            M=rows,
            K=cols,
            rows=self._bf16_projection_rows(cols),
        )
        return output

    @staticmethod
    def _projection_rows(input_columns):
        return 8 if input_columns > 2048 else 32

    @staticmethod
    def _bf16_projection_rows(input_columns):
        return 4 if input_columns > 2048 else 32

    def _slice(self, tensor, total_size, offset, size):
        output = iron.zeros(size, dtype=bfloat16, device="npu")
        slice_bf16(
            tensor,
            output,
            total_size=total_size,
            offset=offset,
            size=size,
        )
        return output

    def _embedding_for(self, token_id):
        if self.npu_layers:
            cached = self._qwen_embedding_cache.get(token_id)
            if cached is None:
                cached = np.asarray(
                    self._host_embedding[token_id], dtype=np.float32
                ).astype(bfloat16)
                self._qwen_embedding_cache[token_id] = cached
            else:
                self._qwen_embedding_cache.move_to_end(token_id)
            while len(self._qwen_embedding_cache) > self._qwen_embedding_cache_limit:
                self._qwen_embedding_cache.popitem(last=False)
            # The decoder graphs update the hidden state in place, so the
            # buffer is rewritten (and synced to the device) for every token.
            self._embedding_buffer[:] = cached
            return self._embedding_buffer
        raise RuntimeError("NPU embedding requested on a CPU-only decoder")

    def _rope_lut(self, position):
        cached = self._rope_lut_cache.get(position)
        if cached is not None:
            return cached
        angle = np.float32(position) * self._rope_inv_freq
        parts = [
            np.cos(angle),
            np.sin(angle),
            np.array([position, 0.0], np.float32),
        ]
        lut = np.concatenate(parts).astype(np.float32)
        cached = iron.tensor(lut.astype(bfloat16), dtype=bfloat16, device="npu")
        self._rope_lut_cache[position] = cached
        return cached

    def _record_phase(self, name, started):
        if self._profile_enabled:
            self._phase_totals[name] = self._phase_totals.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def _packed_layer(self, layer):
        if layer not in self._packed_layer_weights:
            prefix = f"layer{layer:02d}"

            packed = np.concatenate(
                [
                    np.asarray(
                        self.model.bf16_projection(f"{prefix}.qkv")
                    ).reshape(-1),
                    np.asarray(
                        self.model.bf16_projection(f"{prefix}.o_proj")
                    ).reshape(-1),
                    np.asarray(
                        self.model.bf16_projection(f"{prefix}.gate_up")
                    ).reshape(-1),
                    np.asarray(
                        self.model.bf16_projection(f"{prefix}.down_proj")
                    ).reshape(-1),
                ]
            ).astype(bfloat16)
            gammas = np.concatenate(
                [
                    np.asarray(self.model.raw(f"{prefix}.input_norm")),
                    np.asarray(self.model.raw(f"{prefix}.post_attn_norm")),
                ]
            ).astype(bfloat16)
            self._packed_layer_weights[layer] = iron.tensor(
                packed, dtype=bfloat16, device="npu"
            )
            self._packed_layer_gammas[layer] = iron.tensor(
                gammas, dtype=bfloat16, device="npu"
            )
        return (
            self._packed_layer_weights[layer],
            self._packed_layer_gammas[layer],
        )

    def _packed_decoder(self, first, count):
        key = (first, count)
        if key not in self._decoder_weights:
            projection_weights = {
                "qkv": [],
                "o_proj": [],
                "gate_up": [],
                "down_proj": [],
            }
            gammas = []
            for layer in range(first, first + count):
                prefix = f"layer{layer:02d}"
                for name in projection_weights:
                    if self._use_quantized_decoder:
                        tensor = self.model.quantized(f"{prefix}.{name}")
                        rows_per_block = 32 if name == "down_proj" else 64
                        weight = np.asarray(tensor.weight, dtype=np.int8)
                        scale = np.asarray(tensor.scale, dtype=np.float32)
                        if weight.shape[0] % rows_per_block:
                            raise RuntimeError(
                                f"{prefix}.{name} cannot be packed into "
                                f"{rows_per_block}-row decoder blocks"
                            )
                        blocks = []
                        for start in range(0, weight.shape[0], rows_per_block):
                            block_weight = np.ascontiguousarray(
                                weight[start : start + rows_per_block]
                            ).view(np.uint8).reshape(-1)
                            block_scale = np.ascontiguousarray(
                                scale[start : start + rows_per_block]
                            ).view(np.uint8).reshape(-1)
                            blocks.append(
                                np.concatenate((block_weight, block_scale))
                            )
                        projection_weights[name].append(
                            np.concatenate(blocks).astype(np.uint8, copy=False)
                        )
                    else:
                        weight = np.asarray(
                            self.model.bf16_projection(f"{prefix}.{name}")
                        ).reshape(-1)
                        if name == "gate_up":
                            # The parallel gate graph receives one physical
                            # block per split endpoint. Reorder each group of
                            # worker pairs from [w0a,w0b,w1a,w1b,...] to
                            # [w0a,w1a,...,w0b,w1b,...] so each worker gets a
                            # complete pair while the merge restores model
                            # row order. Other projections retain model order.
                            block = 32 * self.hidden_size
                            workers = 3
                            group = 2 * workers * block
                            if weight.size % group:
                                raise RuntimeError(
                                    f"{prefix}.{name} has an invalid block layout"
                                )
                            blocks = weight.reshape(-1, block)
                            groups = blocks.reshape(-1, workers, 2, block)
                            groups = groups.transpose(0, 2, 1, 3)
                            weight = np.ascontiguousarray(groups).reshape(-1)
                        projection_weights[name].append(weight)
                gammas.extend(
                    [
                        np.asarray(self.model.raw(f"{prefix}.input_norm")),
                        np.asarray(self.model.raw(f"{prefix}.post_attn_norm")),
                    ]
                )
            packed_dtype = np.uint8 if self._use_quantized_decoder else bfloat16
            self._decoder_weights[key] = iron.tensor(
                np.concatenate(
                    [
                        *projection_weights["qkv"],
                        *projection_weights["o_proj"],
                        *projection_weights["gate_up"],
                        *projection_weights["down_proj"],
                    ]
                ).astype(packed_dtype, copy=False),
                dtype=packed_dtype,
                device="npu",
            )
            self._decoder_gammas[key] = iron.tensor(
                np.concatenate(gammas).astype(bfloat16),
                dtype=bfloat16,
                device="npu",
            )
        return self._decoder_weights[key], self._decoder_gammas[key]

    def _cpu_lm_head_logits_values(self, values):
        if self._cpu_lm_head_f32 is None:
            self._cpu_lm_head_f32 = np.asarray(
                self.model.bf16_projection("lm_head"), dtype=np.float32
            )
        return self._cpu_lm_head_f32 @ np.asarray(
            values, dtype=np.float32
        )

    def _cpu_lm_head_logits(self, normalized):
        return self._cpu_lm_head_logits_values(
            np.asarray(normalized.numpy(), dtype=np.float32)
        )

    def _packed_qwen_chunk(self, first, count):
        key = (first, count)
        if key not in self._qwen_chunks:
            specs = {
                "qkv": (72, 16, 896),
                "o_proj": (56, 16, 896),
                "gate_up": (608, 16, 896),
                "down_proj": (224, 4, 4864),
            }
            projections = {name: [] for name in specs}
            params = []
            for layer in range(first, first + count):
                prefix = f"layer{layer:02d}"
                for name in projections:
                    projections[name].append(
                        np.asarray(
                            self.model.bf16_projection(f"{prefix}.{name}")
                        ).reshape(-1)
                    )
                bias = np.asarray(
                    self.model.raw(f"{prefix}.qkv_bias"),
                    dtype=np.float32,
                )
                bias_high = bias.astype(bfloat16)
                bias_low = (
                    bias - bias_high.astype(np.float32)
                ).astype(bfloat16)
                params.extend(
                    [
                        np.asarray(
                            self.model.raw(f"{prefix}.input_norm")
                        ).astype(bfloat16),
                        np.asarray(
                            self.model.raw(f"{prefix}.post_attn_norm")
                        ).astype(bfloat16),
                        bias_high,
                        bias_low,
                    ]
                )

            packed = []
            for name, (blocks, rows, columns) in specs.items():
                values = np.stack(projections[name])
                if name == "down_proj":
                    values = values.reshape(
                        count, blocks * rows, 19, 256
                    ).transpose(1, 0, 2, 3)
                else:
                    values = values.reshape(
                        count, blocks, rows, columns
                    ).transpose(1, 0, 2, 3)
                packed.append(values.reshape(-1))
            host_weights = np.concatenate(packed).astype(bfloat16)
            host_params = np.concatenate(params).astype(bfloat16)
            self._qwen_host_buffers[key] = (host_weights, host_params)
            self._qwen_chunks[key] = (
                iron.tensor(
                    host_weights, dtype=bfloat16, device="npu"
                ),
                iron.tensor(
                    host_params, dtype=bfloat16, device="npu"
                ),
            )
        return self._qwen_chunks[key]

    def _layer(self, hidden, layer, position, rope_lut):
        if self.model_family == "qwen2":
            return self._layer_unfused(hidden, layer, position, rope_lut)
        weights, gammas = self._packed_layer(layer)
        decoder_layer(
            weights,
            gammas,
            rope_lut,
            hidden,
            self.kv_cache[layer],
        )
        return hidden

    def _layer_unfused(self, hidden, layer, position, rope_lut):
        prefix = f"layer{layer:02d}"
        normalized = iron.zeros(
            self.hidden_size, dtype=bfloat16, device="npu"
        )
        rmsnorm(
            hidden,
            self._raw_bf16(f"{prefix}.input_norm"),
            normalized,
            size=self.hidden_size,
            epsilon=self.rms_norm_eps,
        )
        qkv = self._project_bf16(normalized, f"{prefix}.qkv")
        biased_qkv = iron.zeros(
            (self.q_heads + 2 * self.kv_heads) * self.head_dim,
            dtype=bfloat16,
            device="npu",
        )
        residual_add(
            qkv,
            self._raw_bf16(f"{prefix}.qkv_bias"),
            biased_qkv,
            size=(self.q_heads + 2 * self.kv_heads) * self.head_dim,
        )
        packed_qkv = iron.zeros(
            (
                self.kv_heads,
                (self.q_per_kv + 2) * self.head_dim,
            ),
            dtype=bfloat16,
            device="npu",
        )
        qkv_rope(
            biased_qkv,
            rope_lut,
            packed_qkv,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        attended = iron.zeros(
            (self.kv_heads, self.q_per_kv * self.head_dim),
            dtype=bfloat16,
            device="npu",
        )
        attention_block(
            packed_qkv,
            self.kv_cache[layer],
            attended,
            position=position,
            kv_heads=self.kv_heads,
            q_per_kv=self.q_per_kv,
            head_dim=self.head_dim,
        )
        after_attention = self._project_bf16_residual(
            attended, hidden, f"{prefix}.o_proj"
        )
        post_normalized = iron.zeros(
            self.hidden_size, dtype=bfloat16, device="npu"
        )
        rmsnorm(
            after_attention,
            self._raw_bf16(f"{prefix}.post_attn_norm"),
            post_normalized,
            size=self.hidden_size,
            epsilon=self.rms_norm_eps,
        )
        gate_up = self._project_bf16(
            post_normalized, f"{prefix}.gate_up"
        )
        mlp_activation = iron.zeros(
            self.intermediate_size, dtype=bfloat16, device="npu"
        )
        swiglu(
            gate_up,
            mlp_activation,
            size=self.intermediate_size,
        )
        return self._project_bf16_residual(
            mlp_activation, after_attention, f"{prefix}.down_proj"
        )

    @staticmethod
    def _decode_result(logits, start, diagnostics, select=None):
        logits_f32 = np.asarray(logits, dtype=np.float32)
        next_token = (
            int(np.argmax(logits_f32)) if select is None else int(select(logits_f32))
        )
        elapsed = time.perf_counter() - start
        if not diagnostics:
            return next_token, elapsed
        top_indices = np.argpartition(logits_f32, -5)[-5:]
        top_indices = top_indices[
            np.argsort(logits_f32[top_indices])[::-1]
        ]
        return next_token, elapsed, {
            "top_token_ids": [int(index) for index in top_indices],
            "top_logits": [
                float(logits_f32[index]) for index in top_indices
            ],
            "top_logit_margin": float(
                logits_f32[top_indices[0]] - logits_f32[top_indices[1]]
            ),
        }

    def decode_token(
        self,
        token_id,
        position,
        *,
        diagnostics=False,
        select=None,
        compute_logits=True,
    ):
        """Run one decode step and return ``(next_token, elapsed_seconds)``.

        ``select`` overrides token selection with a callable taking the float32
        logit vector and returning a token id. It defaults to ``argmax`` so the
        greedy reference path and the exact-token release gates are unchanged.
        ``compute_logits=False`` is used for non-final prefill positions: the
        decoder and KV cache still advance, but the otherwise discarded final
        normalization and 49k-row LM head are skipped.
        """
        if not 0 <= position < self.context_length:
            raise ValueError("position outside the 64-token context")
        # This step overwrites the cache at ``position``; later positions were
        # computed from a different history and are no longer reusable.
        del self._cached_prefix[position:]
        start = time.perf_counter()
        if getattr(self, "_engine", None) is not None:
            phase = time.perf_counter()
            row = self._qwen_embedding_cache.get(token_id)
            if row is None:
                row = np.asarray(
                    self._host_embedding[token_id], dtype=np.float32
                ).astype(bfloat16)
                self._qwen_embedding_cache[token_id] = row
                while (
                    len(self._qwen_embedding_cache)
                    > self._qwen_embedding_cache_limit
                ):
                    self._qwen_embedding_cache.popitem(last=False)
            normalized = self._engine.step(row, position)
            self._record_phase("npu_engine", phase)
            if not compute_logits:
                return None, time.perf_counter() - start
            phase = time.perf_counter()
            logits = self._cpu_lm_head_logits_values(
                np.asarray(normalized, dtype=np.float32)
            )
            self._record_phase("cpu_lm_head", phase)
            return self._decode_result(logits, start, diagnostics, select)
        if self.npu_layers:
            phase = time.perf_counter()
            hidden = self._embedding_for(token_id)
            self._record_phase("embedding", phase)
            phase = time.perf_counter()
            rope_lut = self._rope_lut(position)
            self._record_phase("rope_lut", phase)
            phase = time.perf_counter()
            if self.model_family == "qwen2":
                for first in range(0, self.npu_layers, 2):
                    count = min(2, self.npu_layers - first)
                    weights, params = self._packed_qwen_chunk(
                        first, count
                    )
                    qwen_decoder(
                        weights,
                        params,
                        rope_lut,
                        hidden,
                        self._qwen_cache[first],
                        layers=count,
                    )
            elif self._fused_smollm:
                for first, count in self._smollm_chunks:
                    weights, gammas = self._packed_decoder(first, count)
                    decoder_layer(
                        weights,
                        gammas,
                        rope_lut,
                        hidden,
                        self.kv_cache[first],
                        layers=count,
                        quantized=self._use_quantized_decoder,
                    )
            else:
                for layer in range(self.npu_layers):
                    hidden = self._layer(
                        hidden, layer, position, rope_lut
                    )
            self._record_phase("npu_decoder", phase)
        else:
            phase = time.perf_counter()
            hidden = _bf16_cpu(self._cpu_embedding[token_id])
            self._record_phase("cpu_embedding", phase)
        if self.cpu_stage is not None:
            phase = time.perf_counter()
            if self.npu_layers:
                hidden = hidden.numpy().astype(bfloat16)
            for layer in range(self.npu_layers, self.layers):
                hidden = self.cpu_stage.layer(hidden, layer, position)
            self._record_phase("cpu_decoder", phase)
            if not compute_logits:
                return None, time.perf_counter() - start
            phase = time.perf_counter()
            logits = self.cpu_stage.logits(hidden)
            self._record_phase("cpu_lm_head", phase)
            return self._decode_result(logits, start, diagnostics, select)
        if not compute_logits:
            return None, time.perf_counter() - start
        phase = time.perf_counter()
        normalized = self._final_norm_buffer
        if self._use_cpu_final_norm:
            values = np.asarray(hidden.numpy(), dtype=np.float32)
            if self._cpu_final_norm_f32 is None:
                self._cpu_final_norm_f32 = np.asarray(
                    self.model.raw("final_norm"), dtype=np.float32
                )
            inv = np.float32(
                1.0
                / np.sqrt(
                    np.mean(values * values, dtype=np.float32)
                    + np.float32(self.rms_norm_eps)
                )
            )
            normalized_values = values * inv * self._cpu_final_norm_f32
            logits = self._cpu_lm_head_logits_values(normalized_values)
            self._record_phase("cpu_lm_head", phase)
            return self._decode_result(logits, start, diagnostics, select)
        if normalized is None:
            normalized = iron.zeros(
                self.hidden_size, dtype=bfloat16, device="npu"
            )
        rmsnorm(
            hidden,
            self._raw_bf16("final_norm"),
            normalized,
            size=self.hidden_size,
            epsilon=self.rms_norm_eps,
        )
        if self._use_cpu_lm_head:
            logits = self._cpu_lm_head_logits(normalized)
            self._record_phase("cpu_lm_head", phase)
            return self._decode_result(logits, start, diagnostics, select)
        logits = (
            self._project_bf16(
                normalized, "lm_head", output=self._lm_head_buffer
            )
            if self.model_family == "qwen2"
            else self._project(
                normalized, "lm_head", output=self._lm_head_buffer
            )
        )
        self._record_phase("npu_lm_head", phase)
        return self._decode_result(logits.numpy(), start, diagnostics, select)

    def warmup(self):
        """Compile and populate the hot path before the first request."""
        if not self.npu_layers:
            return
        self.decode_token(self.tokenizer.eos_id, 0)
        if self.model_family == "qwen2":
            for first in range(0, self.npu_layers, 2):
                count = min(2, self.npu_layers - first)
                self._qwen_cache[first] = iron.zeros(
                    (
                        count,
                        self.kv_heads,
                        2 * self.context_length * self.head_dim,
                    ),
                    dtype=bfloat16,
                    device="npu",
                )
        else:
            if self._fused_smollm:
                self.kv_cache = {
                    first: iron.zeros(
                        (
                            count,
                            self.kv_heads,
                            2 * self.context_length * self.head_dim,
                        ),
                        dtype=bfloat16,
                        device="npu",
                    )
                    for first, count in self._smollm_chunks
                }
            else:
                self.kv_cache = [
                    iron.zeros(
                        (
                            1,
                            self.kv_heads,
                            2 * self.context_length * self.head_dim,
                        ),
                        dtype=bfloat16,
                        device="npu",
                    )
                    for _ in range(self.npu_layers)
                ]
        if self.cpu_stage is not None:
            for cache in self.cpu_stage.key_cache.values():
                cache.fill(0)
            for cache in self.cpu_stage.value_cache.values():
                cache.fill(0)
        if getattr(self, "_engine", None) is not None:
            self._engine.reset()
        self._cached_prefix.clear()

    def reset_prefix_cache(self):
        """Forget the reusable prompt prefix so the next prompt is fully prefilled."""
        self._cached_prefix.clear()

    def _reusable_prefix(self, prompt_ids):
        """Return how many leading prompt positions can be reused from the cache.

        The final prompt position is always recomputed because its logits
        select the first generated token.
        """
        if not self._prefix_cache_enabled:
            return 0
        shared = 0
        for cached, token in zip(self._cached_prefix, prompt_ids):
            if cached != token:
                break
            shared += 1
        return min(shared, len(prompt_ids) - 1)

    def _decode_step(self, token_id, position, **kwargs):
        """Run ``decode_token`` and record the token whose cache it filled."""
        del self._cached_prefix[position:]
        result = self.decode_token(token_id, position, **kwargs)
        if len(self._cached_prefix) == position:
            self._cached_prefix.append(int(token_id))
        return result

    def _logprob_entry(self, logits, token_id, top_logprobs):
        """Describe one emitted token in the OpenAI ``logprobs`` format.

        Values are taken from the model's raw distribution, before penalties,
        temperature, and top-k/top-p filtering.
        """
        values = log_softmax(logits)

        def describe(index):
            return {
                "token": self.tokenizer.decode([int(index)]),
                "logprob": float(values[index]),
                "bytes": list(self.tokenizer.token_bytes(int(index))),
            }

        entry = describe(token_id)
        top = []
        if top_logprobs:
            count = min(top_logprobs, values.shape[0])
            indices = np.argpartition(values, -count)[-count:]
            indices = indices[np.argsort(values[indices])[::-1]]
            top = [describe(index) for index in indices]
        entry["top_logprobs"] = top
        return entry

    def generate_messages(
        self,
        messages,
        max_new_tokens=16,
        sampling=None,
        *,
        stop=None,
        logprobs=None,
    ):
        """Generate a response for an OpenAI-style list of chat messages.

        The hardware attention cache is fixed at 64 tokens. When a conversation
        grows beyond that window, it is shortened at turn boundaries so the
        newest message stays whole for as long as possible (see
        :meth:`SmolLMTokenizer.encode_chat_window`).

        ``sampling`` accepts a :class:`~runtime.sampling.SamplingParams` (or a
        transport dict) and defaults to greedy decoding. ``stop`` is a string
        or up to four strings that end generation without being emitted.
        ``logprobs`` is ``None`` (disabled) or the number of alternatives to
        report per token; when enabled, every yielded chunk is a
        ``(text, stats, logprob_entries)`` triple instead of ``(text, stats)``.
        """
        if not 0 < max_new_tokens < self.context_length:
            raise ValueError(
                f"max_new_tokens must be between 1 and {self.context_length - 1}"
            )
        if sampling is None:
            params = GREEDY
        elif isinstance(sampling, SamplingParams):
            params = sampling
        else:
            params = SamplingParams.from_dict(sampling)
        stop_matcher = StopMatcher(normalize_stop(stop))
        top_logprobs = normalize_logprobs(logprobs)
        sampler = Sampler(params)
        if getattr(self, "_profile_enabled", False):
            self._phase_totals = {}
        prompt_ids, truncation = self.tokenizer.encode_chat_window(
            messages, self.context_length - max_new_tokens
        )
        # Penalties are scored against the prompt as well as the generated text,
        # matching llama.cpp and vLLM.
        history = list(prompt_ids)
        # Log-probability entries wait here, keyed by the byte offset where
        # their token starts in the generated text, until that text is emitted.
        # Entries for text a stop sequence discards are never reported.
        pending_logprobs = []
        selected_entry = {}
        generated_bytes = 0
        emitted_bytes = 0

        def choose(logits):
            token = (
                int(np.argmax(logits))
                if sampler.greedy
                else sampler.select(logits, history)
            )
            if top_logprobs is not None:
                selected_entry["entry"] = self._logprob_entry(
                    logits, token, top_logprobs
                )
            return token

        # Greedy requests without log probabilities keep the historical
        # argmax-inside-decode_token path used by the release gates.
        select = None if sampler.greedy and top_logprobs is None else choose

        def release(text, final=False):
            """Return the entries whose tokens contributed to emitted text."""
            nonlocal emitted_bytes
            emitted_bytes += len(text.encode("utf-8"))
            if final and stop_matcher.matched is None:
                ready = len(pending_logprobs)
            else:
                ready = 0
                while (
                    ready < len(pending_logprobs)
                    and pending_logprobs[ready][0] < emitted_bytes
                ):
                    ready += 1
            entries = [entry for _, entry in pending_logprobs[:ready]]
            del pending_logprobs[:ready]
            return entries

        def chunk(text, entries):
            if top_logprobs is None:
                return text, None
            return text, None, entries

        reused = self._reusable_prefix(prompt_ids)
        timings = []
        next_token = None
        last_prompt_index = len(prompt_ids) - 1
        for index in range(reused, len(prompt_ids)):
            # Only the final prompt position produces a token that is used, so
            # the sampler (and its random stream) is engaged exactly once per
            # emitted token instead of once per ingested position.
            next_token, elapsed = self._decode_step(
                prompt_ids[index],
                index,
                select=select if index == last_prompt_index else None,
                compute_logits=index == last_prompt_index,
            )
            timings.append(elapsed)
        prefill_steps = len(timings)
        position = len(prompt_ids)
        detokenizer = StreamDetokenizer(self.tokenizer)
        generated = []
        finish_reason = "length"
        while len(generated) < max_new_tokens:
            # The end-of-turn token is not part of the returned content, so
            # its log probability is dropped with it.
            entry = selected_entry.pop("entry", None)
            if next_token == self.tokenizer.eos_id:
                finish_reason = "stop"
                break
            generated.append(next_token)
            history.append(next_token)
            if entry is not None:
                pending_logprobs.append((generated_bytes, entry))
                generated_bytes += len(self.tokenizer.token_bytes(next_token))
            text = stop_matcher.feed(detokenizer.push(next_token))
            if stop_matcher.matched is not None:
                finish_reason = "stop"
                break
            if text:
                yield chunk(text, release(text))
            # The caller requested exactly this many tokens. Do not run one
            # more full decoder/LM-head step merely to compute a discarded
            # successor token.
            if len(generated) >= max_new_tokens:
                break
            next_token, elapsed = self._decode_step(
                next_token, position, select=select
            )
            timings.append(elapsed)
            position += 1
        if stop_matcher.matched is None:
            text = stop_matcher.feed(detokenizer.flush())
            if stop_matcher.matched is not None:
                finish_reason = "stop"
            text += stop_matcher.flush()
        entries = release(text, final=True) if top_logprobs is not None else []
        if text or entries:
            yield chunk(text, entries)
        prefill_seconds = sum(timings[:prefill_steps])
        decode_seconds = sum(timings[prefill_steps:])
        decode_steps = len(timings) - prefill_steps
        stats = {
            "prompt_tokens": len(prompt_ids),
            "cached_prompt_tokens": reused,
            "prompt_truncation": truncation,
            "generated_tokens": len(generated),
            "generated_token_ids": generated,
            "npu_layers": self.npu_layers,
            "cpu_layers": self.layers - self.npu_layers,
            "finish_reason": finish_reason,
            "stop_sequence": stop_matcher.matched,
            "ttft_seconds": prefill_seconds,
            "prefill_tokens_per_second": (
                prefill_steps / max(1e-9, prefill_seconds)
            ),
            "decode_steps": decode_steps,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                decode_steps / max(1e-9, decode_seconds)
            ),
            "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "sampling": sampler.stats(),
        }
        if getattr(self, "_profile_enabled", False):
            stats["phase_ms"] = {
                name: round(seconds * 1000.0, 3)
                for name, seconds in self._phase_totals.items()
            }
        if top_logprobs is None:
            yield "", stats
        else:
            yield "", stats, []

    def generate(
        self,
        prompt,
        max_new_tokens=32,
        sampling=None,
        *,
        stop=None,
        logprobs=None,
    ):
        """Backward-compatible single-prompt generation."""
        yield from self.generate_messages(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            stop=stop,
            logprobs=logprobs,
        )

    def close(self):
        """Deterministically release every NPU/XRT resource this decoder holds.

        Model switching reuses one physical NPU. Its XRT hardware contexts are a
        driver-level, system-wide constrained resource (six on npu1). Relying on
        garbage collection to release them races the next model's context
        creation and, over a long model-switching run, exhausts the driver pool:
        ``DRM_IOCTL_AMDXDNA_CREATE_HWCTX`` then fails with ``-110`` and the
        kernel logs ``aie2_alloc_resource failed``. This drops every
        device-resident buffer object and releases the cached hardware contexts
        so the next model starts from a clean allocation. It is idempotent.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for attr in (
            "_device_weights",
            "_bf16_weights",
            "_raw_weights",
            "_packed_layer_weights",
            "_packed_layer_gammas",
            "_decoder_weights",
            "_decoder_gammas",
            "_qwen_chunks",
            "_qwen_host_buffers",
            "_qwen_embedding_cache",
            "_rope_lut_cache",
            "_qwen_cache",
        ):
            container = getattr(self, attr, None)
            if isinstance(container, dict):
                container.clear()
        self.kv_cache = []
        self._cached_prefix = []
        self._embedding_buffer = None
        self._decoder_weights = {}
        self._decoder_gammas = {}
        self._cpu_lm_head_f32 = None
        self._cpu_final_norm_f32 = None
        self._embedding = None
        self._final_norm_buffer = None
        self._lm_head_buffer = None
        engine = getattr(self, "_engine", None)
        if engine is not None:
            engine.close()
        self._engine = None
        # Release the cached, driver-level hardware contexts before another
        # model loads. Only meaningful when this decoder actually touched the
        # NPU; the CPU-only path (npu_layers == 0) never created a context.
        if getattr(self, "npu_layers", 0):
            try:
                from aie.utils import cleanup_npu_runtime

                cleanup_npu_runtime()
            except Exception:
                logging.exception(
                    "cleanup_npu_runtime failed while closing NPUDecoder"
                )
        gc.collect()
