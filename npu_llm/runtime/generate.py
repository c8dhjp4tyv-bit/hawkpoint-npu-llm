"""Host orchestration for the supported NPU-only decode graphs."""

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
from designs.rmsnorm import rmsnorm
from designs.tensor_copy import slice_bf16
from runtime.model import XDNA1Model
from runtime.tokenizer import SmolLMTokenizer


class NPUDecoder:
    def __init__(self, model_dir, context_length=64):
        if context_length != 64:
            raise ValueError("the current NPU attention kernel has a fixed 64-token cache")
        set_current_device(from_name("npu", n_cols=4))
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
        self.rms_norm_eps = float(
            self.model.metadata.get("rms_norm_eps", 1e-5)
        )
        self.layers = self.model.metadata["layers"]
        self._device_weights = {}
        self._bf16_weights = {}
        self._raw_weights = {}
        self._packed_layer_weights = {}
        self._packed_layer_gammas = {}
        self._decoder_weights = None
        self._decoder_gammas = None
        self._embedding = self._raw_bf16("token_embedding")
        self.kv_cache = [
            iron.zeros(
                (1, self.kv_heads, 2 * self.context_length * self.head_dim),
                dtype=bfloat16,
                device="npu",
            )
            for _ in range(self.layers)
        ]

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

    def _project(self, activation, name):
        weight, scale, shape = self._quantized(name)
        rows, cols = shape
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

    def _project_bf16(self, activation, name):
        shape = self.model.tensors[name]["shape"]
        rows, cols = shape
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
        return self._slice(
            self._embedding,
            self.model.metadata["vocab_size"] * self.hidden_size,
            token_id * self.hidden_size,
            self.hidden_size,
        )

    def _rope_lut(self, position):
        rope_theta = float(self.model.metadata.get("rope_theta", 100000.0))
        inv_freq = 1.0 / (
            rope_theta
            ** (np.arange(0, self.head_dim, 2) / self.head_dim)
        )
        angle = position * inv_freq
        parts = [np.cos(angle), np.sin(angle)]
        if self.model_family != "qwen2":
            parts.append(np.array([position, 0.0], np.float32))
        lut = np.concatenate(parts).astype(np.float32)
        return iron.tensor(lut.astype(bfloat16), dtype=bfloat16, device="npu")

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

    def _packed_decoder(self):
        if self._decoder_weights is None:
            projection_weights = {
                "qkv": [],
                "o_proj": [],
                "gate_up": [],
                "down_proj": [],
            }
            gammas = []
            for layer in range(self.layers):
                prefix = f"layer{layer:02d}"
                for name in projection_weights:
                    projection_weights[name].append(
                        np.asarray(
                            self.model.bf16_projection(f"{prefix}.{name}")
                        ).reshape(-1)
                    )
                gammas.extend(
                    [
                        np.asarray(self.model.raw(f"{prefix}.input_norm")),
                        np.asarray(self.model.raw(f"{prefix}.post_attn_norm")),
                    ]
                )
            self._decoder_weights = iron.tensor(
                np.concatenate(
                    [
                        *projection_weights["qkv"],
                        *projection_weights["o_proj"],
                        *projection_weights["gate_up"],
                        *projection_weights["down_proj"],
                    ]
                ).astype(bfloat16),
                dtype=bfloat16,
                device="npu",
            )
            self._decoder_gammas = iron.tensor(
                np.concatenate(gammas).astype(bfloat16),
                dtype=bfloat16,
                device="npu",
            )
        return self._decoder_weights, self._decoder_gammas

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

    def decode_token(self, token_id, position):
        if not 0 <= position < self.context_length:
            raise ValueError("position outside the 64-token context")
        start = time.perf_counter()
        hidden = self._embedding_for(token_id)
        rope_lut = self._rope_lut(position)
        for layer in range(self.layers):
            hidden = self._layer(hidden, layer, position, rope_lut)
        normalized = iron.zeros(self.hidden_size, dtype=bfloat16, device="npu")
        rmsnorm(
            hidden,
            self._raw_bf16("final_norm"),
            normalized,
            size=self.hidden_size,
            epsilon=self.rms_norm_eps,
        )
        logits = (
            self._project_bf16(normalized, "lm_head")
            if self.model_family == "qwen2"
            else self._project(normalized, "lm_head")
        )
        next_token = int(np.argmax(logits.numpy().astype(np.float32)))
        return next_token, time.perf_counter() - start

    def generate_messages(self, messages, max_new_tokens=16):
        """Generate a response for an OpenAI-style list of chat messages.

        The hardware attention cache is fixed at 64 tokens. When a conversation
        grows beyond that window, the newest prompt tokens are retained.
        """
        if not 0 < max_new_tokens < self.context_length:
            raise ValueError(
                f"max_new_tokens must be between 1 and {self.context_length - 1}"
            )
        prompt_ids = self.tokenizer.encode_chat(messages)
        if len(prompt_ids) + max_new_tokens > self.context_length:
            prompt_ids = prompt_ids[-(self.context_length - max_new_tokens) :]
        timings = []
        next_token = None
        position = 0
        for token in prompt_ids:
            next_token, elapsed = self.decode_token(token, position)
            timings.append(elapsed)
            position += 1
        generated = []
        while len(generated) < max_new_tokens and next_token != self.tokenizer.eos_id:
            generated.append(next_token)
            yield self.tokenizer.decode([next_token]), None
            next_token, elapsed = self.decode_token(next_token, position)
            timings.append(elapsed)
            position += 1
        stats = {
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "finish_reason": (
                "stop" if next_token == self.tokenizer.eos_id else "length"
            ),
            "ttft_seconds": sum(timings[: len(prompt_ids)]),
            "decode_tokens_per_second": (
                max(0, len(timings) - len(prompt_ids))
                / max(1e-9, sum(timings[len(prompt_ids) :]))
            ),
            "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        yield "", stats

    def generate(self, prompt, max_new_tokens=32):
        """Backward-compatible single-prompt generation."""
        yield from self.generate_messages(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
        )
