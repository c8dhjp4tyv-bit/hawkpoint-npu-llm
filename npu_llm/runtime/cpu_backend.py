"""NumPy decoder layers used when only part of a model is offloaded to XDNA1."""

import numpy as np
from ml_dtypes import bfloat16


def _bf16(array):
    return np.asarray(array, dtype=np.float32).astype(bfloat16)


class CPUDecoderStage:
    """Run a contiguous decoder suffix and its KV cache on the CPU."""

    def __init__(self, model, first_layer, context_length):
        self.model = model
        self.metadata = model.metadata
        self.first_layer = first_layer
        self.context_length = context_length
        self.hidden_size = self.metadata["hidden_size"]
        self.intermediate_size = self.metadata["intermediate_size"]
        self.q_heads = self.metadata["attention_heads"]
        self.kv_heads = self.metadata["kv_heads"]
        self.head_dim = self.metadata["head_dim"]
        self.q_per_kv = self.q_heads // self.kv_heads
        self.rope_theta = float(self.metadata.get("rope_theta", 10000.0))
        self.epsilon = float(self.metadata.get("rms_norm_eps", 1e-5))
        self._weights = {}
        self._raw = {}
        self._lm_head_f32 = None
        self.key_cache = {
            layer: np.zeros(
                (self.kv_heads, context_length, self.head_dim),
                dtype=bfloat16,
            )
            for layer in range(first_layer, self.metadata["layers"])
        }
        self.value_cache = {
            layer: np.zeros(
                (self.kv_heads, context_length, self.head_dim),
                dtype=bfloat16,
            )
            for layer in range(first_layer, self.metadata["layers"])
        }

    def raw(self, name):
        if name not in self._raw:
            self._raw[name] = np.asarray(
                self.model.raw(name), dtype=np.float32
            )
        return self._raw[name]

    def weight(self, name):
        if name not in self._weights:
            self._weights[name] = self.model.bf16_projection(name)
        return self._weights[name]

    def project(self, name, activation):
        result = (
            np.asarray(self.weight(name), dtype=np.float32)
            @ np.asarray(_bf16(activation), dtype=np.float32)
        )
        return _bf16(result)

    def norm(self, activation, gamma_name):
        value = np.asarray(_bf16(activation), dtype=np.float32)
        inverse_rms = 1.0 / np.sqrt(
            np.mean(value * value, dtype=np.float32) + self.epsilon
        )
        return _bf16(value * inverse_rms * self.raw(gamma_name))

    def _rotate(self, heads, position):
        heads = np.asarray(heads, dtype=np.float32)
        half = self.head_dim // 2
        inverse_frequency = 1.0 / (
            self.rope_theta
            ** (np.arange(half, dtype=np.float32) / half)
        )
        angle = position * inverse_frequency
        cosine = np.cos(angle)
        sine = np.sin(angle)
        first, second = heads[..., :half], heads[..., half:]
        return _bf16(
            np.concatenate(
                [
                    first * cosine - second * sine,
                    second * cosine + first * sine,
                ],
                axis=-1,
            )
        )

    def _attention(self, qkv, layer, position):
        q_size = self.q_heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        query = qkv[:q_size].reshape(self.q_heads, self.head_dim)
        key = qkv[q_size : q_size + kv_size].reshape(
            self.kv_heads, self.head_dim
        )
        value = qkv[q_size + kv_size :].reshape(
            self.kv_heads, self.head_dim
        )
        query = self._rotate(query, position)
        key = self._rotate(key, position)
        self.key_cache[layer][:, position] = key
        self.value_cache[layer][:, position] = value

        output = np.empty((self.q_heads, self.head_dim), dtype=bfloat16)
        scale = self.head_dim**-0.5
        for kv_head in range(self.kv_heads):
            keys = np.asarray(
                self.key_cache[layer][kv_head, : position + 1],
                dtype=np.float32,
            )
            values = np.asarray(
                self.value_cache[layer][kv_head, : position + 1],
                dtype=np.float32,
            )
            for group_head in range(self.q_per_kv):
                q_head = kv_head * self.q_per_kv + group_head
                scores = (
                    keys
                    @ np.asarray(query[q_head], dtype=np.float32)
                ) * scale
                scores -= np.max(scores)
                probabilities = np.exp(scores)
                probabilities /= np.sum(probabilities)
                output[q_head] = _bf16(probabilities @ values)
        return output.reshape(-1)

    def layer(self, hidden, layer, position):
        prefix = f"layer{layer:02d}"
        normalized = self.norm(hidden, f"{prefix}.input_norm")
        bias_name = f"{prefix}.qkv_bias"
        qkv_bias = (
            self.raw(bias_name)
            if bias_name in self.model.tensors
            else 0.0
        )
        qkv = _bf16(
            np.asarray(self.project(f"{prefix}.qkv", normalized), np.float32)
            + qkv_bias
        )
        attended = self._attention(qkv, layer, position)
        after_attention = _bf16(
            np.asarray(
                self.project(f"{prefix}.o_proj", attended), np.float32
            )
            + np.asarray(hidden, np.float32)
        )
        post_normalized = self.norm(
            after_attention, f"{prefix}.post_attn_norm"
        )
        gate_up = np.asarray(
            self.project(f"{prefix}.gate_up", post_normalized),
            dtype=np.float32,
        )
        gate, up = np.split(gate_up, 2)
        activation = _bf16(
            (
                gate
                / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
            )
            * up
        )
        return _bf16(
            np.asarray(
                self.project(f"{prefix}.down_proj", activation), np.float32
            )
            + np.asarray(after_attention, np.float32)
        )

    def logits(self, hidden):
        normalized = self.norm(hidden, "final_norm")
        if self._lm_head_f32 is None:
            self._lm_head_f32 = np.asarray(
                self.weight("lm_head"), dtype=np.float32
            )
        return self._lm_head_f32 @ np.asarray(
            _bf16(normalized), dtype=np.float32
        )
