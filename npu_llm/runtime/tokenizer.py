import json
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SYSTEM = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)


def _eos_token_text(value):
    """Return the literal EOS token from a ``tokenizer_config.json`` entry.

    Hugging Face writes ``eos_token`` either as a plain string or as a
    serialized ``AddedToken`` object; both forms appear across the supported
    checkpoints.
    """
    if isinstance(value, dict):
        value = value.get("content")
    if isinstance(value, str) and value:
        return value
    return "<|im_end|>"


class SmolLMTokenizer:
    def __init__(self, model_dir, default_system=DEFAULT_SYSTEM):
        model_dir = Path(model_dir)
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        self.eos_token = _eos_token_text(config.get("eos_token"))
        self.eos_id = self._tokenizer.token_to_id(self.eos_token)
        if self.eos_id is None:
            # Without a real EOS id every generation would run to the token
            # limit and report finish_reason="length", so fail loudly instead.
            raise ValueError(
                f"EOS token {self.eos_token!r} is not in the vocabulary of "
                f"{model_dir}"
            )
        self.default_system = default_system

    def format_chat(self, messages, add_generation_prompt=True):
        if not messages or messages[0]["role"] != "system":
            messages = [
                {"role": "system", "content": self.default_system},
                *messages,
            ]
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode_chat(self, messages):
        return self._tokenizer.encode(self.format_chat(messages)).ids

    def decode(self, token_ids, skip_special_tokens=False):
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
