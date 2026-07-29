import json
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SYSTEM = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)


class SmolLMTokenizer:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        self.eos_token = config.get("eos_token", "<|im_end|>")
        self.eos_id = self._tokenizer.token_to_id(self.eos_token)

    @staticmethod
    def format_chat(messages, add_generation_prompt=True):
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": DEFAULT_SYSTEM}, *messages]
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
