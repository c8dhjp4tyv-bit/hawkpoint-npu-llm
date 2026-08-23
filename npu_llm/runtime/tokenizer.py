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

    def format_chat(
        self,
        messages,
        add_generation_prompt=True,
        include_default_system=True,
    ):
        if include_default_system and (
            not messages or messages[0]["role"] != "system"
        ):
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

    def encode_chat(self, messages, include_default_system=True):
        return self._tokenizer.encode(
            self.format_chat(
                messages, include_default_system=include_default_system
            )
        ).ids

    def encode_chat_within(self, messages, budget):
        """Encode a conversation that fits ``budget`` tokens, by whole turns.

        Slicing the encoded ChatML stream would routinely cut through an
        ``<|im_start|>``, a role marker, or an ``<|im_end|>`` at this context
        size, leaving the model with a malformed prompt. Instead, drop whole
        turns oldest-first, so the retained prompt always begins on a message
        boundary. The escalation order is:

        1. all messages;
        2. the system message plus the newest turns that fit;
        3. the newest turn alone, without the system message -- an actual
           question is worth more of the window than the identity preamble;
        4. the newest turn with its content shortened from the front.

        Only the last step loses part of a message, and it still re-emits the
        surrounding markers, so the result is always well-formed ChatML.
        """
        if budget <= 0:
            raise ValueError("budget must be a positive number of tokens")
        system, turns = self._split_system(messages)
        for start in _turn_starts(turns):
            ids = self.encode_chat([*system, *turns[start:]])
            if len(ids) <= budget:
                return ids
        newest = turns[-1:]
        if system and newest:
            ids = self.encode_chat(newest, include_default_system=False)
            if len(ids) <= budget:
                return ids
        return self._encode_shortened(newest, budget)

    @staticmethod
    def _split_system(messages):
        """Split a leading system message off the conversation turns."""
        if messages and messages[0]["role"] == "system":
            return list(messages[:1]), list(messages[1:])
        return [], list(messages)

    def _encode_shortened(self, messages, budget):
        """Keep the newest content of the last message that still fits."""
        if not messages:
            return self.encode_chat(messages)[-budget:]
        tail = dict(messages[-1])
        content_ids = self._tokenizer.encode(tail["content"]).ids
        kept = None
        low, high = 0, len(content_ids)
        while low <= high:
            middle = (low + high) // 2
            tail["content"] = (
                self.decode(content_ids[-middle:]) if middle else ""
            )
            ids = self.encode_chat(
                [*messages[:-1], tail], include_default_system=False
            )
            if len(ids) <= budget:
                kept = ids
                low = middle + 1
            else:
                high = middle - 1
        if kept is not None:
            return kept
        # The ChatML scaffolding alone exceeds the budget: nothing well-formed
        # can be built, so fall back to the newest ids of an empty turn.
        tail["content"] = ""
        return self.encode_chat(
            [*messages[:-1], tail], include_default_system=False
        )[-budget:]

    def decode(self, token_ids, skip_special_tokens=False):
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


def _turn_starts(turns):
    """Return candidate window starts, widest first, at turn boundaries."""
    starts = [
        index
        for index, message in enumerate(turns)
        if message["role"] == "user"
    ]
    if not starts:
        return list(range(len(turns)))
    if starts[-1] != len(turns) - 1:
        starts.append(len(turns) - 1)
    return starts
