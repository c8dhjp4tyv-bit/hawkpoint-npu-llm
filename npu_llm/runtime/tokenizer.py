import json
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SYSTEM = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)


class PromptBudgetError(ValueError):
    """The prompt budget is too small to hold any well-formed ChatML turn."""


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
        self._minimum_prompt_tokens = None

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

    @property
    def minimum_prompt_tokens(self):
        """Smallest well-formed ChatML prompt this tokenizer can emit.

        One empty user turn plus the generation prompt. A budget below this
        cannot hold a valid prompt at all, so no amount of trimming helps.
        """
        if self._minimum_prompt_tokens is None:
            self._minimum_prompt_tokens = len(
                self.encode_chat(
                    [{"role": "user", "content": ""}],
                    include_default_system=False,
                )
            )
        return self._minimum_prompt_tokens

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
        surrounding markers, so **every** returned encoding is well-formed
        ChatML. A budget too small to hold even an empty turn raises
        ``PromptBudgetError`` rather than degrading to a raw token slice.

        A conversation of only a system message keeps that message: the
        caller's text is never silently replaced by the family default.
        """
        if budget < self.minimum_prompt_tokens:
            raise PromptBudgetError(
                f"a prompt budget of {budget} token(s) cannot hold a valid "
                f"ChatML turn; this tokenizer needs at least "
                f"{self.minimum_prompt_tokens}"
            )
        system, turns = self._split_system(messages)
        if not turns:
            # System-only (or empty) conversation: encode it as given so an
            # explicit system message is preserved verbatim.
            ids = self.encode_chat(system, include_default_system=False)
            if len(ids) <= budget:
                return ids
            return self._encode_shortened(system, budget)
        for start in _turn_starts(turns):
            ids = self.encode_chat([*system, *turns[start:]])
            if len(ids) <= budget:
                return ids
        newest = turns[-1:]
        if system:
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
        if kept is None:
            # Only reachable if this message's own markers are larger than an
            # empty user turn; never emit a malformed slice for it.
            raise PromptBudgetError(
                f"a prompt budget of {budget} token(s) cannot hold a valid "
                f"{tail['role']!r} turn"
            )
        return kept

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
