#!/usr/bin/env python3
"""Context-window trimming and default-system-prompt selection tests.

These cover the host-side conversation handling without needing NumPy, an
MLIR-AIE environment, or a converted checkpoint: the trimming logic is
exercised through a deterministic stand-in tokenizer.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu_llm.runtime.prompts import (  # noqa: E402
    default_system_prompt,
    seed_messages,
)
from npu_llm.runtime.tokenizer import SmolLMTokenizer  # noqa: E402


MARKERS = ("<|im_start|>", "<|im_end|>")
CONTEXT = 64


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Word-level tokenizer where each ChatML marker is exactly one token."""

    def __init__(self):
        self._ids = {}
        self._pieces = []

    def _id(self, piece):
        if piece not in self._ids:
            self._ids[piece] = len(self._pieces)
            self._pieces.append(piece)
        return self._ids[piece]

    @staticmethod
    def _split(text):
        pieces = []
        for chunk in text.replace("\n", " ").split(" "):
            while chunk:
                start = min(
                    (chunk.find(marker) for marker in MARKERS
                     if marker in chunk),
                    default=-1,
                )
                if start == -1:
                    pieces.append(chunk)
                    break
                marker = next(m for m in MARKERS if chunk.startswith(m, start))
                if start:
                    pieces.append(chunk[:start])
                pieces.append(marker)
                chunk = chunk[start + len(marker):]
        return [piece for piece in pieces if piece]

    def encode(self, text):
        return FakeEncoding([self._id(piece) for piece in self._split(text)])

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self._pieces[index] for index in ids)

    def token_to_id(self, token):
        return self._id(token)


def build_tokenizer(default_system="You are a helpful AI assistant named SmolLM."):
    tokenizer = SmolLMTokenizer.__new__(SmolLMTokenizer)
    tokenizer._tokenizer = FakeTokenizer()
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_id = tokenizer._tokenizer.token_to_id("<|im_end|>")
    tokenizer.default_system = default_system
    return tokenizer


def pieces(tokenizer, ids):
    return [tokenizer._tokenizer._pieces[index] for index in ids]


def assert_well_formed(tokenizer, ids):
    """The retained prompt must be a complete ChatML sequence."""
    decoded = pieces(tokenizer, ids)
    assert decoded[0] == "<|im_start|>", decoded[:4]
    assert decoded[1] in {"system", "user", "assistant"}, decoded[:4]
    assert decoded[-2:] == ["<|im_start|>", "assistant"], decoded[-4:]
    starts = decoded.count("<|im_start|>")
    ends = decoded.count("<|im_end|>")
    # Every turn is closed except the trailing generation prompt.
    assert starts == ends + 1, decoded


def test_short_conversation_is_untouched():
    tokenizer = build_tokenizer()
    messages = [{"role": "user", "content": "Say hello."}]
    budget = CONTEXT - 16
    ids = tokenizer.encode_chat_within(messages, budget)
    assert ids == tokenizer.encode_chat(messages)
    assert len(ids) <= budget
    assert_well_formed(tokenizer, ids)


def test_long_history_drops_whole_turns_and_keeps_the_system_prompt():
    tokenizer = build_tokenizer()
    messages = [{"role": "system", "content": "Follow the house style."}]
    for turn in range(6):
        messages.append(
            {"role": "user", "content": f"question number {turn} about things"}
        )
        messages.append(
            {"role": "assistant", "content": f"answer number {turn} with detail"}
        )
    messages.append({"role": "user", "content": "and finally what about now"})

    budget = CONTEXT - 16
    ids = tokenizer.encode_chat_within(messages, budget)
    decoded = pieces(tokenizer, ids)
    assert len(ids) <= budget
    assert_well_formed(tokenizer, ids)
    # The full history does not fit, so trimming really happened.
    assert len(tokenizer.encode_chat(messages)) > budget
    # System message and newest user turn survive; the oldest turn does not.
    assert "style." in decoded
    assert decoded.count("system") == 1
    assert "now" in decoded
    assert "0" not in decoded


def test_newest_turn_wins_the_window_over_the_system_prompt():
    tokenizer = build_tokenizer()
    messages = [
        {"role": "system", "content": " ".join(["boilerplate"] * 12)},
        {"role": "user", "content": "why does the sky look blue during the day"},
    ]
    budget = 20
    ids = tokenizer.encode_chat_within(messages, budget)
    decoded = pieces(tokenizer, ids)
    assert len(ids) <= budget
    assert_well_formed(tokenizer, ids)
    assert "boilerplate" not in decoded
    # The question survives whole: the identity preamble was dropped, the
    # user's words were not shortened.
    assert "why" in decoded and "blue" in decoded and "day" in decoded


def test_single_oversized_turn_is_shortened_but_stays_well_formed():
    tokenizer = build_tokenizer()
    messages = [{"role": "user", "content": " ".join(f"word{i}" for i in range(80))}]
    budget = 24
    ids = tokenizer.encode_chat_within(messages, budget)
    decoded = pieces(tokenizer, ids)
    assert len(ids) <= budget
    assert_well_formed(tokenizer, ids)
    # The newest content is what survives.
    assert "word79" in decoded
    assert "word0" not in decoded


def test_generation_budget_is_always_respected():
    tokenizer = build_tokenizer()
    messages = [
        {"role": "system", "content": " ".join(["policy"] * 30)},
        *[
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": " ".join(f"t{index}w{word}" for word in range(9)),
            }
            for index in range(9)
        ],
    ]
    for max_new_tokens in range(1, CONTEXT):
        ids = tokenizer.encode_chat_within(messages, CONTEXT - max_new_tokens)
        assert len(ids) + max_new_tokens <= CONTEXT, max_new_tokens


def test_default_system_prompt_is_family_specific():
    assert "Qwen" in default_system_prompt("qwen2")
    assert "SmolLM" not in default_system_prompt("qwen2")
    assert "SmolLM" in default_system_prompt("llama")
    # Unknown families fall back rather than raising.
    assert default_system_prompt("mystery-arch")


def test_cli_only_seeds_a_system_message_when_asked():
    # Without --system-prompt the CLI must not inject an identity, so the
    # decoder's family default (Qwen for a Qwen checkpoint) applies.
    assert seed_messages() == []
    assert seed_messages(None) == []
    assert seed_messages("") == []
    assert seed_messages("be terse") == [
        {"role": "system", "content": "be terse"}
    ]

    qwen = build_tokenizer(default_system=default_system_prompt("qwen2"))
    prompt = qwen.format_chat(seed_messages() + [{"role": "user", "content": "hi"}])
    assert "You are Qwen" in prompt
    assert "SmolLM" not in prompt


def main():
    test_short_conversation_is_untouched()
    test_long_history_drops_whole_turns_and_keeps_the_system_prompt()
    test_newest_turn_wins_the_window_over_the_system_prompt()
    test_single_oversized_turn_is_shortened_but_stays_well_formed()
    test_generation_budget_is_always_respected()
    test_default_system_prompt_is_family_specific()
    test_cli_only_seeds_a_system_message_when_asked()
    print("PASS chat context trimming and prompt defaults")


if __name__ == "__main__":
    main()
