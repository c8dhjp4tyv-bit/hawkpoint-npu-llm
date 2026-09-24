import json
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SYSTEM = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
REPLACEMENT_CHARACTER = "\ufffd"


def _byte_decoder():
    """Invert the GPT-2 byte-to-unicode table used by byte-level BPE vocabularies."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): byte for byte, character in zip(printable, characters)}


_BYTE_DECODER = _byte_decoder()


class SmolLMTokenizer:
    def __init__(self, model_dir, default_system=DEFAULT_SYSTEM):
        model_dir = Path(model_dir)
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # Message content is untrusted text. Encoding it with special-token
        # parsing disabled keeps a literal "<|im_end|>" in a message from
        # closing the turn and forging another role; the chat scaffold below
        # inserts the real control tokens by id instead.
        self._content_tokenizer = Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        )
        self._content_tokenizer.encode_special_tokens = True
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        self.eos_token = config.get("eos_token", IM_END)
        self.eos_id = self._tokenizer.token_to_id(self.eos_token)
        self.default_system = default_system
        self._im_start_id = self._tokenizer.token_to_id(IM_START)
        self._im_end_id = self._tokenizer.token_to_id(IM_END)
        if self._im_start_id is None or self._im_end_id is None:
            raise ValueError(
                "tokenizer does not define the ChatML control tokens "
                f"{IM_START} and {IM_END}"
            )
        self._newline_ids = self._content_tokenizer.encode(
            "\n", add_special_tokens=False
        ).ids

    def _with_default_system(self, messages):
        if not messages or messages[0]["role"] != "system":
            return [{"role": "system", "content": self.default_system}, *messages]
        return list(messages)

    def format_chat(self, messages, add_generation_prompt=True):
        messages = self._with_default_system(messages)
        text = "".join(
            f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
            for m in messages
        )
        if add_generation_prompt:
            text += f"{IM_START}assistant\n"
        return text

    def _encode_text(self, text):
        return self._content_tokenizer.encode(text, add_special_tokens=False).ids

    def _encode_turn(self, role, content):
        # Encoding each turn separately yields the same ids as encoding the
        # rendered template in one call: the tokenizer already splits the text
        # at every control token and never merges across them.
        return [
            self._im_start_id,
            *self._encode_text(f"{role}\n{content}"),
            self._im_end_id,
            *self._newline_ids,
        ]

    def _generation_prompt(self):
        return [self._im_start_id, *self._encode_text("assistant\n")]

    def _encode_turns(self, messages):
        ids = []
        for message in messages:
            ids.extend(self._encode_turn(message["role"], message["content"]))
        ids.extend(self._generation_prompt())
        return ids

    def encode_chat(self, messages):
        return self._encode_turns(self._with_default_system(messages))

    def encode_chat_window(self, messages, budget):
        """Encode a conversation into at most ``budget`` prompt tokens.

        The hardware attention window is small, so long conversations must be
        shortened. Cutting raw tokens from the front would start the prompt in
        the middle of a turn and keep only the tail of the system prompt.
        Instead, the newest message is kept whole for as long as possible and
        the window is reduced at turn boundaries, in this order:

        1. drop the oldest turns after the system prompt;
        2. shorten the system prompt from its end, dropping it when not even
           its first word fits;
        3. keep only the newest characters of the newest message.

        Only when even the empty newest turn cannot fit does it fall back to
        keeping the newest ``budget`` tokens. Returns ``(ids, truncation)``
        where ``truncation`` describes what was removed.
        """
        if budget <= 0:
            raise ValueError("prompt budget must be positive")
        messages = self._with_default_system(messages)
        truncation = {
            "dropped_messages": 0,
            "truncated_system": False,
            "dropped_system": False,
            "truncated_content": False,
            "token_fallback": False,
        }
        full = self._encode_turns(messages)
        if len(full) <= budget:
            return full, truncation

        system, turns = messages[0], messages[1:]
        if not turns:
            # Only the system prompt was supplied; it is the newest message.
            system, turns = None, [system]
        while len(turns) > 1:
            turns = turns[1:]
            truncation["dropped_messages"] += 1
            ids = self._encode_turns([system, *turns])
            if len(ids) <= budget:
                return ids, truncation

        newest = turns[-1]
        if system is not None:
            fitted = self._fit_system_prefix(system, newest, budget)
            if fitted is not None:
                truncation["truncated_system"] = True
                return fitted, truncation
            truncation["dropped_system"] = True
            truncation["dropped_messages"] += 1
            ids = self._encode_turns([newest])
            if len(ids) <= budget:
                return ids, truncation

        fitted = self._fit_newest_suffix(newest, budget)
        if fitted is not None:
            truncation["truncated_content"] = True
            return fitted, truncation
        truncation["token_fallback"] = True
        return full[-budget:], truncation

    @staticmethod
    def _longest_fitting(length, fits):
        """Largest ``n`` in ``0..length`` with ``fits(n)``, or ``None``.

        Token counts grow with the kept text, but BPE merges make that only
        approximately monotonic, so the binary-search result is confirmed.
        """
        if not fits(0):
            return None
        low, high = 0, length
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        while low > 0 and not fits(low):
            low -= 1
        return low

    def _fit_system_prefix(self, system, newest, budget):
        """Keep the longest whole-word beginning of the system prompt that fits."""
        content = system["content"]

        def encode(length):
            kept = content[:length]
            if 0 < length < len(content):
                # Prefer ending on a word boundary over a cut-off word.
                boundary = kept.rstrip().rfind(" ")
                if boundary > 0:
                    kept = kept[:boundary]
            return self._encode_turns(
                [{"role": system["role"], "content": kept.rstrip()}, newest]
            )

        length = self._longest_fitting(
            len(content), lambda length: len(encode(length)) <= budget
        )
        if not length:
            return None
        return encode(length)

    def _fit_newest_suffix(self, turn, budget):
        """Keep the longest suffix of the newest message's content that fits."""
        content = turn["content"]

        def encode(length):
            suffix = content[len(content) - length :] if length else ""
            return self._encode_turns(
                [{"role": turn["role"], "content": suffix}]
            )

        length = self._longest_fitting(
            len(content), lambda length: len(encode(length)) <= budget
        )
        if length is None:
            return None
        return encode(length)

    def decode(self, token_ids, skip_special_tokens=False):
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def token_bytes(self, token_id):
        """Return the exact bytes a token contributes to the generated text.

        Byte-level BPE tokens can hold part of a multi-byte UTF-8 character,
        which ``decode`` alone renders as U+FFFD.
        """
        piece = self._tokenizer.id_to_token(int(token_id))
        if piece is not None and all(char in _BYTE_DECODER for char in piece):
            return bytes(_BYTE_DECODER[char] for char in piece)
        return self.decode([int(token_id)]).encode("utf-8")


class StreamDetokenizer:
    """Turn a growing token sequence into text without splitting characters.

    Decoding one token at a time corrupts characters whose UTF-8 bytes span
    several byte-level BPE tokens (for example Turkish "ç" in SmolLM2, or most
    emoji), producing U+FFFD. This decodes the whole sequence each time and
    releases only text that is complete; a trailing partial character is held
    until the next token completes it or :meth:`flush` is called.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._ids = []
        self._text = ""
        self._released = 0

    @property
    def text(self):
        """All text decoded so far, including any held partial character."""
        return self._text

    def push(self, token_id):
        """Append a token and return the newly completed text, if any."""
        self._ids.append(int(token_id))
        self._text = self._tokenizer.decode(self._ids)
        if self._text.endswith(REPLACEMENT_CHARACTER):
            return ""
        return self._release()

    def flush(self):
        """Return any text still held, even if it ends in a partial character."""
        return self._release()

    def _release(self):
        released = self._text[self._released :]
        self._released = len(self._text)
        return released
