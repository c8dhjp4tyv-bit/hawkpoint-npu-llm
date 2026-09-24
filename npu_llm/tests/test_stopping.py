#!/usr/bin/env python3
"""Hardware-free tests for stop sequences and log-probability options."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.stopping import (  # noqa: E402
    StopMatcher,
    log_softmax,
    normalize_logprobs,
    normalize_stop,
)


def feed_all(matcher, pieces):
    out = [matcher.feed(piece) for piece in pieces]
    out.append(matcher.flush())
    return "".join(out)


def test_normalize_stop():
    assert normalize_stop(None) == ()
    assert normalize_stop("END") == ("END",)
    assert normalize_stop(["a", "b", "a"]) == ("a", "b")
    for bad in ("", [""], [1], {"x": 1}, ["a", "b", "c", "d", "e"], ["x" * 257]):
        try:
            normalize_stop(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")


def test_normalize_logprobs():
    assert normalize_logprobs(None) is None
    assert normalize_logprobs(0) == 0
    assert normalize_logprobs(20) == 20
    for bad in (-1, 21, 1.5, True, "3"):
        try:
            normalize_logprobs(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")


def test_no_stops_passes_text_through():
    matcher = StopMatcher()
    assert matcher.feed("abc") == "abc"
    assert matcher.flush() == ""


def test_stop_spanning_pieces_is_never_emitted():
    matcher = StopMatcher(("\nUser:",))
    assert matcher.feed("Hello") == "Hello"
    assert matcher.feed("\nUs") == ""
    assert matcher.feed("er: hi") == ""
    assert matcher.matched == "\nUser:"
    assert matcher.feed("more") == ""
    assert matcher.flush() == ""


def test_ruled_out_prefix_is_released():
    matcher = StopMatcher(("STOP",))
    assert feed_all(matcher, ["ab", "ST", "OR", "E"]) == "abSTORE"
    assert matcher.matched is None


def test_earliest_stop_wins():
    matcher = StopMatcher(("world", "lo w"))
    assert matcher.feed("hello world") == "hel"
    assert matcher.matched == "lo w"


def test_held_suffix_is_released_at_the_end():
    matcher = StopMatcher(("###",))
    assert matcher.feed("answer #") == "answer "
    assert matcher.feed("#") == ""
    assert matcher.flush() == "##"


def test_log_softmax():
    values = log_softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert abs(np.exp(values).sum() - 1.0) < 1e-12
    assert values.argmax() == 2
    masked = log_softmax(np.array([0.0, -np.inf], dtype=np.float32))
    assert masked[0] == 0.0 and masked[1] == -np.inf


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)} stopping tests")


if __name__ == "__main__":
    main()
