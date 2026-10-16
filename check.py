"""Train on a text file, assert round-trips, compare splits with tiktoken.

Usage:
    pip install regex tiktoken
    python check.py novel.txt            # with GPT-2 pre-split
    python check.py novel.txt --raw      # no pre-split, to see the ugly tokens
"""
import sys
import time

import tiktoken

from bpe import Tokenizer

TESTS = [
    "12345 + 678",
    "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n",
    "मुझे हिंदी में बात करना पसंद है।",
    # extra edge cases for the round-trip check
    "",
    "hello world",
    "  leading, trailing   and\n\n\nmultiple whitespace  ",
    "it's they'll we've I'M",
    "emoji 🙂, accents café, CJK 日本語",
    "1000000 3.14159 2026-10-15",
]


def fmt(b: bytes) -> str:
    """Show a token's bytes readably; partial UTF-8 sequences shown as hex."""
    try:
        s = b.decode("utf-8")
        return repr(s)[1:-1]
    except UnicodeDecodeError:
        return "<" + b.hex(" ") + ">"


def show(label, pieces):
    print(f"  {label:<12} ({len(pieces):>3}) " + " | ".join(fmt(p) for p in pieces))


def main():
    path = sys.argv[1]
    raw = "--raw" in sys.argv
    text = open(path, encoding="utf-8").read()
    print(f"corpus: {len(text):,} chars, {len(text.encode('utf-8')):,} bytes")

    tok = Tokenizer(pattern=None if raw else Tokenizer().pattern.pattern)
    t0 = time.time()
    tok.train(text, 1000)
    print(f"trained to {len(tok.vocab)} tokens in {time.time() - t0:.1f}s\n")

    # 1. round trips
    for s in TESTS:
        assert tok.decode(tok.encode(s)) == s, f"round-trip failed: {s!r}"
    sample = text[:50_000]
    assert tok.decode(tok.encode(sample)) == sample, "round-trip failed on corpus sample"
    print("all round-trips pass\n")

    # 2. compression on the corpus sample
    n_bytes = len(sample.encode("utf-8"))
    print(f"compression on sample: {n_bytes / len(tok.encode(sample)):.2f} bytes/token\n")

    # 3. what did it learn?
    print("last 30 merges:", " | ".join(fmt(tok.vocab[i]) for i in range(970, 1000)), "\n")

    # 4. side-by-side splits
    refs = {name: tiktoken.get_encoding(name) for name in ("gpt2", "cl100k_base", "o200k_base")}
    for s in TESTS[:3]:
        print(repr(s))
        show("yours", [tok.vocab[i] for i in tok.encode(s)])
        for name, enc in refs.items():
            show(name, [enc.decode_single_token_bytes(t) for t in enc.encode(s)])
        print()


if __name__ == "__main__":
    main()