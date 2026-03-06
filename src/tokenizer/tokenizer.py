from __future__ import annotations
from typing import List

# ── Special token IDs ─────────────────────────────────────────────────────────
PAD_ID   = 0
START_ID = 1
END_ID   = 2

# ── Character vocabulary (IDs 3–19) ──────────────────────────────────────────
# Digits, operators, and punctuation only — no alphabets in this domain.
_CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  # IDs 3–12
    "+", "-", "*", "/",                                   # IDs 13–16
    "=", "?", " ",                                        # IDs 17–19
]
VOCAB_SIZE = 3 + len(_CHARS)  # 20


class ArithmeticTokenizer:
    """Character-level tokenizer for arithmetic expressions.

    Encoding format for a full sample:
        [START] <question_chars> <space> <answer_chars> [END]

    During inference (RL rollout), only the prompt is encoded:
        [START] <question_chars> <space>
    The model generates answer characters autoregressively until END.
    """

    def __init__(self) -> None:
        self._c2i = {ch: i + 3 for i, ch in enumerate(_CHARS)}
        self._i2c = {v: k for k, v in self._c2i.items()}
        self.pad_id   = PAD_ID
        self.start_id = START_ID
        self.end_id   = END_ID
        self.vocab_size = VOCAB_SIZE

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Encode a raw string to token IDs (no special tokens added)."""
        try:
            return [self._c2i[ch] for ch in text]
        except KeyError as e:
            raise ValueError(f"Character {e} not in vocabulary") from e

    def encode_question(self, question: str) -> List[int]:
        """Encode question only with START token — used during rollout/inference.
        Trailing space separates question from where the answer will be generated."""
        return [self.start_id] + self.encode(question) + [self._c2i[" "]]

    def encode_sample(self, question: str, answer: int) -> List[int]:
        """Encode a full training sample with START and END tokens.
        Loss should be computed on answer tokens + END only (not the question)."""
        return (
            [self.start_id]
            + self.encode(question)
            + [self._c2i[" "]]
            + self.encode(str(answer))
            + [self.end_id]
        )

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to string, skipping special tokens."""
        return "".join(self._i2c[i] for i in ids if i in self._i2c)

    def decode_answer(self, ids: List[int]) -> str:
        """Decode generated answer IDs, stopping at END token."""
        chars = []
        for i in ids:
            if i == self.end_id:
                break
            if i in self._i2c:
                chars.append(self._i2c[i])
        return "".join(chars).strip()
