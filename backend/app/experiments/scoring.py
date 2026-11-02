"""
Deterministic score extraction from guest replies.

A score is recorded only when the reply contains exactly one unambiguous value
on the variant's scale; anything else is left as None rather than guessed.
"""

import re
from typing import Optional

FRACTION_PATTERN = re.compile(r"(?<![\d.])(\d{1,2})\s*(?:/|out of)\s*(5|10)(?![\d.])", re.IGNORECASE)
INTEGER_PATTERN = re.compile(r"(?<![\d.])(\d{1,2})(?![\d.])")


def parse_score(text: Optional[str], scale: Optional[int]) -> Optional[int]:
    if not text or scale is None:
        return None

    fractions = FRACTION_PATTERN.findall(text)
    if fractions:
        values = {int(value) for value, denominator in fractions if int(denominator) == scale}
        if len(values) == 1 and len(fractions) == 1:
            value = values.pop()
            return value if 1 <= value <= scale else None
        return None

    values = INTEGER_PATTERN.findall(text)
    if len(values) != 1:
        return None
    value = int(values[0])
    return value if 1 <= value <= scale else None
