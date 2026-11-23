"""Fitting an answer into a time limit (``generation.time_limit_s``).

On a CPU an answer has three parts that take time: the search, the answer model
reading the evidence, and the answer model writing. The first is over when this module
is asked. It then does two things about the other two.

**Before the model is called** it decides how much evidence the model is given. The
model reads at a steady speed, so the time left for reading, times that speed, is the
most evidence that fits::

    time left for reading = time_limit_s - time used so far - writing_reserve_s
    evidence              = time left for reading x reading speed (characters per second)

never less than ``min_evidence_chars`` and never more than ``evidence_char_budget``.
"Evidence" here is everything laid out for the model: the passages and the header of
each one. The reading speed is the one measured on this machine from the answers in
the query log (``Database.read_speed``): characters of evidence given, divided by the
seconds until the first word. Until there is one, ``read_chars_per_s`` from the
configuration stands in for it.

**While the model writes** it says when to stop. That is when the limit runs out, but
never sooner than ``writing_reserve_s`` after the first word: an answer cut after two
words is of no use to anyone. ``complete_part`` then keeps the text up to its last
complete statement.

What it does not do is interrupt the reading. If the model reads more slowly than it
was measured to, the answer starts late and ends late. So the limit is kept when the
measurement holds, and the first answers on a machine, which are planned from the
assumed speed, may run over it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from docqa.config import Config

# A reference the model wrote, "[E1, p. 12]", with the full stop or the like after it.
_REFERENCE = re.compile(r"\[[^\[\]\n]{1,80}\][.;:!?]?")
# The end of a sentence that carries no reference.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


@dataclass(frozen=True)
class AnswerPlan:
    """What one answer may use: how much evidence, and until when it may be written."""

    evidence_chars: int
    """Most evidence, in characters, that the answer model is given. With a time limit
    the headers of the passages count towards it (``choose_evidence``)."""
    time_limit_s: float | None = None
    """``generation.time_limit_s``. ``None``: there is no limit, and the answer is never stopped."""
    ends_at: float | None = None
    """When the limit runs out, as a ``time.perf_counter()`` value. ``None`` without a limit."""
    writing_reserve_s: float = 0.0
    read_chars_per_s: float | None = None
    """The reading speed the plan is based on."""
    speed_measured: bool = False
    """True if that speed was measured on this machine; False if it is the assumed one."""
    shortened: bool = False
    """True if the limit left room for less evidence than ``generation.evidence_char_budget``."""

    @property
    def evidence_budget(self) -> int | None:
        """What ``choose_evidence`` is given: ``evidence_chars`` when there is a time limit.

        Without one it is ``None``, and the evidence is cut as it always was, to
        ``generation.evidence_char_budget`` characters of passage text.
        """
        return self.evidence_chars if self.time_limit_s is not None else None

    def stop_writing_at(self, first_word_at: float) -> float | None:
        """When writing has to stop, as a ``time.perf_counter()`` value. ``None``: never.

        Args:
            first_word_at: When the model's first word arrived, on the same clock.
        """
        if self.ends_at is None:
            return None
        return max(self.ends_at, first_word_at + self.writing_reserve_s)

    def expected_read_s(self, chars: int) -> float | None:
        """How long the model can be expected to read ``chars`` characters of evidence."""
        if not self.read_chars_per_s:
            return None
        return chars / self.read_chars_per_s


def plan_answer(
    config: Config,
    *,
    used_s: float = 0.0,
    read_chars_per_s: float | None = None,
    now: float | None = None,
) -> AnswerPlan:
    """Decide how much evidence an answer may be given, and when its time is up.

    Args:
        config: The loaded configuration.
        used_s: Seconds since the question arrived: what the search has taken.
        read_chars_per_s: The answer model's reading speed as measured on this machine,
            or ``None`` if nothing has been measured yet.
        now: The current ``time.perf_counter()`` value. Tests pass their own.
    """
    generation = config.generation
    full = generation.evidence_char_budget
    measured = read_chars_per_s is not None and read_chars_per_s > 0
    speed = read_chars_per_s if measured else generation.read_chars_per_s
    if generation.time_limit_s is None:
        return AnswerPlan(evidence_chars=full, read_chars_per_s=speed, speed_measured=measured)

    left = generation.time_limit_s - used_s
    fits = int(max(left - generation.writing_reserve_s, 0.0) * speed)
    chars = min(full, max(generation.min_evidence_chars, fits))
    return AnswerPlan(
        evidence_chars=chars,
        time_limit_s=generation.time_limit_s,
        ends_at=(time.perf_counter() if now is None else now) + left,
        writing_reserve_s=generation.writing_reserve_s,
        read_chars_per_s=speed,
        speed_measured=measured,
        shortened=chars < full,
    )


def complete_part(text: str) -> str:
    """An answer that was stopped while it was being written, without its unfinished end.

    Every statement of an answer ends with its reference, so the text is kept up to
    its last complete reference. Text without any reference is kept up to the end of
    its last sentence. If there is neither, it is returned as it is: the note that
    the answer was stopped tells the reader the rest. A reference that the model had
    only begun to write, such as ``[E1, p.``, is dropped first.
    """
    text = text.rstrip()
    opened = text.rfind("[")
    if opened != -1 and "]" not in text[opened:]:
        text = text[:opened].rstrip()
    last = None
    for last in _REFERENCE.finditer(text):
        pass
    if last is not None:
        return text[: last.end()]
    for last in _SENTENCE_END.finditer(text):
        pass
    if last is not None:
        return text[: last.end()]
    return text
