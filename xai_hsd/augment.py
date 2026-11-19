"""Hard-negative augmentation: counter-speech that quotes hateful text.

The most common false positive in moderation is a post that *contains* a slur
because it quotes, reports, or condemns one. This module builds labelled
non-hateful training examples by wrapping hateful training posts in a
counter-speech frame. Generated rows are train-only and must never reach
validation or test data.
"""

from __future__ import annotations

from dataclasses import replace

from .data import Record, normalize_text

COUNTER_FRAMES = (
    'quoting this so people can see what we are up against: "{post}" - do not talk '
    "like this.",
    'someone actually posted "{post}" and then deleted it. that is not acceptable here.',
    'the message quoted in the article was "{post}". historians study this as propaganda.',
    'reporting on a post that read "{post}" because the accounts behind it are being '
    "investigated.",
    'the reply said "{post}". i disagree completely and so does everyone else in this '
    "thread.",
    'found this old post: "{post}". glad the community has moved on from that language.',
    'their argument boils down to "{post}", which is exactly the kind of claim the study '
    "debunks.",
    'as a moderator i removed a comment that said "{post}". the author has been contacted.',
)


def counter_speech(post: str, frame: str | None = None) -> str:
    """Wrap a post in a condemning frame. Deterministic for a given frame."""
    text = normalize_text(post)
    if not text:
        raise ValueError("Counter-speech needs a nonempty post")
    return normalize_text((frame or COUNTER_FRAMES[0]).format(post=text))


def hard_negatives(
    records,
    *,
    copies: int = 1,
    seed: int = 42,
    frames=COUNTER_FRAMES,
) -> list[Record]:
    """Build non-hateful counter-speech examples from hateful training posts.

    Each source post produces ``copies`` rows, cycling through ``frames`` so the
    same post appears inside several different contexts. Identifiers and the
    ``train`` split are set explicitly; the input is not modified.
    """
    if copies < 0:
        raise ValueError("Copies must be nonnegative")
    if copies and not frames:
        raise ValueError("At least one counter-speech frame is required")
    records = list(records)
    sources: list[Record] = []
    source_keys: set[str] = set()
    for record in sorted(
        (record for record in records if record.label == 1), key=lambda item: item.id
    ):
        key = normalize_text(record.text).casefold()
        if key and key not in source_keys:
            source_keys.add(key)
            sources.append(record)

    seen = {normalize_text(record.text).casefold() for record in records}
    generated: list[Record] = []
    for copy in range(copies):
        for position, record in enumerate(sources):
            # Offset by source as well as copy so repeated copies use new frames.
            frame = frames[(position * copies + copy) % len(frames)]
            try:
                text = counter_speech(record.text, frame)
            except (KeyError, IndexError) as error:
                raise ValueError(f"Invalid counter-speech frame at index {position}") from error
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            generated.append(replace(
                record,
                id=f"{record.id}-counter{seed}-{copy}",
                text=text,
                label=0,
                split="train",
            ))
    return generated
