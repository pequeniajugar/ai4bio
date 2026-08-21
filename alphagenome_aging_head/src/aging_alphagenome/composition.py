"""Sequence-composition utilities for shortcut/confound analysis."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import random

BASES = ("A", "C", "G", "T")
DINUCLEOTIDES = tuple(a + b for a in BASES for b in BASES)


def canonical_sequence(sequence: str) -> str:
    return "".join(base for base in sequence.upper() if base in BASES)


def base_counts(sequence: str) -> dict[str, int]:
    sequence = sequence.upper()
    return {base: sequence.count(base) for base in BASES}


def canonical_count(sequence: str) -> int:
    return sum(base_counts(sequence).values())


def gc_count(sequence: str) -> int:
    counts = base_counts(sequence)
    return counts["G"] + counts["C"]


def base_fractions(sequence: str) -> dict[str, float]:
    counts = base_counts(sequence)
    total = sum(counts.values())
    if total == 0:
        return {base: 0.0 for base in BASES}
    return {base: counts[base] / total for base in BASES}


def dinucleotide_counts(sequence: str) -> dict[str, int]:
    sequence = sequence.upper()
    counts = Counter(
        sequence[index : index + 2]
        for index in range(max(0, len(sequence) - 1))
        if sequence[index] in BASES and sequence[index + 1] in BASES
    )
    return {dinucleotide: counts.get(dinucleotide, 0) for dinucleotide in DINUCLEOTIDES}


def dinucleotide_fractions(sequence: str) -> dict[str, float]:
    counts = dinucleotide_counts(sequence)
    total = sum(counts.values())
    if total == 0:
        return {dinucleotide: 0.0 for dinucleotide in DINUCLEOTIDES}
    return {dinucleotide: counts[dinucleotide] / total for dinucleotide in DINUCLEOTIDES}


def shannon_entropy(sequence: str) -> float:
    fractions = base_fractions(sequence)
    return -sum(
        fraction * math.log2(fraction)
        for fraction in fractions.values()
        if fraction > 0.0
    )


def cpg_fraction(sequence: str) -> float:
    fractions = dinucleotide_fractions(sequence)
    return fractions["CG"]


def sequence_features(sequence: str) -> dict[str, float | int]:
    sequence = sequence.upper()
    counts = base_counts(sequence)
    fractions = base_fractions(sequence)
    dinucleotides = dinucleotide_fractions(sequence)
    canonical = sum(counts.values())
    features: dict[str, float | int] = {
        "length": len(sequence),
        "canonical_count": canonical,
        "ambiguous_fraction": 1.0 - canonical / len(sequence) if sequence else 1.0,
        "gc_count": counts["G"] + counts["C"],
        "gc_fraction": fractions["G"] + fractions["C"],
        "at_fraction": fractions["A"] + fractions["T"],
        "cpg_fraction": dinucleotides["CG"],
        "shannon_entropy": shannon_entropy(sequence),
    }
    for base in BASES:
        features[f"count_{base}"] = counts[base]
        features[f"fraction_{base}"] = fractions[base]
    for dinucleotide in DINUCLEOTIDES:
        features[f"dinuc_{dinucleotide}"] = dinucleotides[dinucleotide]
    return features


def composition_match(
    candidate: str,
    target: str,
    *,
    gc_count_tolerance: int,
    base_fraction_tolerance: float,
    dinucleotide_l1_tolerance: float,
) -> bool:
    """Return whether candidate is a stringent low-order composition match."""

    if abs(gc_count(candidate) - gc_count(target)) > gc_count_tolerance:
        return False
    candidate_bases = base_fractions(candidate)
    target_bases = base_fractions(target)
    if max(abs(candidate_bases[b] - target_bases[b]) for b in BASES) > base_fraction_tolerance:
        return False
    candidate_dinuc = dinucleotide_fractions(candidate)
    target_dinuc = dinucleotide_fractions(target)
    l1 = sum(abs(candidate_dinuc[k] - target_dinuc[k]) for k in DINUCLEOTIDES)
    return l1 <= dinucleotide_l1_tolerance


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def mononucleotide_shuffle(sequence: str, *, seed: int, key: str = "") -> str:
    """Shuffle positions while preserving every character count exactly."""

    values = list(sequence.upper())
    random.Random(stable_seed(seed, key)).shuffle(values)
    return "".join(values)


def adjacent_pair_counts(sequence: str) -> Counter[str]:
    """Count every overlapping adjacent character pair with stride 1.

    Unlike :func:`dinucleotide_counts`, this includes pairs containing ambiguous
    characters such as ``N``.  It is used by the exact shuffle verifier so that
    the transformed sequence cannot silently change any adjacent-pair count.
    """

    sequence = sequence.upper()
    return Counter(
        sequence[index : index + 2]
        for index in range(max(0, len(sequence) - 1))
    )


def dinucleotide_shuffle(sequence: str, *, seed: int, key: str = "") -> str:
    """Randomize a sequence while preserving every overlapping 2-mer exactly.

    A sequence is an Eulerian trail through a directed multigraph whose vertices
    are characters and whose edges are adjacent pairs.  Randomizing the order in
    which outgoing edges are consumed and reconstructing an Eulerian trail with
    Hierholzer's algorithm gives a deterministic-per-seed shuffle that preserves
    *all* stride-1 pair counts exactly.  Therefore A/C/G/T counts, GC content,
    CpG count, and all 16 canonical dinucleotide counts are preserved as well.

    The shuffle is not guaranteed to differ from the input: some sequences have
    a unique Eulerian trail.
    """

    sequence = sequence.upper()
    if len(sequence) < 2:
        return sequence

    rng = random.Random(stable_seed(seed, key))
    adjacency: dict[str, list[str]] = {}
    for left, right in zip(sequence[:-1], sequence[1:]):
        adjacency.setdefault(left, []).append(right)

    # Random edge order gives a randomized Eulerian trail while keeping the
    # directed multigraph (and therefore every adjacent-pair count) unchanged.
    for neighbors in adjacency.values():
        rng.shuffle(neighbors)

    stack = [sequence[0]]
    reversed_trail: list[str] = []
    while stack:
        vertex = stack[-1]
        neighbors = adjacency.get(vertex)
        if neighbors:
            stack.append(neighbors.pop())
        else:
            reversed_trail.append(stack.pop())

    shuffled = "".join(reversed(reversed_trail))
    verify_dinucleotide_preservation(sequence, shuffled)
    return shuffled


def verify_dinucleotide_preservation(original: str, shuffled: str) -> None:
    """Raise if an exact dinucleotide shuffle changed any required statistic."""

    original = original.upper()
    shuffled = shuffled.upper()
    if len(original) != len(shuffled):
        raise RuntimeError(
            f"Dinucleotide shuffle changed length: {len(original)} -> {len(shuffled)}"
        )
    if Counter(original) != Counter(shuffled):
        raise RuntimeError("Dinucleotide shuffle changed character counts.")
    if adjacent_pair_counts(original) != adjacent_pair_counts(shuffled):
        raise RuntimeError("Dinucleotide shuffle changed adjacent-pair counts.")


def hamming_fraction(first: str, second: str) -> float:
    """Fraction of positions that differ; sequences must have equal length."""

    if len(first) != len(second):
        raise ValueError("Hamming distance requires equal-length sequences.")
    if not first:
        return 0.0
    return sum(a != b for a, b in zip(first, second)) / len(first)
