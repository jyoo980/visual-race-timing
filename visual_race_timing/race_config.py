"""Wave-start assignment from race config.

``starts`` entries in ``config.yaml`` carry a ``time`` (timecode string) and
either a ``bib_range`` ``[lo, hi)`` or an explicit ``bibs`` list (ints),
assigning each participant to whichever wave they belong to.
"""

from __future__ import annotations

from typing import Dict

from timecode import Timecode


def assign_start_by_runner(race_config: dict, fps) -> Dict[int, Timecode]:
    """Map runner_id -> wave-start Timecode, from config['starts'].

    Explicit ``bibs`` win over ``bib_range`` (more specific), matching a
    sensible order regardless of the order ``starts`` entries are declared
    in. ``time`` may already be a Timecode.
    """
    starts = race_config.get("starts", {})
    participants = list(race_config.get("participants", {}).keys())
    # Normalize participant ids to ints.
    part_ids = []
    for k in participants:
        part_ids.append(int(k, 16) if isinstance(k, str) else int(k))

    range_specs = []  # (lo, hi, timecode)
    bib_specs = []    # (set_of_ids, timecode)
    for name, details in starts.items():
        t = details["time"]
        if not isinstance(t, Timecode):
            t = Timecode(fps, t)
        if "bib_range" in details:
            lo, hi = details["bib_range"]
            range_specs.append((int(lo), int(hi), t))
        if "bibs" in details:
            bib_specs.append((set(int(b) for b in details["bibs"]), t))

    out: Dict[int, Timecode] = {}
    for rid in part_ids:
        assigned = None
        for ids, t in bib_specs:
            if rid in ids:
                assigned = t
                break
        if assigned is None:
            for lo, hi, t in range_specs:
                if lo <= rid < hi:
                    assigned = t
                    break
        if assigned is not None:
            out[rid] = assigned
    return out


def build_start_realtime(race_config: dict, fps) -> Dict[int, float]:
    """Map runner_id -> wave-start realtime seconds."""
    return {rid: t.to_realtime(as_float=True)
            for rid, t in assign_start_by_runner(race_config, fps).items()}
