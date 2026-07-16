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


def _resolve_point(point, frame_width, frame_height):
    """A ``[x, y]`` config point to a ``(x, y)`` pixel-coordinate tuple.

    Float coordinates are treated as fractions of (frame_width, frame_height)
    in [0, 1] (resolution-independent); ints are already pixel coordinates.
    """
    x, y = point
    if isinstance(x, float) or isinstance(y, float):
        if frame_width is None or frame_height is None:
            raise ValueError("finish_line has normalized float coordinates but no "
                             "frame_width/frame_height was given to resolve them to pixels.")
        return x * frame_width, y * frame_height
    return float(x), float(y)


def get_finish_line(race_config: dict, timecode: Timecode, frame_width: int = None, frame_height: int = None):
    """Resolve ``config['finish_line']`` to a ``(p0, p1)`` pixel-coordinate pair for ``timecode``.

    ``finish_line`` may be either a fixed ``[p0, p1]`` pair, or a dict of
    timecode-string -> ``[p0, p1]`` waypoints (e.g. to track a panning camera
    that's moved to a new fixed position part way through a race). This is a
    step function, not an interpolation: the line holds at the most recent
    waypoint at or before ``timecode``, and at the earliest waypoint for any
    time before it. Points may be int pixel coordinates or float coordinates
    in [0, 1], normalized to (frame_width, frame_height).
    """
    raw = race_config['finish_line']
    if not isinstance(raw, dict):
        return (_resolve_point(raw[0], frame_width, frame_height),
                _resolve_point(raw[1], frame_width, frame_height))

    def _waypoint_frames(k):
        t = k if isinstance(k, Timecode) else Timecode(timecode.framerate, k)
        return t.frames

    waypoints = sorted(
        ((_waypoint_frames(k), v) for k, v in raw.items()),
        key=lambda entry: entry[0])
    cur = timecode.frames
    active = waypoints[0][1]
    for f, v in waypoints:
        if f > cur:
            break
        active = v
    return (_resolve_point(active[0], frame_width, frame_height),
            _resolve_point(active[1], frame_width, frame_height))
