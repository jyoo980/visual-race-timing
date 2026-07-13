"""Interactive, partially-supervised race tracker re-hosted on boxmot BotSort.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from boxmot.trackers.bbox.botsort import BotSort
from boxmot.trackers.common.track_models.botsort import STrack  # noqa: F401  (re-exported for callers)
from boxmot.trackers.common.tracking.track import sync_track_meta

from visual_race_timing.prompts import ask_for_id
from visual_race_timing.logging import get_logger

logger = get_logger(__name__)

# IDs at/above this base are "unknowns": tracked people without a confirmed bib.
UNKNOWN_ID_BASE = 0xF00

VALID_POLICIES = ("prompt", "known_id", "spawn_unknown")


class RaceTracker(BotSort):
    """A BotSort whose unmatched-detection handling is human-in-the-loop.

    Parameters
    ----------
    reid_model :
        The boxmot ReID *backend* (i.e. ``ReID(...).model``) to share with the
        ReIDBank. Required when ``with_reid=True``.
    participants : dict
        ``{bib_hex_lower: name}`` used to populate the interactive prompt.
    policy : str
        One of :data:`VALID_POLICIES`.
    use_cmc : bool
        Camera-motion compensation. Defaults to **False**: the interactive
        annotate workflow feeds sparse, non-contiguous frames (jumping to
        on-the-line moments), so ECC motion estimation between them is invalid
        and would corrupt track motion. The old ultralytics-based path did not
        run meaningful CMC either.
    """

    def __init__(
        self,
        reid_model: Any | None = None,
        *,
        participants: Optional[dict] = None,
        policy: str = "prompt",
        frame_rate: int = 30,
        use_cmc: bool = False,
        with_reid: bool = True,
        **kwargs: Any,
    ):
        if policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}, got {policy!r}")
        super().__init__(
            reid_model=reid_model,
            frame_rate=frame_rate,
            use_cmc=use_cmc,
            with_reid=with_reid,
            **kwargs,
        )
        self.participants = participants or {}
        self.policy = policy
        # id -> name for people we track without a confirmed bib.
        self.unknowns: dict[int, str] = {}
        # Optional hook: receives an annotated image while prompting.
        self.display_delegate = lambda img: None
        # Per-frame side channel: original-detection-index -> forced bib id.
        self._frame_known_ids: dict[int, int] = {}
        self._cur_img: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public update: accepts per-detection known IDs, returns TrackResults.
    # ------------------------------------------------------------------
    def update(self, dets, img=None, embs=None, masks=None, known_ids=None, **kwargs):
        """Update with one frame.

        ``dets`` is ``(N, 6) = (x1, y1, x2, y2, conf, cls)``. ``known_ids`` is an
        optional length-N iterable of bib IDs (``-1``/``None`` = unknown) aligned
        to the ``dets`` rows. Returns a boxmot ``TrackResults`` (N, 8):
        ``(x1, y1, x2, y2, id, conf, cls, det_ind)``.
        """
        self._frame_known_ids = {}
        if known_ids is not None:
            for i, kid in enumerate(known_ids):
                if kid is None:
                    continue
                kid = int(kid)
                if kid != -1:
                    self._frame_known_ids[i] = kid
        self._cur_img = img
        return super().update(dets, img=img, embs=embs, masks=masks, **kwargs)

    # ------------------------------------------------------------------
    # Seam 1: donate known IDs to tracks matched during first association.
    # ------------------------------------------------------------------
    def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img,
                           detections, activated_stracks, refind_stracks, strack_pool):
        matches, u_track, u_detection = super()._first_association(
            dets, dets_first, active_tracks, unconfirmed, img,
            detections, activated_stracks, refind_stracks, strack_pool,
        )
        if self._frame_known_ids:
            for itracked, idet in matches:
                forced = self._frame_known_ids.get(int(detections[idet].det_ind), -1)
                if forced != -1:
                    # An annotated box matched an existing track: adopt its bib.
                    self._assign_id(strack_pool[itracked], forced)
        return matches, u_track, u_detection

    # ------------------------------------------------------------------
    # Seam 2: apply the unmatched-detection policy (THE interactive injection).
    # Mirrors BotSort._initialize_new_tracks but routes each unmatched
    # detection through prompt / known_id / spawn_unknown.
    # ------------------------------------------------------------------
    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        for inew in u_detections:
            track = detections[inew]
            forced_id = self._frame_known_ids.get(int(track.det_ind), -1)

            if forced_id != -1:
                # Trusted annotation: activate and force the bib (bypass conf gate).
                track.activate(self.kalman_filter, self.frame_count)
                self._assign_id(track, forced_id)
                activated_stracks.append(track)
                continue

            if self.policy in ("known_id", "spawn_unknown"):
                track.activate(self.kalman_filter, self.frame_count)
                self._assign_id(track, self._next_unknown_id())
                activated_stracks.append(track)
                continue

            # policy == "prompt": ask a human what this unmatched detection is.
            assigned = self._prompt_for_id(track, detections)
            if assigned is None:
                continue  # user skipped
            track.activate(self.kalman_filter, self.frame_count)
            self._assign_id(track, assigned)
            activated_stracks.append(track)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assign_id(self, track: STrack, new_id: int) -> None:
        """Force ``new_id`` onto ``track`` and confirm it for this frame.

        boxmot only emits tracks with ``is_activated=True`` (which ``activate``
        sets only on frame 1). A human/annotation decision confirms the track
        immediately, so we set it here to make the ID appear in the current
        frame's output, matching the old tracker's behavior.
        """
        track.id = int(new_id)
        track.is_activated = True
        sync_track_meta(track, track.common_tracked_state)

    def _next_unknown_id(self) -> int:
        new_id = UNKNOWN_ID_BASE + len(self.unknowns)
        self.unknowns[new_id] = ""
        return new_id

    def _register_unknown(self, name: str) -> int:
        new_id = UNKNOWN_ID_BASE + len(self.unknowns)
        self.unknowns[new_id] = name
        return new_id

    def _prompt_for_id(self, track: STrack, detections) -> Optional[int]:
        """Interactive prompt for one unmatched detection.

        Returns the integer id to assign, or ``None`` to skip. ``'U<name>'``
        spawns a named unknown; ``'skip'`` skips.
        """
        self._show_candidate(track)
        # Candidate list: configured participants first, then current unknowns.
        bibs = list(self.participants.keys())
        bibs.extend(
            format(uid, "02x").lower() for uid in self.unknowns
            if format(uid, "02x").lower() not in bibs
        )
        names = [self.participants.get(bib, "") or self.unknowns.get(int(bib, 16), "") for bib in bibs]
        choices = [(bib, (name,)) for bib, name in zip(bibs, names)]

        user_input = ask_for_id(choices, show_default=False, allow_other=True)
        if user_input is None:
            return None
        stripped = user_input.strip()
        if stripped.lower() == "skip" or stripped == "":
            return None
        if stripped[0].lower() == "u":
            return self._register_unknown(stripped[1:].strip())
        try:
            return int(stripped, 16)
        except ValueError:
            logger.warning("Unparseable bib %r; skipping detection.", user_input)
            return None

    def _show_candidate(self, track: STrack) -> None:
        """Best-effort: draw the candidate box and hand it to display_delegate."""
        if self._cur_img is None:
            return
        try:
            img = self._cur_img.copy()
            x1, y1, x2, y2 = np.asarray(track.xyxy, dtype=int)[:4]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            self.display_delegate(img)
        except Exception:  # display is a convenience, never fatal to tracking
            logger.debug("display_delegate failed", exc_info=True)
