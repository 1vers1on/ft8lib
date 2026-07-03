"""Realtime (streaming) decoding: feed audio as it arrives, get decodes early.

Instead of collecting a full receive period and decoding it at once, a
:class:`RealtimeDecoder` accepts audio in arbitrary-size blocks and runs
decode attempts part-way through the period over whatever has arrived so
far, the way WSJT-X does (for FT8 at t = 11.8, 13.5 and 14.7 s into the
cycle).  Most messages therefore appear seconds before the period ends.
Each unique message is reported once per cycle, the first time it decodes;
cycle rollover is handled automatically for continuous operation.

Typical use with a sound card or other block source::

    rt = ft8lib.RealtimeDecoder("FT8", mycall="K1ABC")
    for block in audio_blocks:          # feeding starts at a cycle boundary
        for result in rt.feed(block):
            print(result)
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .decode import Decode, _decode_ft4_events, _decode_ft8_events
from .pack import HashTable
from .protocol import FT4, FT8, SAMPLE_RATE

# Per-mode defaults: decode-attempt times (s into the cycle) and the
# candidate-search parameters of decode_ft8/decode_ft4.  The FT8 attempt
# times are the three WSJT-X decode passes; FT4 signals end by ~6.0 s, so
# one early attempt plus one at the end of the audio buffer suffice.
_MODES = {
    "FT8": dict(events=_decode_ft8_events, nmax=FT8.NMAX, period=FT8.PERIOD,
                syncmin=1.3, max_candidates=120,
                attempt_times=(11.8, 13.5, 14.7)),
    "FT4": dict(events=_decode_ft4_events, nmax=FT4.NMAX, period=FT4.PERIOD,
                syncmin=1.18, max_candidates=100,
                attempt_times=(5.6, 6.048)),
}


class RealtimeDecoder:
    """Incremental FT8/FT4 decoder for live 12 kHz audio.

    Parameters are those of :func:`ft8lib.decode_ft8` / ``decode_ft4``
    (``syncmin`` and ``max_candidates`` default per mode), plus:

    attempt_times : decode-attempt schedule, in seconds into the receive
        cycle.  Defaults to the WSJT-X schedule (FT8: 11.8, 13.5, 14.7 s;
        FT4: 5.6, 6.048 s).

    The ``hashes`` table persists across cycles, so hashed callsigns heard
    in one cycle resolve in later ones.  Attributes ``cycle`` (0-based
    receive-period counter) and ``t`` (seconds into the current cycle) track
    where the decoder is; feeding is assumed to start at a cycle boundary.
    """

    def __init__(self, mode: str = "FT8", *,
                 freq_min: float = 200.0, freq_max: float = 4000.0,
                 syncmin: Optional[float] = None,
                 max_candidates: Optional[int] = None,
                 hashes: Optional[HashTable] = None, depth: int = 3,
                 mycall: str = "", dxcall: str = "", npasses: int = 3,
                 attempt_times: Optional[Iterable[float]] = None):
        try:
            params = _MODES[mode.upper()]
        except KeyError:
            raise ValueError(f"mode must be 'FT8' or 'FT4', got {mode!r}")
        self.mode = mode.upper()
        self._events = params["events"]
        self._nmax = params["nmax"]
        self._period = int(round(params["period"] * SAMPLE_RATE))
        if attempt_times is None:
            attempt_times = params["attempt_times"]
        self._attempts = sorted(
            min(int(round(t * SAMPLE_RATE)), self._period)
            for t in attempt_times)
        if not self._attempts or self._attempts[0] <= 0:
            raise ValueError("attempt_times must be positive")
        self.hashes = hashes if hashes is not None else HashTable()
        self._kwargs = dict(
            freq_min=freq_min, freq_max=freq_max,
            syncmin=params["syncmin"] if syncmin is None else syncmin,
            max_candidates=(params["max_candidates"] if max_candidates is None
                            else max_candidates),
            hashes=self.hashes, depth=depth, mycall=mycall, dxcall=dxcall,
            npasses=npasses)
        self.cycle = 0
        self._start_cycle()

    def _start_cycle(self):
        self._buf = np.zeros(self._nmax)
        self._filled = 0          # samples buffered (capped at nmax)
        self._pos = 0             # samples consumed this cycle (up to period)
        self._seen = set()        # messages already reported this cycle
        self._next_attempt = 0

    @property
    def t(self) -> float:
        """Seconds into the current receive cycle."""
        return self._pos / SAMPLE_RATE

    def reset(self):
        """Discard buffered audio and realign to a cycle boundary."""
        self.cycle = 0
        self._start_cycle()

    def feed(self, samples) -> List[Decode]:
        """Consume the next block of audio (12000 S/s, any scale, any length).

        Returns the messages newly decoded as a result of this block —
        usually empty, and non-empty when the block carried the audio past
        a scheduled decode-attempt time.  Blocks may span cycle boundaries.
        """
        x = np.asarray(samples, dtype=np.float64).ravel()
        out: List[Decode] = []
        while x.size:
            take = min(x.size, self._period - self._pos)
            chunk, x = x[:take], x[take:]
            room = self._nmax - self._filled
            if room > 0:
                n = min(take, room)
                self._buf[self._filled:self._filled + n] = chunk[:n]
                self._filled += n
            self._pos += take
            out.extend(self._run_due_attempt())
            if self._pos >= self._period:
                self.cycle += 1
                self._start_cycle()
        return out

    def flush(self) -> List[Decode]:
        """Decode whatever audio is buffered, off the attempt schedule.

        Useful when a recording ends mid-cycle.  Does not advance the cycle;
        messages already reported this cycle are not repeated.
        """
        if self._filled == 0:
            return []
        return self._attempt()

    def _run_due_attempt(self) -> List[Decode]:
        # Run at most one decode per feed step: the latest due attempt.
        # A large block can jump several attempt times at once; the earlier
        # ones would only see the same audio, so they are skipped.
        due = False
        while (self._next_attempt < len(self._attempts)
               and self._pos >= self._attempts[self._next_attempt]):
            due = True
            self._next_attempt += 1
        if not due or self._filled == 0:
            return []
        return self._attempt()

    def _attempt(self) -> List[Decode]:
        new = []
        for result, is_new in self._events(self._buf, **self._kwargs):
            if is_new and result.message not in self._seen:
                self._seen.add(result.message)
                new.append(result)
        return new


def decode_realtime(blocks: Iterable, mode: str = "FT8", **kwargs):
    """Yield decodes from an iterable of audio blocks as they become available.

    Convenience generator wrapping :class:`RealtimeDecoder`: feeds each block
    in turn, yielding new decodes immediately, and flushes any remaining
    buffered audio when the iterable is exhausted.  Keyword arguments are
    those of RealtimeDecoder.
    """
    rt = RealtimeDecoder(mode, **kwargs)
    for block in blocks:
        yield from rt.feed(block)
    yield from rt.flush()
