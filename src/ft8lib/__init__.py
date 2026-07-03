"""ft8lib: FT8, FT4 and WSPR digital-mode encoder/decoder.

A Python (numpy) port of the FT8/FT4/WSPR protocol code from WSJT-X, with
the decoder hot paths in a small C extension.

Quick start::

    import numpy as np
    import ft8lib

    # Encode a message to a 12 kHz audio waveform
    wave = ft8lib.encode_ft8("CQ K1ABC FN42", f0=1500.0)

    # Decode a 15-second, 12 kHz receive period
    for result in ft8lib.decode_ft8(audio_samples):
        print(result)

    # WSPR: 2-minute periods, 110.6 s transmissions
    wave = ft8lib.encode_wspr("K1ABC FN42 37")
    for result in ft8lib.decode_wspr(audio_2min):
        print(result)
"""

from .crc import crc14, crc14_check
from .decode import (
    Decode,
    decode_ft4,
    decode_ft4_stream,
    decode_ft8,
    decode_ft8_stream,
)
from .encode import (
    encode_ft4,
    encode_ft8,
    ft4_tones_from_bits,
    ft8_tones_from_bits,
    gen_ft4wave,
    gen_ft8wave,
)
from .ldpc import bp_decode, decode174_91, encode174_91, osd_decode
from .pack import HashTable, pack77, unpack77
from .protocol import FT4, FT8, SAMPLE_RATE, WSPR
from .realtime import RealtimeDecoder, decode_realtime
from .subtract import subtract_ft4, subtract_ft8
from .wspr import decode_wspr, encode_wspr, gen_wsprwave, wspr_tones_from_message
from .wspr_pack import WsprHashTable, pack_wspr, unpack_wspr

__version__ = "0.2.0"

__all__ = [
    "FT4",
    "FT8",
    "SAMPLE_RATE",
    "WSPR",
    "Decode",
    "HashTable",
    "RealtimeDecoder",
    "WsprHashTable",
    "__version__",
    "bp_decode",
    "crc14",
    "crc14_check",
    "decode174_91",
    "decode_ft4",
    "decode_ft4_stream",
    "decode_ft8",
    "decode_ft8_stream",
    "decode_realtime",
    "decode_wspr",
    "encode174_91",
    "osd_decode",
    "subtract_ft4",
    "subtract_ft8",
    "encode_ft4",
    "encode_ft8",
    "encode_wspr",
    "ft4_tones_from_bits",
    "ft8_tones_from_bits",
    "gen_ft4wave",
    "gen_ft8wave",
    "gen_wsprwave",
    "pack77",
    "pack_wspr",
    "unpack77",
    "unpack_wspr",
    "wspr_tones_from_message",
]
