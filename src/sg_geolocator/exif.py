"""Read GPS coordinates from image EXIF metadata."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

GPS_IFD = 0x8825


def _to_degrees(dms) -> float:
    d, m, s = (float(x) for x in dms)
    return d + m / 60.0 + s / 3600.0


def read_gps(path: str | Path) -> tuple[float, float] | None:
    """Return (lat, lon) from EXIF, or None if the image has no usable GPS data."""
    try:
        with Image.open(path) as img:
            gps = img.getexif().get_ifd(GPS_IFD)
    except (OSError, ValueError):
        return None

    # GPS tags: 1=LatRef, 2=Lat, 3=LonRef, 4=Lon
    if not all(k in gps for k in (1, 2, 3, 4)):
        return None
    try:
        lat = _to_degrees(gps[2])
        lon = _to_degrees(gps[4])
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if str(gps[1]).upper().startswith("S"):
        lat = -lat
    if str(gps[3]).upper().startswith("W"):
        lon = -lon
    return lat, lon
