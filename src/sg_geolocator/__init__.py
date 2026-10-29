"""Estimate where in Singapore a photo was taken."""

from .geo import Prediction
from .pipeline import Geolocator

__all__ = ["Geolocator", "Prediction"]
