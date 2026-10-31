import numpy as np

from sg_geolocator.areasearch import candidate_spots


def _transit():
    return {"bus": {"36": [[[1.300, 103.900], [1.300, 103.905]]]},
            "rail": [{"line": "Thomson-East Coast Line", "elevated": False, "underground": True, "pts": [[1.301, 103.900], [1.301, 103.902]]},
                     {"line": "East West Line", "elevated": True, "underground": False, "pts": [[1.320, 103.900], [1.320, 103.905]]}],
            "stations": []}


def test_bus_route_near_line():
    s = candidate_spots(_transit(), bus="36", line="thomson", line_radius_m=150, step_m=30)
    assert len(s) > 0 and (s[:, 1] <= 103.9025).all()  # only the part of the route by the line


def test_area_near_elevated_track():
    s = candidate_spots(_transit(), area=(1.31, 103.9025, 1500), elevated_radius_m=100, step_m=50)
    assert len(s) > 0 and (np.abs(s[:, 0] - 1.320) < 0.0015).all()
