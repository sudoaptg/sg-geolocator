import numpy as np

from sg_geolocator import layout, raster as R


def _slab(lat, lon):
    return {"pts": [[lat, lon], [lat, lon + 0.0009], [lat + 0.0001, lon + 0.0009], [lat + 0.0001, lon], [lat, lon]],
            "building:levels": "12"}


def test_finds_a_stack_of_parallel_slabs_with_forest_beyond():
    rows = [_slab(1.3500 + 0.0004 * k, 103.8000) for k in range(5)]  # five slabs ~44 m apart, 100 m long
    forest = [[1.3600, 103.7950], [1.3600, 103.8100], [1.3700, 103.8100], [1.3700, 103.7950], [1.3600, 103.7950]]
    plan = R.Plan({"buildings": rows, "water": [], "green": [{"pts": forest}]})
    slabs = layout.slab_blocks(plan)
    assert len(slabs) == 5
    st = layout.stacks(slabs, 4)
    assert st and max(len(s[1]) for s in st) == 5
    views = layout.candidate_views(plan)
    assert views and abs(((views[0][2] + 180) % 360) - 180) < 15  # looking north, towards the forest
