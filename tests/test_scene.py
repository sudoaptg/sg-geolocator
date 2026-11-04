import numpy as np

from sg_geolocator import scene as S
from sg_geolocator.plan2d import to_xy

from test_plan2d import _blocks


def test_render_puts_sky_above_and_buildings_ahead():
    b = _blocks()
    sc = S.MapScene(b, 1.35, 103.80, 2000)
    img = sc.render(to_xy(1.35, 103.80), 1.5, 40.0, 70.0, 0.5, 120, 90)
    assert (img[0] == S.SKY).mean() > 0.5  # top row mostly sky
    assert (img == S.BUILDING).any()  # the 60 m block to the north-east is in view
    assert (img[-1] == S.GROUND).all()


def test_agreement_prefers_the_true_pose():
    b = _blocks()
    sc = S.MapScene(b, 1.35, 103.80, 2000)
    cam = to_xy(1.35, 103.80)
    photo = sc.render(cam, 1.5, 40.0, 70.0, 0.5, 120, 90)
    right = S.agreement(photo, photo)
    wrong = S.agreement(photo, sc.render(cam, 1.5, 200.0, 70.0, 0.5, 120, 90))
    assert right == 1.0 and wrong < 0.9


def test_water_spots_lie_on_water():
    b = _blocks()
    ring = [[1.3490, 103.7990], [1.3490, 103.8000], [1.3495, 103.8000], [1.3495, 103.7990], [1.3490, 103.7990]]
    sc = S.MapScene(b, 1.35, 103.80, 2000, water=[ring])
    spots = S.water_spots(sc, to_xy(1.3492, 103.7995), 200, 10)
    assert len(spots) > 5 and S.inside(sc.water[0], spots).all()


def test_raster_render_matches_the_plan():
    from sg_geolocator import raster as R

    house = [[1.3500, 103.8002], [1.3500, 103.8004], [1.3502, 103.8004], [1.3502, 103.8002], [1.3500, 103.8002]]
    pond = [[1.3490, 103.7995], [1.3490, 103.8005], [1.3494, 103.8005], [1.3494, 103.7995], [1.3490, 103.7995]]
    plan = {"buildings": [{"pts": house, "building:levels": "10"}], "water": [{"pts": pond}], "green": []}
    pr = R.PlanRaster(plan, 1.3497, 103.8000, 300)
    cam = to_xy(1.3497, 103.8000)
    north = pr.render(cam, 2.0, 20.0, 60.0, 0.5, 60, 40)
    south = pr.render(cam, 2.0, 180.0, 60.0, 0.5, 60, 40)
    assert (north == S.BUILDING).any() and not (south == S.BUILDING).any()  # the 30 m block is to the north-east
    assert (south == S.WATER).any()  # the pond is to the south
    assert R.Plan(plan).has("water", 1.3497, 103.8000, 100) and not R.Plan(plan).has("water", 1.36, 103.81, 100)


def test_shore_spots_and_candidate_waters():
    from sg_geolocator import raster as R

    pond = [[1.3490, 103.7990], [1.3490, 103.8010], [1.3500, 103.8010], [1.3500, 103.7990], [1.3490, 103.7990]]
    towers = [{"pts": [[1.3502 + 0.0003 * k, 103.7992], [1.3502 + 0.0003 * k, 103.7994], [1.3503 + 0.0003 * k, 103.7994],
                       [1.3503 + 0.0003 * k, 103.7992], [1.3502 + 0.0003 * k, 103.7992]], "building:levels": "20"}
              for k in range(7)]
    plan = R.Plan({"buildings": towers, "water": [{"pts": pond}], "green": []})
    lakes = R.candidate_waters(plan, min_area=5000)
    assert len(lakes) == 1  # the pond with seven 60 m towers by it
    spots = R.shore_spots(lakes[0], step_m=25, cap=10)
    assert len(spots) == 10


def test_edge_trust_and_rail_cues():
    labels = np.full((10, 10), S.SKY)
    assert S.edges_trustworthy(labels)
    labels[5:] = S.WATER
    labels[:5, :2] = S.GREEN
    assert not S.edges_trustworthy(labels)  # 60% water and greenery
    assert S.rail_in_view({"railing": 0.08, "bridge": 0.06}) and not S.rail_in_view({"building": 0.9})
