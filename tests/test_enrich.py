import numpy as np

from sg_geolocator import enrich as E
from sg_geolocator.raster import building_heights


def _sq(lat, lon, d=0.0003):
    return [[lat, lon], [lat, lon + d], [lat + d, lon + d], [lat + d, lon], [lat, lon]]


def _plan():
    return {"buildings": [
        {"pts": _sq(1.3500, 103.8000), "building": "residential", "addr:housenumber": "221", "addr:street": "Bishan Street 23"},
        {"pts": _sq(1.3510, 103.8000), "building": "residential"},  # an HDB block without an address in OSM
        {"pts": _sq(1.3500, 103.8010), "building": "yes"},  # a condo tower without a height
        {"pts": _sq(1.3500, 103.8020), "building": "yes", "building:levels": "4"},
    ], "water": [], "green": []}


def test_hdb_blocks_give_numbers_streets_and_storeys():
    plan = _plan()
    shrink = lambda r: (np.asarray(r) * 1.0 + np.array([0.00002, 0.00002]))  # official outline a few m off
    blocks = [(shrink(_sq(1.3500, 103.8000, 0.00026)), {"blk": "221", "street_code": "BIS23"}),
              (shrink(_sq(1.3510, 103.8000, 0.00026)), {"blk": "222", "street_code": "BIS23"}),
              (np.asarray(_sq(1.3530, 103.8000)), {"blk": "223", "street_code": "BIS23"})]  # not in the map
    rows = [{"blk_no": "221", "street": "BISHAN ST 23", "max_floor_lvl": "12", "year_completed": "1987"},
            {"blk_no": "222", "street": "BISHAN ST 23", "max_floor_lvl": "25", "year_completed": "2005"},
            {"blk_no": "223", "street": "BISHAN ST 23", "max_floor_lvl": "16", "year_completed": "1990"}]
    stats = E.apply_hdb(plan, blocks, rows, log=lambda *_: None)
    b = plan["buildings"]
    assert stats["matched"] == 2 and stats["added"] == 1 and stats["streets"] == 1
    assert b[0]["height_hdb"] == 39.0 and b[0]["hdb_year"] == "1987"
    # the unaddressed block learns its number and street (street code voted by block 221)
    assert b[1]["addr:housenumber"] == "222" and b[1]["addr:street"] == "BISHAN ST 23" and b[1]["height_hdb"] == 78.0
    assert b[4]["src"] == "hdb" and b[4]["height_hdb"] == 51.0
    assert "height_hdb" not in b[2]


def test_height_estimates_fill_only_missing_heights_and_missing_outlines_are_added():
    plan = _plan()
    polys = [(np.asarray(_sq(1.3500, 103.8010)), 95.0),  # the condo
             (np.asarray(_sq(1.3500, 103.8020)), 40.0),  # tagged 4 levels in OSM: keep the map's own value
             (np.asarray(_sq(1.3540, 103.8050, 0.0004)), 30.0)]  # not in the map at all
    assert E.apply_heights(plan, polys, "gba", log=lambda *_: None) == 1
    assert plan["buildings"][2]["height_est"] == 95.0 and plan["buildings"][2]["height_src"] == "gba"
    assert E.add_missing(plan, [(r, {"height_est": h}) for r, h in polys], "gba", log=lambda *_: None) == 1
    hs = building_heights(plan)
    assert hs[2] == 95.0 and hs[3] == 12.0 and hs[-1] == 30.0


def test_buildings_from_plan_keeps_known_heights_only():
    plan = _plan()
    plan["buildings"][0]["height_hdb"] = 39.0
    plan["buildings"][2]["height_est"] = 95.0
    b = E.buildings_from_plan(plan, log=lambda *_: None)
    assert len(b.labels) == 3  # the unaddressed block with no height is left out
    assert b.labels[0] == "Blk 221 Bishan Street 23"
    assert sorted({float(h) for h in b.heights}) == [12.0, 39.0, 95.0]


def test_height_table_corrects_biased_estimates():
    # an estimator that reads tall blocks at half height and low ones about right
    rng = np.random.default_rng(0)
    truth = rng.uniform(6, 90, 2000)
    est = np.where(truth > 30, truth * 0.5, truth) + rng.normal(0, 1.5, len(truth))
    t = E.HeightTable.fit([(a, b, 0, 3.0) for a, b in zip(truth, est)])
    assert abs(t.predict(35.0, 0, 3.0) - 70) < 12  # a 35 m reading is a ~70 m block
    assert abs(t.predict(12.0, 0, 3.0) - 12) < 4
    assert t.predict(500.0, 9, 3.0) > 0  # unseen type and band still give a value
