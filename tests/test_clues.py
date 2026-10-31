import pytest

from sg_geolocator.clues import BusStops, ClueLocator, OneMap, combine, extract, Clue, normalize_street
from sg_geolocator.roads import Roads


@pytest.fixture(autouse=True)
def no_road_download(monkeypatch):
    """Tests never download the OSM road network; ones that need roads pass their own."""
    def fail(cls, path=None, log=None):
        raise OSError("no network in tests")
    monkeypatch.setattr(Roads, "load", classmethod(fail))


def _r(searchval, lat, lon, blk="", road="NIL", postal="NIL", address=None):
    # Shape copied from a real OneMap search response.
    return {"SEARCHVAL": searchval, "BLK_NO": blk, "ROAD_NAME": road, "BUILDING": searchval,
            "ADDRESS": address or searchval, "POSTAL": postal, "X": "0", "Y": "0",
            "LATITUDE": str(lat), "LONGITUDE": str(lon)}


class FakeOneMap(OneMap):
    def __init__(self, table):
        self.table = {k.upper(): v for k, v in table.items()}
        self.queries = []

    def search(self, query, pages=1):
        self.queries.append(query.upper())
        return self.table.get(query.upper(), [])

    def save(self):
        pass


BUS = BusStops({"83139": [103.90998, 1.31451, "Opp Telok Kurau Pr Sch", "Telok Kurau Rd"],
                "75009": [103.94265, 1.35376, "Tampines Int", "Tampines Ctrl 1"]})


def test_extract_reads_singapore_sign_formats():
    found = extract(["BLK 123A", "Bus Stop 83139", "Tampines Ave 5", "Singapore 238801", "Tampines MRT"])
    assert found["postal"] == ["238801"]
    assert "83139" in found["bus_stop"]
    assert found["block"] == ["123A"]
    assert "TAMPINES AVENUE 5" in found["street"]
    assert found["station"] == ["TAMPINES"]


def test_extract_ignores_invalid_postal_sectors_and_long_numbers():
    found = extract(["Call 98765432", "S(990123)", "Total 1234567"])
    assert found["postal"] == [] and found["bare_postal"] == []
    assert found["bus_stop"] == []  # 5-digit runs inside longer numbers don't count


def test_normalize_street_expands_abbreviations():
    assert normalize_street("Jln Bt Merah") == "JALAN BUKIT MERAH"
    assert normalize_street("Ang Mo Kio Ave 3") == "ANG MO KIO AVENUE 3"


def test_postal_code_pins_exact_building():
    om = FakeOneMap({"238801": [_r("ION ORCHARD", 1.30398, 103.83203, blk="2", road="ORCHARD TURN",
                                   postal="238801", address="2 ORCHARD TURN ION ORCHARD SINGAPORE 238801")]})
    res = ClueLocator(onemap=om, bus_stops=BUS).locate_text(["ION Orchard", "Singapore 238801"])
    assert res and abs(res.lat - 1.30398) < 1e-4 and res.confidence >= 0.95
    assert "ORCHARD TURN" in res.evidence[0]


def test_bus_stop_code_uses_stop_list_and_rejects_unknown_codes():
    loc = ClueLocator(onemap=FakeOneMap({}), bus_stops=BUS)
    res = loc.locate_text(["83139"])
    assert res and abs(res.lon - 103.90998) < 1e-4 and res.clues[0].kind == "bus_stop"
    assert loc.locate_text(["12345"]) is None


def test_block_needs_matching_block_number_and_prefers_prior():
    om = FakeOneMap({"123 TAMPINES STREET 11": [
        _r("123 TAMPINES STREET 11", 1.3500, 103.9440, blk="123", road="TAMPINES STREET 11"),
        _r("1123 TAMPINES STREET 11", 1.3600, 103.9500, blk="1123", road="TAMPINES STREET 11"),
    ]})
    res = ClueLocator(onemap=om, bus_stops=BUS).locate_text(["BLK 123", "TAMPINES ST 11"])
    assert res and res.clues[0].kind == "block" and abs(res.lat - 1.35) < 1e-4


def test_station_match_requires_exact_name():
    om = FakeOneMap({"TAMPINES MRT STATION": [
        _r("TAMPINES EAST MRT STATION (DT33)", 1.35619, 103.95463),
        _r("TAMPINES MRT STATION (EW2 / DT32)", 1.35345, 103.94512),
    ]})
    res = ClueLocator(onemap=om, bus_stops=BUS).locate_text(["Tampines MRT"])
    assert res and abs(res.lat - 1.35345) < 1e-4


def test_agreeing_clues_beat_a_stray_one_and_raise_confidence():
    clues = [
        Clue("bus_stop", "75009", 1.35376, 103.94265, 0.9, ""),
        Clue("street", "TAMPINES CENTRAL 1", 1.3535, 103.9440, 0.5, ""),
        Clue("bus_stop", "10009", 1.2821, 103.81722, 0.9, ""),  # misread number, far away
    ]
    res = combine(clues)
    assert abs(res.lat - 1.35376) < 1e-4
    assert res.confidence > 0.9 and len(res.clues) == 2


def test_no_text_means_no_clue():
    assert ClueLocator(onemap=FakeOneMap({}), bus_stops=BUS).locate_text([]) is None


def test_extract_handles_ocr_dropped_spaces():
    found = extract(["BusStop83139", "Tampines Ave5", "BLK123A", "S238801"])
    assert "83139" in found["bus_stop"]
    assert "TAMPINES AVENUE 5" in found["street"]
    assert found["block"] == ["123A"]
    assert found["postal"] == ["238801"]


def test_evaluate_text_scores_placed_photos(tmp_path, monkeypatch):
    import csv

    from PIL import Image

    import sg_geolocator.clues as clues_mod
    from sg_geolocator.evaluate import evaluate_text, text_to_markdown

    (tmp_path / "images").mkdir()
    rows = []
    for i in range(40):
        Image.new("RGB", (8, 8)).save(tmp_path / "images" / f"{i}.jpg")
        rows.append({"id": str(i), "lat": 1.31451, "lon": 103.90998, "sequence": f"s{i}", "captured_at": "", "path": f"{i}.jpg"})
    with open(tmp_path / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    class FakeReader:
        def read(self, path):
            return ["BUS STOP 83139"] if int(path.stem) % 2 else []

    monkeypatch.setattr(clues_mod, "TextReader", FakeReader)
    monkeypatch.setattr(clues_mod.BusStops, "load", classmethod(lambda cls, path=None: BUS))
    monkeypatch.setattr(clues_mod, "OneMap", lambda *a, **k: FakeOneMap({}))

    res = evaluate_text(str(tmp_path / "metadata.csv"), test_fraction=1.0, log=lambda *_: None)
    assert res["n"] == 40 and res["with_text"] == 20 and res["placed"] == 20
    assert res["metrics"]["within_0.1km"] == 1.0
    assert res["by_kind"]["bus_stop"]["n"] == 20
    assert "Placed from text: 20" in text_to_markdown(res)


# Two roads crossing at (1.3500, 103.9000), plus a parallel numbered avenue.
ROADS = Roads({
    "YIO CHU KANG ROAD": [[[1.3500, 103.8900], [1.3500, 103.8950], [1.3500, 103.9000], [1.3500, 103.9100]]],
    "SELETAR ROAD": [[[1.3400, 103.9000], [1.3450, 103.9000], [1.3500, 103.9000], [1.3600, 103.9000]]],
    "TAMPINES AVENUE 3": [[[1.3300, 103.9400], [1.3400, 103.9400]]],
    "TAMPINES AVENUE 5": [[[1.3300, 103.9500], [1.3400, 103.9500]]],
})


def test_road_match_corrects_ocr_but_never_changes_numbers():
    assert ROADS.match("YO CHU KANG ROAD") == "YIO CHU KANG ROAD"
    assert ROADS.match("SELETAR ROAD") == "SELETAR ROAD"
    assert ROADS.match("TAMPINES AVENUE 5") == "TAMPINES AVENUE 5"
    assert ROADS.match("TAMPINES AVENUE 7") is None  # no such avenue; 3 and 5 must not match
    assert ROADS.match("TOTALLY DIFFERENT ROAD") is None


def test_road_match_rejects_short_or_different_garbles():
    roads = Roads({n: [[[1.3, 103.8], [1.31, 103.81]]] for n in (
        "BATH ROAD", "ALPS AVENUE", "MAY ROAD", "SEMBAWANG WAY", "HALUS LINK", "SCOTTS ROAD")})
    # Real misreads from the evaluation that were wrongly matched before:
    assert roads.match("BAHRU ROAD") is None       # was BATH ROAD, 13 km off
    assert roads.match("AST AVENUE") is None       # was ALPS AVENUE, 11.6 km off
    assert roads.match("AALY ROAD") is None        # was MAY ROAD, 15.7 km off
    # Real misreads that should still be corrected:
    assert roads.match("SERHAWANG WAY") == "SEMBAWANG WAY"
    assert roads.match("HALOS LINK") == "HALUS LINK"
    assert roads.match("SCOTS ROAD") == "SCOTTS ROAD"
    assert roads.match("SEMBAWANG ROAD") is None   # right core, wrong road type


def test_junction_of_two_roads():
    js = ROADS.junctions("YIO CHU KANG ROAD", "SELETAR ROAD")
    assert len(js) == 1 and abs(js[0][0] - 1.35) < 1e-4 and abs(js[0][1] - 103.90) < 1e-4
    assert ROADS.junctions("TAMPINES AVENUE 3", "TAMPINES AVENUE 5") == []  # parallel, ~1 km apart


def test_two_road_names_on_a_sign_place_the_junction():
    loc = ClueLocator(onemap=FakeOneMap({}), bus_stops=BUS, roads=ROADS)
    res = loc.locate_text(["Yo Chu Kang Rd", "Seletar Rd"])
    assert res.clues[0].kind == "junction"
    assert abs(res.lat - 1.35) < 1e-4 and abs(res.lon - 103.90) < 1e-4
    assert res.confidence > 0.9  # junction plus both street clues agree


def test_single_misread_road_resolves_via_road_network():
    loc = ClueLocator(onemap=FakeOneMap({}), bus_stops=BUS, roads=ROADS)
    res = loc.locate_text(["Seletar Rd"], prior=(1.341, 103.95))
    assert res.clues[0].kind == "street" and abs(res.lat - 1.34) < 1e-4  # point on the road nearest the prior


def test_overpass_json_parsing():
    data = {"elements": [
        {"type": "way", "tags": {"name": "Delta Road", "highway": "tertiary"},
         "geometry": [{"lat": 1.2901234, "lon": 103.8209876}, {"lat": 1.291, "lon": 103.822}]},
        {"type": "way", "tags": {"highway": "service"}, "geometry": [{"lat": 1.3, "lon": 103.8}]},
    ]}
    r = Roads.from_overpass_json(data)
    assert r.names == ["DELTA ROAD"] and r.roads["DELTA ROAD"][0][0] == [1.29012, 103.82099]


def test_named_place_from_garbled_sign():
    from sg_geolocator.clues import ClueLocator, OneMap
    from sg_geolocator.placenames import PlaceNames

    names = PlaceNames([("APSN Tanglin School", 1.3760, 103.8570), ("Tanglin Secondary School", 1.3050, 103.7700)])
    onemap = OneMap(cache_path="/nonexistent/x.json", offline=True)
    loc = ClueLocator(onemap=onemap, bus_stops=BusStops({}), roads=False, names=names)
    res = loc.locate_text(["APEN", "TANGUNSCHOOL"])
    assert res is not None and abs(res.lat - 1.376) < 1e-6
    assert any("APSN Tanglin School" in e for e in res.evidence)
    assert loc.locate_text(["SCHOOL BUS", "NO PARKING"]) is None


def test_tile_offsets_cover_the_image():
    from sg_geolocator.clues import _starts

    assert _starts(1000, 1280, 1024) == [0]
    s = _starts(4096, 1280, 1024)
    assert s[0] == 0 and s[-1] == 4096 - 1280 and all(b - a <= 1024 for a, b in zip(s, s[1:]))
