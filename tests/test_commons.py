from sg_geolocator import commons


def test_search_subdivides_full_pages(monkeypatch):
    calls = []

    def fake(s, lat, lon, radius):
        calls.append(radius)
        if len(calls) == 1:  # first area is saturated: must be split into 4
            return [{"pageid": i, "title": f"File:{i}.jpg", "lat": lat, "lon": lon} for i in range(commons.MAX_RESULTS)]
        return [{"pageid": 10_000 + len(calls), "title": "File:x.jpg", "lat": lat, "lon": lon}]

    monkeypatch.setattr(commons, "_geosearch", fake)
    monkeypatch.setattr(commons.time, "sleep", lambda _: None)
    found = commons.search(boundary=None, log=lambda *_: None, s=object())
    assert any(r < calls[0] for r in calls[1:])  # smaller circles after the split
    assert all(p >= 10_000 for p in found)  # the saturated page itself is not kept


def test_records_keep_photos_only(monkeypatch):
    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class S:
        def get(self, url, params, timeout):
            return Resp({"query": {"pages": {
                "1": {"imageinfo": [{"mime": "image/jpeg", "thumburl": "https://u/1.jpg", "user": "A"}]},
                "2": {"imageinfo": [{"mime": "image/svg+xml", "thumburl": "https://u/2.png", "user": "B"}]},
            }}})

    monkeypatch.setattr(commons.time, "sleep", lambda _: None)
    found = {1: {"title": "a", "lat": 1.3, "lon": 103.8}, 2: {"title": "b", "lat": 1.3, "lon": 103.8}}
    recs = commons.records(found, log=lambda *_: None, s=S())
    assert [r["id"] for r in recs] == ["commons:1"]
    assert recs[0]["sequence"] == "commons-user:A"
