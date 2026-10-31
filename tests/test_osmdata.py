import pytest

osmium = pytest.importorskip("osmium")

from sg_geolocator import osmdata  # noqa: E402

OSM = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" lat="1.3500" lon="103.8500" version="1"><tag k="shop" v="mall"/><tag k="name" v="Pearl Mall"/><tag k="name:zh" v="珍珠坊"/></node>
  <node id="2" lat="1.3600" lon="103.8600" version="1"/>
  <node id="3" lat="1.3600" lon="103.8610" version="1"/>
  <node id="4" lat="1.3610" lon="103.8610" version="1"/>
  <node id="5" lat="1.3610" lon="103.8600" version="1"/>
  <node id="6" lat="1.3700" lon="103.8700" version="1"/>
  <node id="7" lat="1.3710" lon="103.8710" version="1"/>
  <node id="8" lat="5.0000" lon="100.0000" version="1"><tag k="shop" v="mall"/></node>
  <way id="10" version="1"><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="5"/><nd ref="2"/>
    <tag k="building" v="residential"/><tag k="addr:housenumber" v="123"/><tag k="addr:street" v="Ang Mo Kio Avenue 3"/>
    <tag k="building:levels" v="12"/></way>
  <way id="11" version="1"><nd ref="6"/><nd ref="7"/><tag k="railway" v="subway"/></way>
  <way id="12" version="1"><nd ref="6"/><nd ref="7"/><tag k="railway" v="subway"/><tag k="tunnel" v="yes"/></way>
  <way id="13" version="1"><nd ref="2"/><nd ref="3"/><nd ref="4"/></way>
  <way id="14" version="1"><nd ref="4"/><nd ref="5"/><nd ref="2"/></way>
  <relation id="20" version="1"><member type="way" ref="13" role="outer"/><member type="way" ref="14" role="outer"/>
    <tag k="type" v="multipolygon"/><tag k="amenity" v="school"/></relation>
</osm>
"""


def test_extract_features_and_buildings(tmp_path):
    path = tmp_path / "t.osm"
    path.write_text(OSM)
    data = osmdata.extract(path, log=lambda *_: None)
    f = data["features"]
    assert len(f["mall"]) == 1  # the one outside Singapore is dropped
    assert len(f["rail"]) == 1  # the tunnel is not visible
    assert len(f["tower_block"]) == 1
    assert len(f["school"]) == 1 and len(f["school"][0]) == 5  # multipolygon assembled into a closed ring
    assert [n[0] for n in data["names"]] == ["Pearl Mall", "珍珠坊"]
    (ring, tags), = data["buildings"]
    assert tags["addr:street"] == "Ang Mo Kio Avenue 3" and len(ring) == 5
