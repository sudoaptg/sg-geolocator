from sg_geolocator import reddit
from sg_geolocator.placenames import PlaceNames


def test_place_in_title_prefers_precise_places():
    names = PlaceNames([("Punggol Waterway Park", 1.4086, 103.905)])
    assert reddit.place_in("Sunset at Punggol Waterway Park", names)[3] == "place"
    town = reddit.place_in("Rainy evening in Toa Payoh", names)
    assert town[0] == "Toa Payoh" and town[3] == "area"
    assert reddit.place_in("Clouds", names) is None


def test_image_url_from_post():
    assert reddit.image_url({"url": "https://i.redd.it/abc.jpg"}) == "https://i.redd.it/abc.jpg"
    gallery = {"is_gallery": True, "media_metadata": {"x": {"s": {"u": "https://preview.redd.it/x.jpg"}}}}
    assert reddit.image_url(gallery) == "https://preview.redd.it/x.jpg"
    assert reddit.image_url({"url": "https://www.reddit.com/r/x/comments/1"}) is None
