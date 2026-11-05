"""Collect candidate test photos from a subreddit: links and place names only.

Photos stay with their owners: we keep the post link, the image link, the title and the
place the title names (resolved to coordinates). The benchmark downloads images on the fly
when it runs, so nothing is republished. Unauthenticated listing JSON is used; Reddit may
block it from some networks, in which case a Reddit app key would be needed.
"""

from __future__ import annotations

import re
import time

USER_AGENT = "sg-geolocator-benchmark/0.1 (research; github.com/sudoaptg/sg-geolocator)"

# Town and district names often used in photo titles (approximate centres): area-level truth.
TOWNS = {
    "ang mo kio": (1.3691, 103.8454), "bedok": (1.3236, 103.9273), "bishan": (1.3526, 103.8352),
    "bukit batok": (1.3590, 103.7637), "bukit merah": (1.2819, 103.8239), "bukit panjang": (1.3774, 103.7719),
    "bukit timah": (1.3294, 103.8021), "changi": (1.3644, 103.9915), "chinatown": (1.2836, 103.8443),
    "choa chu kang": (1.3840, 103.7470), "clementi": (1.3162, 103.7649), "geylang": (1.3201, 103.8918),
    "holland village": (1.3112, 103.7959), "hougang": (1.3612, 103.8863), "jurong east": (1.3329, 103.7436),
    "jurong west": (1.3404, 103.7090), "kallang": (1.3100, 103.8651), "katong": (1.3050, 103.9050),
    "joo chiat": (1.3122, 103.9010), "little india": (1.3066, 103.8518), "kampong glam": (1.3022, 103.8600),
    "marina bay": (1.2834, 103.8607), "marine parade": (1.3020, 103.9070), "novena": (1.3204, 103.8438),
    "orchard": (1.3048, 103.8318), "pasir ris": (1.3721, 103.9474), "punggol": (1.3984, 103.9072),
    "queenstown": (1.2942, 103.7861), "sembawang": (1.4491, 103.8185), "sengkang": (1.3868, 103.8914),
    "sentosa": (1.2494, 103.8303), "serangoon": (1.3554, 103.8679), "tampines": (1.3496, 103.9568),
    "tanjong pagar": (1.2764, 103.8458), "tiong bahru": (1.2865, 103.8270), "toa payoh": (1.3343, 103.8563),
    "woodlands": (1.4382, 103.7890), "yishun": (1.4304, 103.8354), "clarke quay": (1.2906, 103.8465),
    "raffles place": (1.2840, 103.8514), "dempsey": (1.3048, 103.8100), "telok blangah": (1.2706, 103.8100),
    "redhill": (1.2896, 103.8168), "macpherson": (1.3266, 103.8899), "potong pasir": (1.3313, 103.8688),
}

_IMAGE = re.compile(r"\.(jpe?g|png|webp)(\?|$)", re.I)


def _session():
    """A requests session; with REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET set (a free "script"
    app from reddit.com/prefs/apps), an app-only OAuth token and the oauth.reddit.com host,
    which Reddit doesn't block from cloud servers. Returns (session, base url)."""
    import os

    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
    if cid and secret:
        r = s.post("https://www.reddit.com/api/v1/access_token", auth=(cid, secret),
                   data={"grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        s.headers["Authorization"] = f"bearer {r.json()['access_token']}"
        return s, "https://oauth.reddit.com"
    return s, "https://www.reddit.com"


def fetch_posts(sub: str, listing: str = "top", t: str = "all", pages: int = 5, log=print, session=None) -> list[dict]:
    """Posts from /r/sub/<listing> (up to 100 per page)."""
    out, after = [], None
    s, base = session or _session()
    for _ in range(pages):
        params = {"limit": 100, "t": t, "raw_json": 1}
        if after:
            params["after"] = after
        r = s.get(f"{base}/r/{sub}/{listing}.json" if "oauth" not in base else f"{base}/r/{sub}/{listing}",
                  params=params, timeout=30)
        if r.status_code != 200:
            log(f"  reddit returned {r.status_code} for {listing}: {r.text[:120]!r}")
            break
        data = r.json()["data"]
        out += [c["data"] for c in data["children"]]
        after = data.get("after")
        if not after:
            break
        time.sleep(2)  # be polite
    return out


def image_url(post: dict) -> str | None:
    url = post.get("url_overridden_by_dest") or post.get("url") or ""
    if _IMAGE.search(url) or "i.redd.it" in url:
        return url
    if post.get("is_gallery") and post.get("media_metadata"):  # first gallery image
        for item in post["media_metadata"].values():
            src = (item.get("s") or {}).get("u")
            if src:
                return src
    return None


def place_in(title: str, names=None) -> tuple[str, float, float, str] | None:
    """(place, lat, lon, precision) named in a title: a gazetteer place (precise) wins over
    a town or district (area)."""
    t = " " + re.sub(r"[^a-z0-9 ]", " ", title.lower()) + " "
    best = None
    for name, lat, lon in (names.entries if names else []):
        n = name.lower()
        if len(n) >= 8 and f" {n} " in t and (best is None or len(n) > len(best[0])):
            best = (name, lat, lon, "place")
    if best:
        return best
    for town, (lat, lon) in sorted(TOWNS.items(), key=lambda kv: -len(kv[0])):
        if f" {town} " in t:
            return (town.title(), lat, lon, "area")
    return None


def collect(sub: str, names=None, pages: int = 5, log=print) -> list[dict]:
    seen, out = set(), []
    session = _session()
    log("  using " + ("an OAuth app key" if "oauth" in session[1] else "anonymous access (often refused from cloud servers)"))
    for listing, t in (("top", "all"), ("top", "year"), ("new", "all")):
        posts = fetch_posts(sub, listing, t, pages, log, session)
        log(f"  {listing}/{t}: {len(posts)} posts")
        for p in posts:
            if p.get("id") in seen or p.get("over_18"):
                continue
            seen.add(p.get("id"))
            img = image_url(p)
            where = place_in(p.get("title", ""), names)
            if img and where:
                out.append({"id": p["id"], "title": p.get("title", ""), "post": f"https://www.reddit.com{p.get('permalink', '')}",
                            "image": img, "author": p.get("author", ""), "place": where[0], "lat": round(where[1], 6),
                            "lon": round(where[2], 6), "precision": where[3]})
    return out
