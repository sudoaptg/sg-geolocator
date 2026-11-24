# sg-geolocator

Estimate **where in Singapore** a photo was taken, using visual cues such as landmarks,
architecture (HDB blocks, shophouses, CBD towers), signage, road markings and vegetation.

## How it works

The pipeline runs these stages and returns the most confident answer:

1. **EXIF GPS:** if the image still carries GPS metadata, use it directly. This is the ground truth.
2. **Text clues (most precise):** signs in the photo are read with free OCR (RapidOCR) and looked
   up in Singapore gazetteers. A **postal code** (one building), **bus stop code** (5,200 stops
   with exact coordinates), **block number + street**, **MRT/LRT station** or **street name**
   can pin the photo to within tens of metres. Several clues that agree raise confidence; a
   stray misread number that disagrees with the rest is outvoted. Lookups use the public
   [OneMap](https://www.onemap.gov.sg/) search API (set `ONEMAP_TOKEN` to use a free account
   token), the bus stop list from [BusRouter SG](https://busrouter.sg/) (LTA DataMall data), and
   the [OpenStreetMap](https://www.openstreetmap.org/copyright) road network (ODbL), which
   corrects misread road names and, when a sign shows two roads, places the photo at the
   junction where they meet.
3. **Exact-spot matching (place recognition):** a dense index of Mapillary photos every ~17 m,
   embedded with [EigenPlaces](https://github.com/gmberton/EigenPlaces), a model trained to
   recognise the *same place* across viewpoints and years. When at least half of the 50
   closest matches agree on one spot (within 100 m), that spot is the answer. On held-out
   Mapillary capture sessions (190k-photo index) that happens for ~19% of photos, and those
   answers are within 50 m 88% of the time and within 1 km 95% of the time; other photos
   fall back to the area guess. Built by the **Build place index** workflow (20 parallel
   jobs; only descriptors are kept). With `model: megaloc` the workflow builds the same
   index with [MegaLoc](https://github.com/gmberton/MegaLoc) (DINOv2-based, trained on
   several place-recognition datasets at once; 8448-d descriptors shrunk to 1024 with PCA
   when the shards are merged). The locate workflows use the EigenPlaces index unless told
   otherwise (`places` input). The **Add Commons photos** workflow adds geotagged
   [Wikimedia Commons](https://commons.wikimedia.org/) photos (schools, MRT stations,
   views from flats), which are taken from viewpoints the car-based Mapillary photos miss.
4. **Landmark recognition (zero-shot):** a CLIP model scores the image against text prompts for
   well-known Singapore landmarks (`src/sg_geolocator/data/landmarks.json`). A confident match
   gives a precise coordinate.
5. **Area guess + map matching:** StreetCLIP plus a trained area classifier gives a
   km-level guess (median ~6 km). The map matcher then predicts from the photo which
   things are within a short walk (school, sports pitch, multi-storey carpark, MRT
   track, expressway, canal, mall, HDB blocks by era, landed houses, ...) and scores every
   ~55 m spot of Singapore by how well [OpenStreetMap](https://www.openstreetmap.org/copyright)
   and HDB's block data (data.gov.sg) agree there. Built by the **Build map matcher**
   workflow, which reports how much it improves on the area guess alone.
   On held-out photos this improves within-1 km from 14% to 18%; the median error is
   unchanged (~5.8 km).
   For elevated views (photos from a flat), an experimental **skyline** step compares the
   outline of surrounding blocks with ones rendered from building footprints and heights
   and lists the best-matching areas as hints. In testing it ranked the true area above
   300 random spots but not above every estate in Singapore, so it does not change the
   guess.
6. **Visual retrieval:** the image is embedded with CLIP and compared against a gallery of
   geotagged reference photos you build yourself. The location is a similarity-weighted average of
   the top-k neighbours' coordinates. This handles ordinary streets, HDB estates, parks and so on,
   which have no famous landmark.

Every prediction reports `lat`, `lon`, a confidence score, which stage produced it, a map link
and, for text clues, the evidence (e.g. `bus_stop: '83139' -> Opp Telok Kurau Pr Sch, Telok Kurau Rd`).

## Pinning the exact spot (plan view first, height second)

Once the area is known, `resect` works out where the camera stood and which way it faced,
in two stages:

1. **Plan view (where and which way).** Long vertical edges in the photo (building sides,
   tower outlines) are each a direction. From every candidate spot (every 4 m) the map gives
   the same kind of list: where the nearest visible building outline changes, nearer
   buildings hiding farther ones. The spot, heading and field of view that line the two
   lists up win, then are refined. Anything you recognise in the photo (`--landmark`) pins
   the heading and sharpens the search a lot. This stage needs building outlines only, not
   heights, and no guess of camera tilt or horizon.
2. **Height (floor).** With the spot and heading fixed, the eye height and camera tilt are
   the only unknowns, read from where the map's building tops sit on the photo's skyline.
   It is marked `"reliable": false` when the skyline does not fit (e.g. towers running out
   of the top of the frame).

```bash
sg-geolocate resect photo.jpg --buildings buildings.npz --names place_names.json.gz \
    --landmark "Lau Pa Sat=545" --landmark "60 Robinson Road:right=205"
```

For a view from a high window add `--window MIN_HEIGHT`: only spots on the outline of
buildings at least that tall are tried, and the floor fit stops at that building's roof.

`--landmark NAME[:left|right|centre]=X` gives the photo column (pixels, or 0-1 across) of
something recognised; `NAME` is `lat,lon`, a building address or a place name. Without
landmarks give `--near` (lat,lon or a place) and `--radius`. On a night photo of Lau Pa Sat
with two landmarks it places the camera at the Boon Tat Street / Raffles Quay corner,
facing north-north-west, in about 20 seconds.

### Render-and-compare and the benchmark

`--render` adds a second check: every pixel of the photo is labelled (sky, building, water,
greenery, ground) by a small segmentation model (SegFormer-B0, published by the **Publish
scene segmenter** workflow; NVIDIA licence, non-commercial), and the map is drawn from each
candidate camera (buildings as walls and roofs, mapped water on the ground). Candidates are
ranked by how much the two agree, with eye height and horizon fitted along the way.
`--water` takes a water layer (the **Export area** workflow with `layers=water`), and
`--on-water` searches only spots on the water, for photos taken at a river's edge:

```bash
sg-geolocate resect lizard.jpg --buildings buildings.npz --water sgwater.json.gz --block 221=825 --on-water --render
```

On that photo (Kallang River, Bishan) edges alone ranked the right Blk 221 third of eleven;
render-and-compare ranks it first.

For drone or other high views of an estate, `--drone` (with `--plan`) first finds every
place where `--slabs` long slab blocks stand one behind another (and a big green area lies
beyond, unless `--no-forest`) from footprints alone, which takes seconds for the whole
island, then ranks those places by render-and-compare from 60-150 m up.

`sg-geolocate benchmark benchmarks/cases.json --images DIR --buildings ...` runs every
confirmed case and reports the error in metres and degrees, so changes are judged on
numbers rather than one example. Cases without a confirmed answer (`benchmarks/hinted.json`:
photos with only an area hint) are run and reported, not scored.

### Block numbers, including stacked ones

`--auto` reads the photo first. A block number painted on a facade (`204`) pins the search
to the twenty or so blocks with that number island-wide, ranked by the plan-view match.
Numbers painted down a column, one digit above another (`2/2/1`), are missed by line-based
text readers, and reading the column as one rotated line drops repeated digits (`221` comes
out as `21`). So character-sized regions (MSER) are grouped into columns, each column's
characters are laid side by side on a clean strip with gaps, and the strip is read; a
misread last character that is really a block letter (`5968` for `596B`) is checked
against the HDB block list. Two stacked numbers on blocks of the same street (`596D`,
`596B`) confirm each other. This takes about a second per photo.

### Map data: HDB blocks and building heights

`enrich-plan` (workflow **Enrich plan map**) adds to the island-wide plan map:

- **HDB Existing Building** (data.gov.sg): all ~13,500 HDB block outlines with block
  numbers. Matched to the OpenStreetMap outlines; street codes are turned into street names
  by the blocks whose OSM address agrees; blocks missing from OSM are added.
- **HDB Property Information** (data.gov.sg): storeys and completion year of every block,
  so HDB heights are storeys x 3 m + 3 m of roof structures.
- **Building heights from GlobalBuildingAtlas** (Zhu et al., ESSD 2025; CC BY-NC 4.0, so
  non-commercial use only): a satellite-derived height for each outline, used for the
  condos, offices and malls that OSM has no height for (20-40% of block-sized buildings
  in central estates).

Heights are taken in that order: HDB storeys, then the map's own height or levels, then the
satellite estimate, then a guess by building type and size.

## Try it on your own photos (on GitHub)

Upload photos to the [`photos/`](photos/) folder (**Add file → Upload files**). The **Locate
photos** workflow guesses each location *without* using GPS stored in the photo, then shows
how far off it was if the photo has GPS, so ordinary phone photos make a good test.
Uploaded photos are public in a public repository, including their GPS.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Predict a single photo (EXIF + landmark stages work out of the box)
sg-geolocate predict path/to/photo.jpg

# Build a retrieval gallery from a folder of geotagged photos (GPS in EXIF)
sg-geolocate build-index path/to/reference_photos/ --out index.npz

# Predict using the gallery as well
sg-geolocate predict path/to/photo.jpg --index index.npz
```

## Building the reference gallery on GitHub

The **Build reference index** workflow (`.github/workflows/build-index.yml`) does everything on
GitHub's runners. No local GPU or downloads are needed.

1. Get a free Mapillary client token: <https://www.mapillary.com/dashboard/developers>, then
   "Register application" and copy the **client token** (it starts with `MLY|`).
2. Add it as a repository secret named `MAPILLARY_TOKEN`
   (Settings → Secrets and variables → Actions → New repository secret).
3. Go to Actions → Build reference index → Run workflow.

The workflow will:

- download Singapore's land boundary (geoBoundaries) so photos from Johor and Batam are dropped
- search Mapillary across the island and thin the results to at most 2 photos per ~220 m grid
  square, each from a different capture session; 360° panoramas are skipped
- embed the photos with CLIP and build `index.npz`
- **evaluate** on held-out capture sessions (median error, % within 0.5/1/5/10 km, compared with
  a "guess the centre" baseline) and show the table on the run's summary page
- publish `index.npz`, `metadata.csv` and `eval.json` as a GitHub **Release**

Then use the released index locally:

```bash
sg-geolocate predict photo.jpg --index index.npz
```

No model training happens here. CLIP is used as-is, and the gallery is what makes it work.
Fine-tuning is a later step, worth doing only if the evaluation numbers show it's needed.

You can also run the same steps locally:

```bash
export MAPILLARY_TOKEN='MLY|...'
sg-geolocate fetch-mapillary --out data/mapillary --max-images 2000
sg-geolocate build-index --metadata data/mapillary/metadata.csv --out index.npz
sg-geolocate evaluate index.npz
```

Other gallery sources: your own phone photos (GPS stays in EXIF by default), using
`sg-geolocate build-index path/to/folder/`.

Reference imagery © Mapillary contributors, licensed CC BY-SA 4.0. Only embeddings and image IDs
are published, not the photos themselves.

## Roadmap

- [x] Collect a reference gallery from Mapillary (GitHub Actions workflow)
- [x] Evaluation: median error (km) and accuracy within 0.5 / 1 / 5 / 10 km
- [x] Map matching: scene features vs OpenStreetMap / HDB data
- [x] Wikimedia Commons photos as extra place references
- [ ] Skyline matching: evaluate on geotagged elevated photos
- [ ] Map predictions to URA planning areas using official boundary GeoJSON
- [x] OCR stage for signage (street names, MRT station names, block numbers)
- [ ] Fine-tune CLIP on Singapore imagery (geo-cell classification)
- [ ] Simple web UI showing the prediction on a map

## Notes

Landmark coordinates in `landmarks.json` are approximate (about 100 m). Check them before
relying on them for evaluation.
