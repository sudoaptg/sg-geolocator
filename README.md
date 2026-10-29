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
   token) and the bus stop list from [BusRouter SG](https://busrouter.sg/) (LTA DataMall data).
3. **Landmark recognition (zero-shot):** a CLIP model scores the image against text prompts for
   well-known Singapore landmarks (`src/sg_geolocator/data/landmarks.json`). A confident match
   gives a precise coordinate.
4. **Visual retrieval:** the image is embedded with CLIP and compared against a gallery of
   geotagged reference photos you build yourself. The location is a similarity-weighted average of
   the top-k neighbours' coordinates. This handles ordinary streets, HDB estates, parks and so on,
   which have no famous landmark.

Every prediction reports `lat`, `lon`, a confidence score, which stage produced it, a map link
and, for text clues, the evidence (e.g. `bus_stop: '83139' -> Opp Telok Kurau Pr Sch, Telok Kurau Rd`).

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
- [ ] Map predictions to URA planning areas using official boundary GeoJSON
- [ ] OCR stage for signage (street names, MRT station names, block numbers)
- [ ] Fine-tune CLIP on Singapore imagery (geo-cell classification)
- [ ] Simple web UI showing the prediction on a map

## Notes

Landmark coordinates in `landmarks.json` are approximate (about 100 m). Check them before
relying on them for evaluation.
