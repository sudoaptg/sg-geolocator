# sg-geolocator

Estimate **where in Singapore** a photo was taken, using visual cues such as landmarks,
architecture (HDB blocks, shophouses, CBD towers), signage, road markings and vegetation.

## How it works

The pipeline runs three stages and returns the most confident answer:

1. **EXIF GPS:** if the image still carries GPS metadata, use it directly. This is the ground truth.
2. **Landmark recognition (zero-shot):** a CLIP model scores the image against text prompts for
   well-known Singapore landmarks (`src/sg_geolocator/data/landmarks.json`). A confident match
   gives a precise coordinate.
3. **Visual retrieval (kNN):** the image is embedded with CLIP and compared against a gallery of
   geotagged reference photos you build yourself. The location is a similarity-weighted average of
   the top-k neighbours' coordinates. This handles ordinary streets, HDB estates, parks and so on,
   which have no famous landmark.

Every prediction reports `lat`, `lon`, a confidence score, which stage produced it, and the nearest
known area.

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
