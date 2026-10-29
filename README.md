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

## Getting reference data

The retrieval stage is only as good as its gallery. Good sources:

- **Mapillary**: crowd-sourced street-level imagery with coordinates (API, CC BY-SA).
- **Your own photos**: phone pictures keep GPS in EXIF by default.
- **data.gov.sg / URA Master Plan**: planning-area boundaries, useful for reporting results by area.

Aim for even coverage across the island. A grid of roughly 200–500 m spacing is a sensible start.

## Roadmap

- [ ] Collect a reference gallery covering all 55 planning areas
- [ ] Evaluation script: median error (km) and accuracy within 1 km / 5 km
- [ ] Map predictions to URA planning areas using official boundary GeoJSON
- [ ] OCR stage for signage (street names, MRT station names, block numbers)
- [ ] Fine-tune CLIP on Singapore imagery (geo-cell classification)
- [ ] Simple web UI showing the prediction on a map

## Notes

Landmark coordinates in `landmarks.json` are approximate (about 100 m). Check them before
relying on them for evaluation.
