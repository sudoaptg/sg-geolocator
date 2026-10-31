"""CLIP wrapper for image and text embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
# CLIP ViT-L/14-336 fine-tuned for image geolocation (Haas et al., 2023). Much slower on CPU.
STREETCLIP = "geolocal/StreetCLIP"


class ClipEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        source = _local_copy(model_name) or model_name
        self.model = CLIPModel.from_pretrained(source).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(source)

    def embed_images(self, paths: list[str | Path]) -> np.ndarray:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return _normalize(_as_array(feats))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return _normalize(_as_array(feats))

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())


def _local_copy(model_name: str) -> str | None:
    """A local copy of a Hugging Face model, for machines that can't reach the Hub: set
    SG_GEOLOCATOR_MODELS to a folder holding e.g. streetclip/ (from the models release)."""
    import os

    root = os.environ.get("SG_GEOLOCATOR_MODELS")
    if not root:
        return None
    path = Path(root) / model_name.split("/")[-1].lower()
    return str(path) if path.is_dir() else None


def _as_array(feats) -> np.ndarray:
    # transformers 4.x returns the projected features as a tensor; 5.x wraps them in a
    # model output with the projection stored in `pooler_output`.
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    return feats.cpu().numpy()


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# Visual place recognition: trained to give the *same spot* similar descriptors across
# viewpoints, lighting and years, rather than describing what kind of scene it is.
# EigenPlaces (Berton et al., ICCV 2023), loaded from its official torch.hub repo.
EIGENPLACES = "eigenplaces/ResNet50/512"
PLACE_SIZE = (512, 384)  # (width, height) fed to the place model


class PlaceEmbedder:
    def __init__(self, model_name: str = EIGENPLACES, device: str | None = None):
        import torch

        family, backbone, dim = model_name.split("/")
        if family != "eigenplaces":
            raise ValueError(f"Unknown place-recognition model: {model_name}")
        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _trust_hub_repos(torch, ["gmberton_eigenplaces", "gmberton_cosplace"])
        # Use the local torch.hub copy when present (e.g. unpacked from the models release):
        # loading by GitHub name re-checks the repo online, which fails without GitHub API access.
        hub = Path(torch.hub.get_dir())
        if (hub / "gmberton_eigenplaces_main").is_dir() and (hub / "gmberton_cosplace_main").is_dir():
            original = torch.hub.load

            def local_load(repo, *args, **kwargs):  # EigenPlaces itself loads CosPlace by name
                if isinstance(repo, str) and repo.startswith("gmberton/"):
                    kwargs.pop("trust_repo", None)
                    return original(str(hub / (repo.replace("/", "_") + "_main")), *args, source="local", **kwargs)
                return original(repo, *args, **kwargs)

            torch.hub.load = local_load
            try:
                self.model = local_load("gmberton/eigenplaces", "get_trained_model", backbone=backbone,
                                        fc_output_dim=int(dim))
            finally:
                torch.hub.load = original
        else:
            self.model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone=backbone,
                                        fc_output_dim=int(dim), trust_repo=True)
        self.model = self.model.to(self.device).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def embed_images(self, paths: list[str | Path]) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        batch = np.stack([
            np.asarray(Image.open(p).convert("RGB").resize(PLACE_SIZE, Image.BILINEAR), dtype=np.float32) / 255.0
            for p in paths
        ])
        x = (torch.from_numpy(batch).permute(0, 3, 1, 2) - self._mean) / self._std
        with torch.no_grad():
            feats = self.model(x.to(self.device))
        return _normalize(feats.cpu().numpy())


def _trust_hub_repos(torch, repos: list[str]) -> None:
    """Pre-approve torch.hub repos. EigenPlaces loads CosPlace through torch.hub without
    trust_repo=True, which otherwise stops at an interactive y/N prompt (and crashes in CI)."""
    path = Path(torch.hub.get_dir()) / "trusted_list"
    path.parent.mkdir(parents=True, exist_ok=True)
    known = set(path.read_text().split()) if path.exists() else set()
    missing = [r for r in repos if r not in known]
    if missing:
        with open(path, "a") as f:
            f.write("".join(f"{r}\n" for r in missing))


def make_embedder(model_name: str = DEFAULT_MODEL):
    """The embedder an index was built with, from the model id stored in it."""
    if model_name.startswith("eigenplaces/"):
        return PlaceEmbedder(model_name)
    return ClipEmbedder(model_name)
