"""CLIP wrapper for image and text embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

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


def _as_array(feats) -> np.ndarray:
    # transformers 4.x returns the projected features as a tensor; 5.x wraps them in a
    # model output with the projection stored in `pooler_output`.
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    return feats.cpu().numpy()


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
