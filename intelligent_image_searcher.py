from sentence_transformers import SentenceTransformer
from image_search_utils import duckduckgo_image_search, wikimedia_image_search
import torch
from torch import nn
import clip
from PIL import Image
import requests
from io import BytesIO
import os

# Load the trained model from default path
DEFAULT_MODEL_PATH = "models/latest_model.pt"


def load_trained_mapper_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the trained text->CLIP mapper model.

    The saved file only contains the model state dict so we need to
    instantiate the architecture used during training which mirrors the
    one defined in :func:`model_trainer.train_model`.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Please train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(384, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Main search and ranking logic
def search_and_rank_images(script_lines, mapper_model, top_k):
    """Search for images and rank them based on semantic similarity."""

    if not isinstance(script_lines, list):
        raise ValueError("script_lines must be a list of strings.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapper_model.to(device)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()
    clip_model.to(device)

    results = []

    for line in script_lines:
        if not line.strip():
            continue

        print(f"\n[INFO] Processing line: {line}")

        # Compute sentence embedding and map to CLIP space
        text_emb = embedder.encode(line, convert_to_tensor=True).to(device)
        mapped_emb = mapper_model(text_emb).detach()
        mapped_emb = mapped_emb / mapped_emb.norm()

        search_query = line.strip()

        # Retrieve candidate image URLs
        try:
            image_urls = duckduckgo_image_search(search_query, max_results=top_k)
            if not image_urls:
                raise ValueError("DuckDuckGo returned no results.")
        except Exception as e:
            print(f"[WARN] DuckDuckGo failed for '{line}': {e}. Trying Wikimedia...")
            try:
                image_urls = wikimedia_image_search(search_query, max_results=top_k)
            except Exception as e:
                print(f"[ERROR] Wikimedia also failed for '{line}': {e}")
                image_urls = []

        scored_urls = []

        for url in image_urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                img_emb = clip_model.encode_image(image_input).squeeze(0)
                img_emb = img_emb / img_emb.norm()
                score = torch.nn.functional.cosine_similarity(img_emb, mapped_emb, dim=0)
                scored_urls.append((score.item(), url))
            except Exception as e:
                print(f"[WARN] Failed to embed image from {url}: {e}")

        scored_urls.sort(key=lambda x: x[0], reverse=True)
        ranked_urls = [u for _, u in scored_urls]

        results.append((line, ranked_urls))

    return results
