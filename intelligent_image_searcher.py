from sentence_transformers import SentenceTransformer
from image_search_utils import duckduckgo_image_search, wikimedia_image_search
import torch
import os

# Load the trained model from default path
DEFAULT_MODEL_PATH = "models/latest_model.pt"
def load_trained_mapper_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please train the model first.")
    model = torch.load(DEFAULT_MODEL_PATH)
    model.eval()
    return model

# Main search and ranking logic
def search_and_rank_images(script_lines, mapper_model, top_k):
    if not isinstance(script_lines, list):
        raise ValueError("script_lines must be a list of strings.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapper_model.to(device)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    final_results = {}

    for line in script_lines:
        if not line.strip():
            continue

        print(f"\n[INFO] Processing line: {line}")

        # Generate embedding (currently unused placeholder)
        embedder.encode([line], convert_to_tensor=True).to(device)

        # For simplicity, use the same text for keyword search
        search_query = line.strip()

        # Try DuckDuckGo first, fallback to Wikimedia
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

        final_results[line] = image_urls

    return final_results
