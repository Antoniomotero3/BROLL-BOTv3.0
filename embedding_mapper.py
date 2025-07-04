import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch import nn, optim
from torchvision import transforms

# Paths
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.pt")
TRAINING_DATA_PATH = os.path.join(os.getcwd(), "training_data.json")

# Load training data
def load_training_data(data_path=TRAINING_DATA_PATH):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Get CLIP image embedding
def get_image_embedding(image_path, clip_model, preprocess):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(next(clip_model.parameters()).device)
    embedding = clip_model.encode_image(image_input).squeeze(0)
    return embedding / embedding.norm()

# Save model to disk
def save_trained_mapper_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Trained model saved to {MODEL_PATH}")

# Load model from disk (or default untrained)
def load_trained_mapper_model(model_path="models/latest_model.pt"):
    model = nn.Sequential()  # instantiate your model architecture
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Train the mapper model
def train_model(data_path=TRAINING_DATA_PATH, epochs=10, lr=1e-4):
    print("🧠 Starting training pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval()

    mapper_model = nn.Sequential(
        nn.Linear(384, 512),
        nn.ReLU(),
        nn.Linear(512, 512)
    ).to(device)

    optimizer = optim.Adam(mapper_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data = load_training_data(data_path)
    print(f"📁 Loaded {len(data)} training samples.")

    for epoch in range(epochs):
        total_loss = 0.0
        for item in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}"):
            text = item["text"]
            image_path = item["image"]

            try:
                text_emb = sentence_model.encode(text, convert_to_tensor=True).to(device).detach().clone().float()
                img_emb = get_image_embedding(image_path, clip_model, preprocess).to(device).detach().clone().float()
            except Exception as e:
                print(f"⚠️ Skipping sample due to error: {e}")
                continue

            pred_emb = mapper_model(text_emb.unsqueeze(0)).squeeze(0)
            loss = loss_fn(pred_emb, img_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📉 Epoch {epoch+1} Loss: {total_loss / len(data):.4f}")

    save_trained_mapper_model(mapper_model)
    print("✅ Training complete.")
