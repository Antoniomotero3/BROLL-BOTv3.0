import argparse
import os
import json
from video_transcriber import process_video_to_training_data
from model_trainer import train_model
from intelligent_image_searcher import (
    search_and_rank_images,
    load_trained_mapper_model,
    DEFAULT_MODEL_PATH,
)
from downloader import download_images_for_script

CONFIG_FILE = "config.json"


def load_openai_key():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("openai_api_key")
    return None


def save_openai_key(key: str):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"openai_api_key": key}, f, indent=2)


def cmd_save_key(args):
    save_openai_key(args.key)
    print("✅ OpenAI API key saved")


def cmd_train(args):
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    process_video_to_training_data(args.video)
    train_model()
    print("✅ Training complete")


def cmd_search(args):
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {DEFAULT_MODEL_PATH}. Run train first."
        )
    with open(args.script, "r", encoding="utf-8") as f:
        script_lines = [line.strip() for line in f if line.strip()]

    api_key = load_openai_key()
    mapper = load_trained_mapper_model()
    results = search_and_rank_images(
        script_lines, mapper, args.top_k, api_key
    )
    script_dir, failures = download_images_for_script(results, args.top_k)
    print(f"Images saved to: {script_dir}")
    if failures:
        print(f"{len(failures)} image(s) failed to download")


def main():
    parser = argparse.ArgumentParser(description="BrollBot CLI")
    subparsers = parser.add_subparsers(dest="command")

    key_parser = subparsers.add_parser("save-key", help="Save OpenAI API key")
    key_parser.add_argument("key", help="OpenAI API key")
    key_parser.set_defaults(func=cmd_save_key)

    train_parser = subparsers.add_parser("train", help="Transcribe video and train model")
    train_parser.add_argument("video", help="Path to training video")
    train_parser.set_defaults(func=cmd_train)

    search_parser = subparsers.add_parser("search", help="Search B-roll images")
    search_parser.add_argument("script", help="Path to script text file")
    search_parser.add_argument("--top-k", type=int, default=4, help="Images per scene")
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
