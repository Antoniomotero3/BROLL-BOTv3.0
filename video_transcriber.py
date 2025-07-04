import os
import json
import subprocess
import tempfile
from moviepy import VideoFileClip
from whisper import load_model as load_whisper
import torch
import whisper
import cv2
from tqdm import tqdm
from pathlib import Path

def process_video_to_training_data(video_path, output_dir="frames", frame_interval=1, model_size="base"):
    """
    Transcribes video audio and extracts frames at regular intervals.
    Saves data in JSON format for training the mapper model.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save frames and JSON file.
        frame_interval (int): Number of seconds between frame captures.
        model_size (str): Whisper model size to use (tiny, base, small, medium, large).
    """
    print("🎬 Starting video processing and transcription...")

    os.makedirs(output_dir, exist_ok=True)
    training_data = []

    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size).to(device)

    # Transcribe audio
    result = model.transcribe(video_path, verbose=False)
    segments = result.get("segments", [])

    if not segments:
        print("❌ No transcription segments found.")
        return

    # Load video
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"🎞️ Video FPS: {fps}, Duration: {duration:.2f}s")

    for i, segment in enumerate(tqdm(segments, desc="🧠 Mapping text to frames")):
        text = segment["text"].strip()
        time = segment["start"]
        frame_number = int(time * fps)

        # Set frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if not ret:
            print(f"⚠️ Skipping frame at {time:.2f}s")
            continue

        # Save frame
        frame_path = os.path.join(output_dir, f"scene_{i+1:03d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Append training sample
        training_data.append({"text": text, "image": frame_path})

    video.release()

    # Save training data
    json_path = os.path.join(os.getcwd(), "training_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed {len(training_data)} training samples.")
    print(f"📦 Data saved to {json_path}")

def extract_audio_from_video(video_path, output_path):
    video = VideoFileClip(video_path)
    audio_path = output_path if output_path.endswith(".mp3") else output_path + ".mp3"
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio_whisper(audio_path, model_size="base"):
    model = load_whisper(model_size)
    result = model.transcribe(audio_path)
    return result["segments"]  # Each segment has 'start', 'end', 'text'

def convert_segments_to_srt(segments):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds * 1000) % 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def save_srt_file(segments, output_path):
    srt = convert_segments_to_srt(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt)

def extract_subtitles(video_path, output_dir="subtitles"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(tempfile.gettempdir(), base_name + ".mp3")

    print(f"🔊 Extracting audio from {video_path}...")
    extract_audio_from_video(video_path, audio_path)

    print("📜 Transcribing audio with Whisper...")
    segments = transcribe_audio_whisper(audio_path)

    srt_path = os.path.join(output_dir, base_name + ".srt")
    save_srt_file(segments, srt_path)
    print(f"✅ Subtitles saved to {srt_path}")

    return segments  # Return raw segments for further frame alignment
