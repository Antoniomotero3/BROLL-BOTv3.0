# BrollBot

This project provides scripts for processing video files, training a mapper model and searching for B-roll images.

## Setup

Install dependencies using pip and ensure `ffmpeg` is installed on your system:

```bash
pip install -r requirements.txt

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Python 3.8+ is recommended. Some packages may require additional system dependencies such as `ffmpeg`.

## Usage

The GUI application can be started with:

```bash
python BROLL-BOT.py
```

Note: running the GUI requires a display environment.

### Command Line Usage

In headless environments you can use the CLI instead of the Tkinter GUI.

Save your OpenAI API key:

```bash
python brollbot_cli.py save-key YOUR_KEY
```

Train the model from a video:

```bash
python brollbot_cli.py train path/to/video.mp4
```

Search for B-roll images based on a text script:

```bash
python brollbot_cli.py search path/to/script.txt --top-k 4
```
