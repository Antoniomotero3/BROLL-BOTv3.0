import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar, Style
import threading
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

def save_openai_key_to_config(key):
    config_data = {"openai_api_key": key}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print("✅ OpenAI API key saved")

def load_openai_key_from_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("openai_api_key")
    return None

class BrollBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 BrollBot Pro")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("800x600")

        style = Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#333", background="#4CAF50", thickness=20)

        tk.Label(root, text="OpenAI API Key:", fg="white", bg="#1e1e1e").pack()
        self.openai_entry = tk.Entry(root, width=60, bg="#2e2e2e", fg="white")
        self.openai_entry.pack(pady=5)
        tk.Button(root, text="📂 Save API Key", command=self.save_key).pack(pady=3)

        tk.Button(root, text="🎞️ Select Video for Training", command=self.load_video).pack(pady=5)

        tk.Label(root, text="✍️ Paste Your Script Here (1 line = 1 scene)", fg="white", bg="#1e1e1e").pack()
        self.script_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, bg="#2e2e2e", fg="white")
        self.script_input.pack(pady=10)

        tk.Label(root, text="Images per Scene:", fg="white", bg="#1e1e1e").pack()
        self.image_count_spinbox = tk.Spinbox(root, from_=1, to=10, width=5)
        self.image_count_spinbox.pack(pady=5)

        tk.Button(root, text="🚀 Train Model", command=self.start_training).pack(pady=5)
        tk.Button(root, text="🔍 Search B-Roll", command=self.start_search).pack(pady=5)

        self.progress = Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)
        self.status_label = tk.Label(root, text="Idle", fg="white", bg="#1e1e1e")
        self.status_label.pack()

        self.progress['value'] = 0

        self.selected_video = None

    def update_status(self, message, progress=None):
        self.status_label.config(text=message)
        if progress is not None:
            self.progress['value'] = progress

    def save_key(self):
        key = self.openai_entry.get().strip()
        if not key:
            messagebox.showwarning("Missing Key", "Please enter your OpenAI API key.")
            return
        save_openai_key_to_config(key)
        messagebox.showinfo("Success", "✅ OpenAI API key saved!")

    def load_video(self):
        self.selected_video = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
        if self.selected_video:
            self.status_label.config(text=f"Selected: {os.path.basename(self.selected_video)}")

    def start_training(self):
        if not self.selected_video:
            messagebox.showwarning("No Video", "Please select a video file to train from.")
            return
        self.update_status("Starting training...", 0)
        threading.Thread(target=self.run_training).start()

    def start_search(self):
        script_text = self.script_input.get("1.0", tk.END)
        script_lines = [line.strip() for line in script_text.splitlines() if line.strip()]
        if not script_lines:
            messagebox.showwarning(
                "Empty Script",
                "Please paste a script with at least one non-empty line.",
            )
            return

        if not os.path.exists(DEFAULT_MODEL_PATH):
            messagebox.showwarning(
                "Model Missing",
                "No trained model found. Please train the model before searching.",
            )
            return

        self.update_status("Starting search...", 0)

        top_k = int(self.image_count_spinbox.get())
        threading.Thread(target=self.run_search, args=(script_lines, top_k)).start()

    def run_training(self):
        try:
            self.progress['value'] = 0
            process_video_to_training_data(
                self.selected_video, status_callback=self.update_status
            )
            train_model(status_callback=self.update_status)
            self.update_status("✅ Training complete!", 100)
        except Exception as e:
            self.update_status(f"❌ Error during training: {e}")

    def run_search(self, script_lines, top_k):
        try:
            self.progress['value'] = 0
            self.update_status("Searching for images...", 0)
            mapper_model = load_trained_mapper_model()
            api_key = load_openai_key_from_config()
            results = search_and_rank_images(
                script_lines, mapper_model, top_k, api_key, status_callback=self.update_status
            )

            if not results or not all(urls for _, urls in results):
                messagebox.showwarning(
                    "No Results",
                    "One or more script lines produced no image URLs. Download skipped.",
                )
                self.status_label.config(text="No URLs found.")
                return

            script_dir, failures = download_images_for_script(results, top_k)
            self.update_status("✅ Search complete!", 100)
            if failures:
                messagebox.showwarning(
                    "Download Issues",
                    f"{len(failures)} image(s) failed to download. Check the console for details."
                )

        except Exception as e:
            self.update_status(f"❌ Error during search: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrollBotApp(root)
    root.mainloop()
