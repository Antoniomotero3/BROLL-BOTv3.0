import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar, Style
import threading
import os
import json

from video_transcriber import process_video_to_training_data
from model_trainer import train_model 
from intelligent_image_searcher import search_and_rank_images, load_trained_mapper_model
from downloader import download_images_for_script

CONFIG_FILE = "config.json"

def save_openai_key_to_config(key):
    config_data = {"openai_api_key": key}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print("✅ OpenAI API key saved")

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

        self.selected_video = None

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
        threading.Thread(target=self.run_training).start()

    def start_search(self):
        script = self.script_input.get("1.0", tk.END).strip()
        if not script:
            messagebox.showwarning("Empty Script", "Please paste a script to search for.")
            return
        top_k = int(self.image_count_spinbox.get())
        threading.Thread(target=self.run_search, args=(script, top_k)).start()

    def run_training(self):
        try:
            self.status_label.config(text="Transcribing video and extracting frames...")
            process_video_to_training_data(self.selected_video)
            self.progress.step(30)

            self.status_label.config(text="Training Model...")
            train_model()
            self.progress.step(70)

            self.status_label.config(text="✅ Training complete!")
        except Exception as e:
            self.status_label.config(text=f"❌ Error during training: {e}")

    def run_search(self, script, top_k):
        try:
            self.status_label.config(text="Searching for images...")
            mapper_model = load_trained_mapper_model()
            script_text = self.script_input.get("1.0", tk.END)
            script_lines = script_text.strip().split('\n')
            results = search_and_rank_images(script_lines, mapper_model, top_k)
            download_images_for_script(results, top_k)
            self.status_label.config(text="✅ Search complete!")

        except Exception as e:
            self.status_label.config(text=f"❌ Error during search: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrollBotApp(root)
    root.mainloop()
