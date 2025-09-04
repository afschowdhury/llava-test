import base64
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"
ANALYSIS_INTERVAL = 3  # seconds between frame analysis
FRAME_WIDTH = 640
FRAME_HEIGHT = 360


class VideoAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Scene Analyzer")
        self.root.geometry("1400x800")

        # Video properties
        self.video_path = None
        self.cap = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.is_paused = False

        # AI Analysis
        self.last_description = "No analysis yet..."
        self.analysis_queue = queue.Queue()
        self.analysis_thread = None
        self.is_analyzing = False

        # UI Components
        self.setup_ui()

        # Video display properties
        self.video_display_width = 640
        self.video_display_height = 360

    def setup_ui(self):
        """Setup the user interface with split-screen layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # File selection
        ttk.Button(control_frame, text="Select Video", command=self.select_video).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        # Playback controls
        self.play_button = ttk.Button(
            control_frame, text="Play", command=self.toggle_playback
        )
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(control_frame, text="Stop", command=self.stop_video).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.progress_bar.bind("<Button-1>", self.on_progress_click)

        # Time labels
        self.time_label = ttk.Label(control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT)

        # Split screen container
        split_frame = ttk.Frame(main_frame)
        split_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - Video display
        video_frame = ttk.LabelFrame(split_frame, text="Video", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.video_label = ttk.Label(
            video_frame,
            text="Select a video file to begin",
            background="black",
            foreground="white",
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right side - AI Description
        description_frame = ttk.LabelFrame(
            split_frame, text="AI Scene Description", padding=10
        )
        description_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Description text area with scrollbar
        desc_text_frame = ttk.Frame(description_frame)
        desc_text_frame.pack(fill=tk.BOTH, expand=True)

        self.description_text = tk.Text(
            desc_text_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#333333",
            padx=10,
            pady=10,
        )
        scrollbar = ttk.Scrollbar(
            desc_text_frame, orient=tk.VERTICAL, command=self.description_text.yview
        )
        self.description_text.configure(yscrollcommand=scrollbar.set)

        self.description_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Analysis status
        self.status_label = ttk.Label(
            description_frame, text="Ready", foreground="green"
        )
        self.status_label.pack(pady=(10, 0))

        # Analysis controls
        analysis_controls = ttk.Frame(description_frame)
        analysis_controls.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(analysis_controls, text="Analysis Interval (seconds):").pack(
            side=tk.LEFT
        )
        self.interval_var = tk.IntVar(value=ANALYSIS_INTERVAL)
        interval_spinbox = ttk.Spinbox(
            analysis_controls, from_=1, to=10, width=5, textvariable=self.interval_var
        )
        interval_spinbox.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Button(
            analysis_controls,
            text="Analyze Current Frame",
            command=self.analyze_current_frame,
        ).pack(side=tk.RIGHT)

    def select_video(self):
        """Open file dialog to select video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.load_video(file_path)

    def load_video(self, video_path):
        """Load and initialize video file."""
        try:
            # Release previous video if any
            if self.cap:
                self.cap.release()

            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return

            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0

            # Update UI
            self.play_button.config(text="Play")
            self.is_playing = False
            self.is_paused = False

            # Show first frame
            self.show_frame(0)

            # Update time label
            self.update_time_label()

            print(f"Video loaded: {os.path.basename(video_path)}")
            print(f"FPS: {self.fps}, Total frames: {self.total_frames}")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")

    def show_frame(self, frame_number):
        """Display a specific frame in the video label."""
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            # Resize frame to fit display
            frame_resized = cv2.resize(
                frame, (self.video_display_width, self.video_display_height)
            )

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference

            self.current_frame = frame_number
            self.update_progress()

    def update_progress(self):
        """Update progress bar and time label."""
        if self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.progress_var.set(progress)
            self.update_time_label()

    def update_time_label(self):
        """Update time display label."""
        if self.total_frames > 0 and self.fps > 0:
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps

            current_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            total_str = f"{int(total_time//60):02d}:{int(total_time%60):02d}"

            self.time_label.config(text=f"{current_str} / {total_str}")

    def on_progress_click(self, event):
        """Handle click on progress bar to seek to position."""
        if not self.cap or self.total_frames == 0:
            return

        # Calculate clicked position
        progress_bar_width = self.progress_bar.winfo_width()
        click_x = event.x
        click_ratio = click_x / progress_bar_width

        # Seek to frame
        target_frame = int(click_ratio * self.total_frames)
        self.show_frame(target_frame)

    def toggle_playback(self):
        """Toggle video playback."""
        if not self.cap:
            return

        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        """Start video playback."""
        if not self.cap:
            return

        self.is_playing = True
        self.is_paused = False
        self.play_button.config(text="Pause")

        # Start video playback thread
        self.playback_thread = threading.Thread(target=self.video_playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()

        # Start analysis thread if not already running
        if not self.is_analyzing:
            self.start_analysis()

    def pause_video(self):
        """Pause video playback."""
        self.is_playing = False
        self.is_paused = True
        self.play_button.config(text="Play")

    def stop_video(self):
        """Stop video playback and reset to beginning."""
        self.is_playing = False
        self.is_paused = False
        self.play_button.config(text="Play")

        if self.cap:
            self.show_frame(0)

    def video_playback_loop(self):
        """Main video playback loop."""
        frame_delay = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

        while self.is_playing and self.cap:
            if not self.is_paused:
                self.current_frame += 1

                if self.current_frame >= self.total_frames:
                    # End of video
                    self.is_playing = False
                    self.play_button.config(text="Play")
                    break

                # Update UI in main thread
                self.root.after(0, lambda: self.show_frame(self.current_frame))

            time.sleep(frame_delay)

    def start_analysis(self):
        """Start AI analysis thread."""
        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def analysis_loop(self):
        """Main analysis loop that processes frames."""
        last_analysis_time = 0

        while self.is_analyzing:
            if self.cap and self.is_playing:
                current_time = time.time()

                # Check if it's time for analysis
                if current_time - last_analysis_time >= self.interval_var.get():
                    self.analyze_current_frame()
                    last_analysis_time = current_time

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def analyze_current_frame(self):
        """Analyze the current frame and get AI description."""
        if not self.cap:
            return

        # Get current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if not ret:
            return

        # Update status
        self.root.after(
            0,
            lambda: self.status_label.config(text="Analyzing...", foreground="orange"),
        )

        # Start analysis in separate thread
        analysis_thread = threading.Thread(
            target=self.get_ai_description, args=(frame,)
        )
        analysis_thread.daemon = True
        analysis_thread.start()

    def get_ai_description(self, frame):
        """Get AI description for a frame."""
        try:
            # Resize frame for analysis (smaller for faster processing)
            frame_small = cv2.resize(frame, (640, 360))

            # Encode frame to base64
            _, buffer = cv2.imencode(".jpg", frame_small)
            base64_image = base64.b64encode(buffer).decode("utf-8")

            # Send to AI model
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": "Describe the scene in this image in detail. Focus on what's happening, objects, people, and activities.",
                    "stream": False,
                    "images": [base64_image],
                },
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()
            description = data.get("response", "No description found.")

            # Update UI in main thread
            self.root.after(0, lambda: self.update_description(description.strip()))

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with AI: {str(e)}"
            self.root.after(0, lambda: self.update_description(error_msg))
        except Exception as e:
            error_msg = f"Error analyzing frame: {str(e)}"
            self.root.after(0, lambda: self.update_description(error_msg))

    def update_description(self, description):
        """Update the description text area."""
        # Add timestamp
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        timestamp = f"{int(current_time//60):02d}:{int(current_time%60):02d}"

        # Format the description
        formatted_text = f"[{timestamp}] {description}\n\n"

        # Insert at the beginning
        self.description_text.insert(tk.INSERT, formatted_text)

        # Auto-scroll to top
        self.description_text.see(tk.INSERT)

        # Update status
        self.status_label.config(text="Analysis complete", foreground="green")

        # Store last description
        self.last_description = description

    def on_closing(self):
        """Handle application closing."""
        self.is_playing = False
        self.is_analyzing = False

        if self.cap:
            self.cap.release()

        self.root.destroy()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = VideoAnalyzer(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
