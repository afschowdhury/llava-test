#!/usr/bin/env python3
"""
Demo launcher for the Video Scene Analyzer.
Automatically loads the demo video and starts the application.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

from video_analyzer import VideoAnalyzer


def main():
    """Launch the video analyzer with the demo video pre-loaded."""
    # Check if demo video exists
    demo_video_path = "video/demo_video.mp4"

    if not os.path.exists(demo_video_path):
        messagebox.showerror(
            "Demo Video Not Found",
            f"Demo video not found at: {demo_video_path}\n\n"
            "Please make sure the video file exists in the video/ folder.",
        )
        return

    # Create and configure the main window
    root = tk.Tk()
    app = VideoAnalyzer(root)

    # Auto-load the demo video
    try:
        app.load_video(demo_video_path)
        print(f"Demo video loaded: {demo_video_path}")
        print("Video Scene Analyzer is ready!")
        print("\nControls:")
        print("- Click 'Play' to start video playback")
        print("- AI will automatically analyze frames every 3 seconds")
        print("- Click 'Analyze Current Frame' for immediate analysis")
        print("- Use the progress bar to seek to different parts")
        print("- Adjust analysis interval as needed")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load demo video: {str(e)}")
        return

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
