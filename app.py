import base64
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"
ANALYSIS_INTERVAL = 3  # seconds

# Global variables for video analysis
current_video = None
video_analysis_data = []
analysis_thread = None
is_analyzing = False


class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def get_frame_at_time(self, time_seconds):
        """Get frame at specific time in seconds."""
        if not self.cap:
            return None

        frame_number = int(time_seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            # Resize frame for analysis
            frame_resized = cv2.resize(frame, (640, 360))
            return frame_resized
        return None

    def get_frame_base64(self, time_seconds):
        """Get frame as base64 encoded image."""
        frame = self.get_frame_at_time(time_seconds)
        if frame is not None:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            return frame_base64
        return None

    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


def get_ai_description(frame_base64):
    """Get AI description for a frame."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "This is a first person view of a person working on a construction site. Describe the scene in this image in brief. Focus on what is in front of the person and a short description of the environment.s",
                "stream": False,
                "images": [frame_base64],
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()
        description = data.get("response", "No description found.")
        return description.strip()

    except requests.exceptions.RequestException as e:
        return f"Error communicating with AI: {str(e)}"
    except Exception as e:
        return f"Error analyzing frame: {str(e)}"


def analyze_video_worker(video_analyzer, start_time=0, end_time=None, interval=None):
    """Worker function to analyze video frames."""
    global video_analysis_data, is_analyzing

    if end_time is None:
        end_time = video_analyzer.duration

    if interval is None:
        interval = ANALYSIS_INTERVAL

    current_time = start_time
    # Clear previous analysis data
    video_analysis_data = []

    print(
        f"Starting video analysis from {start_time}s to {end_time}s with {interval}s interval"
    )

    while current_time < end_time and is_analyzing:
        # Get frame at current time
        frame_base64 = video_analyzer.get_frame_base64(current_time)

        if frame_base64:
            # Get AI description
            description = get_ai_description(frame_base64)

            # Store analysis data
            analysis_entry = {
                "timestamp": current_time,
                "time_formatted": f"{int(current_time//60):02d}:{int(current_time%60):02d}",
                "description": description,
                "frame_base64": frame_base64,
            }
            # Add to global analysis data immediately
            video_analysis_data.append(analysis_entry)

            print(
                f"Analyzed frame at {analysis_entry['time_formatted']}: {description[:100]}..."
            )
            print(f"Total results so far: {len(video_analysis_data)}")

        # Move to next analysis point
        current_time += interval

    is_analyzing = False
    print(f"Video analysis complete. Analyzed {len(video_analysis_data)} frames.")
    print(f"Final analysis data: {len(video_analysis_data)} results stored")


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/video/info", methods=["POST"])
def get_video_info():
    """Get video information."""
    global current_video

    try:
        video_path = request.json.get("video_path")
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video file not found"}), 404

        # Create video analyzer
        if current_video:
            current_video.release()

        current_video = VideoAnalyzer(video_path)

        return jsonify(
            {
                "fps": current_video.fps,
                "total_frames": current_video.total_frames,
                "duration": current_video.duration,
                "width": int(current_video.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(current_video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/frame", methods=["POST"])
def get_frame():
    """Get frame at specific time."""
    global current_video

    try:
        time_seconds = float(request.json.get("time", 0))

        if not current_video:
            return jsonify({"error": "No video loaded"}), 400

        frame_base64 = current_video.get_frame_base64(time_seconds)

        if frame_base64:
            return jsonify({"frame": frame_base64})
        else:
            return jsonify({"error": "Could not get frame"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/<path:filename>")
def serve_video(filename):
    """Serve video files."""
    video_path = os.path.join("video", filename)
    if os.path.exists(video_path):
        return send_file(video_path)
    else:
        return jsonify({"error": "Video file not found"}), 404


@app.route("/api/analyze/start", methods=["POST"])
def start_analysis():
    """Start video analysis."""
    global current_video, analysis_thread, is_analyzing

    try:
        if not current_video:
            return jsonify({"error": "No video loaded"}), 400

        if is_analyzing:
            return jsonify({"error": "Analysis already in progress"}), 400

        start_time = float(request.json.get("start_time", 0))
        end_time = request.json.get("end_time")
        interval = float(request.json.get("interval", ANALYSIS_INTERVAL))

        if end_time is not None:
            end_time = float(end_time)

        # Start analysis thread
        is_analyzing = True
        analysis_thread = threading.Thread(
            target=analyze_video_worker,
            args=(current_video, start_time, end_time, interval),
        )
        analysis_thread.daemon = True
        analysis_thread.start()

        return jsonify({"message": "Analysis started", "interval": interval})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/stop", methods=["POST"])
def stop_analysis():
    """Stop video analysis."""
    global is_analyzing

    is_analyzing = False
    return jsonify({"message": "Analysis stopped"})


@app.route("/api/analyze/status")
def get_analysis_status():
    """Get analysis status and results."""
    global is_analyzing, video_analysis_data

    response_data = {
        "is_analyzing": is_analyzing,
        "results": video_analysis_data,
        "total_analyzed": len(video_analysis_data),
    }

    # Debug logging
    if len(video_analysis_data) > 0:
        print(f"Status endpoint: returning {len(video_analysis_data)} results")
        print(f"First result timestamp: {video_analysis_data[0]['timestamp']}")

    return jsonify(response_data)


@app.route("/api/analyze/frame", methods=["POST"])
def analyze_single_frame():
    """Analyze a single frame at specific time."""
    global current_video

    try:
        time_seconds = float(request.json.get("time", 0))

        if not current_video:
            return jsonify({"error": "No video loaded"}), 400

        frame_base64 = current_video.get_frame_base64(time_seconds)

        if not frame_base64:
            return jsonify({"error": "Could not get frame"}), 400

        description = get_ai_description(frame_base64)

        return jsonify(
            {
                "timestamp": time_seconds,
                "time_formatted": f"{int(time_seconds//60):02d}:{int(time_seconds%60):02d}",
                "description": description,
                "frame_base64": frame_base64,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/list")
def list_videos():
    """List available video files."""
    video_dir = "video"
    if not os.path.exists(video_dir):
        return jsonify({"videos": []})

    videos = []
    for file in os.listdir(video_dir):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
            file_path = os.path.join(video_dir, file)
            file_size = os.path.getsize(file_path)
            videos.append(
                {
                    "name": file,
                    "path": file_path,
                    "size": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                }
            )

    return jsonify({"videos": videos})


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

    print("Starting Video Scene Analyzer Web App...")
    print("Make sure Ollama is running with LLaVA model:")
    print("  ollama serve")
    print("  ollama pull llava:7b")
    print("\nAccess the app at: http://localhost:5000")

    app.run(debug=True, host="0.0.0.0", port=5000)
