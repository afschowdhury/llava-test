#!/usr/bin/env python3
"""
Web Video Scene Analyzer Launcher
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser


def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            llava_models = [m for m in models if "llava" in m.get("name", "").lower()]
            if llava_models:
                print(
                    f"✓ Ollama is running with LLaVA model: {llava_models[0]['name']}"
                )
                return True
            else:
                print("⚠ Ollama is running but LLaVA model not found")
                print("  Run: ollama pull llava:7b")
                return False
        else:
            print("✗ Ollama is not responding")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        return False


def open_browser():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open("http://localhost:5000")


def main():
    print("🎥 Video Scene Analyzer - Web Application")
    print("=" * 50)

    # Check if video directory exists
    if not os.path.exists("video"):
        print("✗ Video directory not found")
        print("  Please create a 'video' directory and add your video files")
        return

    # Check for video files
    video_files = [
        f
        for f in os.listdir("video")
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv"))
    ]
    if not video_files:
        print("⚠ No video files found in video/ directory")
        print("  Please add video files to the video/ directory")
    else:
        print(f"✓ Found {len(video_files)} video file(s): {', '.join(video_files)}")

    # Check Ollama
    if not check_ollama():
        print("\n⚠ Ollama setup required:")
        print("  1. Install Ollama: https://ollama.ai/")
        print("  2. Start Ollama: ollama serve")
        print("  3. Pull LLaVA model: ollama pull llava:7b")
        print("\n  The web app will still start, but AI analysis won't work.")

    print("\n🚀 Starting web application...")
    print("  URL: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start Flask app
    try:
        from app import app

        app.run(debug=False, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure you have installed the requirements:")
        print("  pip install -r requirements_web.txt")
    except Exception as e:
        print(f"✗ Error starting application: {e}")


if __name__ == "__main__":
    main()
