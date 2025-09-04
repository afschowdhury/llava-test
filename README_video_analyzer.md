# Video Scene Analyzer

A split-screen application that plays video on one side and shows AI-generated scene descriptions on the other side.

## Features

- **Split-screen UI**: Video playback on the left, AI descriptions on the right
- **Real-time Analysis**: Automatically analyzes video frames using LLaVA AI model
- **Interactive Controls**: Play, pause, stop, seek through video
- **Configurable Analysis**: Adjust analysis interval (1-10 seconds)
- **Manual Analysis**: Analyze current frame on demand
- **Progress Tracking**: Visual progress bar and time display
- **Scrollable Descriptions**: View all previous analyses with timestamps

## Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install -r requirements_video_analyzer.txt
   ```

2. **Ollama with LLaVA Model**:
   ```bash
   # Install Ollama (if not already installed)
   # Visit: https://ollama.ai/
   
   # Pull the LLaVA model
   ollama pull llava:7b
   
   # Start Ollama server
   ollama serve
   ```

## Usage

### Quick Start with Demo Video
```bash
python run_demo.py
```
This automatically loads the `video/demo_video.mp4` file and starts the application.

### General Usage
```bash
python video_analyzer.py
```
Then use the "Select Video" button to choose any video file.

## Interface Overview

### Left Side - Video Player
- Video display area
- Playback controls (Play/Pause, Stop)
- Progress bar (click to seek)
- Time display (current/total)

### Right Side - AI Analysis
- Real-time scene descriptions
- Analysis status indicator
- Configurable analysis interval
- "Analyze Current Frame" button
- Scrollable history of all analyses

## Controls

- **Play/Pause**: Start or pause video playback
- **Stop**: Stop playback and return to beginning
- **Progress Bar**: Click anywhere to seek to that position
- **Analysis Interval**: Set how often frames are analyzed (1-10 seconds)
- **Analyze Current Frame**: Get immediate analysis of the current frame

## Video Format Support

Supports common video formats:
- MP4, AVI, MOV, MKV, WMV
- Any format supported by OpenCV

## Technical Details

- **AI Model**: LLaVA:7b (via Ollama)
- **Frame Analysis**: Resized to 640x360 for optimal processing
- **Display Resolution**: 640x360 for video preview
- **Analysis Threading**: Non-blocking AI analysis
- **Memory Management**: Efficient frame handling

## Troubleshooting

### "Unable to connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check if LLaVA model is installed: `ollama list`
- Verify Ollama is accessible at `http://localhost:11434`

### "Could not open video file"
- Check file path and permissions
- Ensure video format is supported
- Try with a different video file

### Performance Issues
- Reduce analysis interval for less frequent analysis
- Use smaller video files for testing
- Close other applications to free up resources

## Demo Video

The included `video/demo_video.mp4` is a construction POV video showing:
- **Duration**: ~14 minutes
- **Resolution**: 1920x1080
- **Content**: Building a wall for a house
- **Perfect for testing**: Contains various scenes, tools, and activities

## Example Output

The AI will generate descriptions like:
```
[02:15] A person is working with construction materials, specifically wooden boards and tools. The scene shows a construction site with various building materials scattered around.

[02:18] The person is using a power drill to attach wooden boards together. There are construction tools and materials visible in the background.

[02:21] Close-up view of construction work with hands manipulating wooden boards and using construction tools.
```

Enjoy analyzing your videos with AI-powered scene understanding!
