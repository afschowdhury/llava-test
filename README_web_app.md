# 🎥 Video Scene Analyzer - Web Application

A beautiful, modern web application that analyzes video content using AI to generate real-time scene descriptions. Built with Flask, OpenCV, and LLaVA AI model.

## ✨ Features

### 🎬 Video Playback
- **Modern Video Player**: Clean, responsive video player with custom controls
- **Multiple Format Support**: MP4, AVI, MOV, MKV, WMV
- **Seek & Navigate**: Click progress bar to jump to any timestamp
- **Keyboard Shortcuts**: Space to play/pause, Escape to close modals

### 🤖 AI Analysis
- **Real-time Analysis**: Automatically analyzes video frames using LLaVA AI
- **Manual Analysis**: Click "Analyze Frame" for immediate analysis
- **Configurable Intervals**: Set analysis frequency (1-10 seconds)
- **Batch Processing**: Analyze entire video or specific time ranges
- **Live Updates**: See analysis results appear in real-time

### 🎨 Beautiful UI
- **Modern Design**: Clean, professional interface with gradient backgrounds
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Split-Screen View**: Video on left, analysis on right
- **Real-time Status**: Live status indicators and progress tracking
- **Smooth Animations**: Polished interactions and transitions

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Install Python dependencies
pip install -r requirements_web.txt

# Install and setup Ollama
# Visit: https://ollama.ai/
ollama pull llava:7b
ollama serve
```

### 2. Run the Application
```bash
# Quick start with auto-browser opening
python run_web_app.py

# Or run directly
python app.py
```

### 3. Access the App
Open your browser and go to: **http://localhost:5000**

## 📁 Project Structure

```
locomotion-prediction/
├── app.py                 # Main Flask application
├── run_web_app.py        # Launcher script with checks
├── requirements_web.txt   # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Modern CSS styling
│   └── js/
│       └── app.js        # Interactive JavaScript
├── video/                # Place your video files here
│   └── demo_video.mp4    # Demo construction video
└── README_web_app.md     # This file
```

## 🎯 How to Use

### 1. Select a Video
- Click "Select Video" button
- Choose from available videos in the `video/` folder
- Or upload new videos (place in `video/` folder)

### 2. Play and Control
- Use video controls: Play, Pause, Stop
- Click progress bar to seek to specific time
- Use keyboard shortcuts (Space for play/pause)

### 3. AI Analysis
- **Single Frame**: Click "Analyze Frame" for current timestamp
- **Full Video**: Click "Start Analysis" to analyze entire video
- **Custom Range**: Modify start/end times in the API
- **Adjust Frequency**: Change analysis interval (1-10 seconds)

### 4. View Results
- Analysis results appear in real-time on the right panel
- Each result shows timestamp and AI-generated description
- Scroll through all previous analyses
- Results are automatically saved during session

## 🔧 API Endpoints

The web app provides a REST API for programmatic access:

### Video Management
- `GET /api/video/list` - List available videos
- `POST /api/video/info` - Get video information
- `POST /api/video/frame` - Get frame at specific time

### AI Analysis
- `POST /api/analyze/frame` - Analyze single frame
- `POST /api/analyze/start` - Start batch analysis
- `POST /api/analyze/stop` - Stop analysis
- `GET /api/analyze/status` - Get analysis status and results

## 🎨 UI Components

### Video Player
- Custom video controls with modern styling
- Progress bar with click-to-seek functionality
- Time display with current/total duration
- Responsive design that adapts to screen size

### Analysis Panel
- Real-time status indicators
- Configurable analysis settings
- Scrollable results with timestamps
- Empty state with helpful instructions

### Modal System
- Video selection modal with file browser
- Loading overlays with progress indicators
- Responsive design for mobile devices

## 🛠️ Technical Details

### Backend (Flask)
- **Video Processing**: OpenCV for frame extraction and manipulation
- **AI Integration**: HTTP requests to Ollama LLaVA model
- **Threading**: Background analysis to prevent UI blocking
- **Error Handling**: Comprehensive error handling and user feedback

### Frontend (Modern Web)
- **Vanilla JavaScript**: No heavy frameworks, fast and lightweight
- **CSS Grid/Flexbox**: Modern layout techniques
- **Responsive Design**: Mobile-first approach
- **Progressive Enhancement**: Works without JavaScript for basic functionality

### AI Model
- **LLaVA 7B**: Large Language and Vision Assistant
- **Base64 Encoding**: Efficient image transmission
- **Timeout Handling**: Robust error handling for AI requests
- **Batch Processing**: Efficient analysis of multiple frames

## 🔍 Demo Video

The included `demo_video.mp4` is perfect for testing:
- **Content**: Construction POV - Building a wall for a house
- **Duration**: ~14 minutes
- **Resolution**: 1920x1080
- **Scenes**: Various construction activities, tools, materials

## 🚨 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Pull LLaVA model if missing
ollama pull llava:7b
```

### "Video file not found"
- Ensure video files are in the `video/` directory
- Check file permissions
- Supported formats: MP4, AVI, MOV, MKV, WMV

### "Analysis not working"
- Verify Ollama is running: `ollama serve`
- Check LLaVA model: `ollama list`
- Test AI connection: `curl -X POST http://localhost:11434/api/generate -d '{"model":"llava:7b","prompt":"test"}'`

### Performance Issues
- Reduce analysis interval for less frequent analysis
- Use smaller video files for testing
- Close other applications to free up resources
- Consider using a more powerful GPU for faster AI processing

## 🎯 Example Analysis Output

The AI generates detailed scene descriptions like:

```
[02:15] A person is working with construction materials, specifically wooden boards and tools. The scene shows a construction site with various building materials scattered around. The person appears to be measuring and cutting wood.

[02:18] The person is using a power drill to attach wooden boards together. There are construction tools and materials visible in the background, including a saw, measuring tape, and safety equipment.

[02:21] Close-up view of construction work with hands manipulating wooden boards and using construction tools. The scene shows detailed work on building a wall structure.
```

## 🌟 Future Enhancements

- **WebSocket Support**: Real-time bidirectional communication
- **Video Upload**: Direct file upload through web interface
- **Export Results**: Download analysis results as JSON/CSV
- **Multiple Models**: Support for different AI models
- **Batch Processing**: Analyze multiple videos simultaneously
- **Advanced Filtering**: Filter analysis results by keywords
- **User Accounts**: Save and manage analysis sessions

## 📄 License

This project is part of the locomotion prediction research project.

---

**Enjoy analyzing your videos with AI-powered scene understanding!** 🎥🤖✨
