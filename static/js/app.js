class VideoAnalyzer {
    constructor() {
        this.currentVideo = null;
        this.isAnalyzing = false;
        this.analysisInterval = null;
        this.initializeElements();
        this.bindEvents();
        this.loadVideoList();
    }

    initializeElements() {
        // Video elements
        this.videoPlayer = document.getElementById('videoPlayer');
        this.frameCanvas = document.getElementById('frameCanvas');
        this.videoWrapper = document.getElementById('videoWrapper');
        this.videoPlaceholder = document.getElementById('videoPlaceholder');
        this.videoControls = document.getElementById('videoControls');

        // Control elements
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.progressBar = document.getElementById('progressBar');
        this.currentTimeEl = document.getElementById('currentTime');
        this.totalTimeEl = document.getElementById('totalTime');

        // Analysis elements
        this.analyzeFrameBtn = document.getElementById('analyzeFrameBtn');
        this.startAnalysisBtn = document.getElementById('startAnalysisBtn');
        this.stopAnalysisBtn = document.getElementById('stopAnalysisBtn');
        this.analysisIntervalSelect = document.getElementById('analysisInterval');
        this.framesAnalyzedEl = document.getElementById('framesAnalyzed');
        this.analysisResults = document.getElementById('analysisResults');

        // Modal elements
        this.selectVideoBtn = document.getElementById('selectVideoBtn');
        this.videoModal = document.getElementById('videoModal');
        this.closeModal = document.getElementById('closeModal');
        this.videoList = document.getElementById('videoList');
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');

        // Status elements
        this.statusIndicator = document.getElementById('statusIndicator');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
    }

    bindEvents() {
        // Video controls
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.stopBtn.addEventListener('click', () => this.stopVideo());
        this.progressBar.addEventListener('input', () => this.seekVideo());
        this.videoPlayer.addEventListener('timeupdate', () => this.updateProgress());
        this.videoPlayer.addEventListener('loadedmetadata', () => this.onVideoLoaded());

        // Analysis controls
        this.analyzeFrameBtn.addEventListener('click', () => this.analyzeCurrentFrame());
        this.startAnalysisBtn.addEventListener('click', () => this.startAnalysis());
        this.stopAnalysisBtn.addEventListener('click', () => this.stopAnalysis());

        // Modal controls
        this.selectVideoBtn.addEventListener('click', () => this.showVideoModal());
        this.closeModal.addEventListener('click', () => this.hideVideoModal());
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        // Close modal on outside click
        this.videoModal.addEventListener('click', (e) => {
            if (e.target === this.videoModal) {
                this.hideVideoModal();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    async loadVideoList() {
        try {
            const response = await fetch('/api/video/list');
            const data = await response.json();
            this.renderVideoList(data.videos);
        } catch (error) {
            console.error('Error loading video list:', error);
            this.videoList.innerHTML = '<div class="loading">Error loading videos</div>';
        }
    }

    renderVideoList(videos) {
        if (videos.length === 0) {
            this.videoList.innerHTML = '<div class="loading">No videos found</div>';
            return;
        }

        this.videoList.innerHTML = videos.map(video => `
            <div class="video-item" onclick="videoAnalyzer.selectVideo('${video.path}')">
                <div class="video-item-info">
                    <h3>${video.name}</h3>
                    <p>${video.size_mb} MB</p>
                </div>
                <i class="fas fa-play-circle video-item-icon"></i>
            </div>
        `).join('');
    }

    showVideoModal() {
        this.videoModal.classList.add('show');
        this.loadVideoList();
    }

    hideVideoModal() {
        this.videoModal.classList.remove('show');
    }

    async selectVideo(videoPath) {
        this.hideVideoModal();
        this.showLoading('Loading video...');

        try {
            const response = await fetch('/api/video/info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_path: videoPath })
            });

            if (!response.ok) {
                throw new Error('Failed to load video');
            }

            const videoInfo = await response.json();
            this.currentVideo = { ...videoInfo, path: videoPath };
            
            // Set video source - use the video serving endpoint
            const videoFilename = videoPath.split('/').pop();
            const videoUrl = `/video/${videoFilename}`;
            console.log('Setting video source to:', videoUrl);
            
            this.videoPlayer.src = videoUrl;
            this.videoPlayer.style.display = 'block';
            this.videoPlaceholder.style.display = 'none';
            this.videoControls.style.display = 'flex';

            this.updateStatus('Video loaded', 'ready');
            this.hideLoading();

        } catch (error) {
            console.error('Error loading video:', error);
            this.updateStatus('Error loading video', 'error');
            this.hideLoading();
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            // For now, we'll just show a message
            // In a real implementation, you'd upload the file to the server
            alert('File upload functionality would be implemented here. Please place your video file in the video/ folder and refresh the page.');
        }
    }

    onVideoLoaded() {
        this.totalTimeEl.textContent = this.formatTime(this.videoPlayer.duration);
        this.progressBar.max = this.videoPlayer.duration;
    }

    togglePlayPause() {
        if (this.videoPlayer.paused) {
            this.videoPlayer.play();
            this.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
        } else {
            this.videoPlayer.pause();
            this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        }
    }

    stopVideo() {
        this.videoPlayer.pause();
        this.videoPlayer.currentTime = 0;
        this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        this.updateProgress();
    }

    seekVideo() {
        this.videoPlayer.currentTime = this.progressBar.value;
    }

    updateProgress() {
        if (this.videoPlayer.duration) {
            this.progressBar.value = this.videoPlayer.currentTime;
            this.currentTimeEl.textContent = this.formatTime(this.videoPlayer.currentTime);
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    async analyzeCurrentFrame() {
        if (!this.currentVideo) {
            alert('Please select a video first');
            return;
        }

        this.updateStatus('Analyzing frame...', 'analyzing');
        
        try {
            const response = await fetch('/api/analyze/frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ time: this.videoPlayer.currentTime })
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const result = await response.json();
            this.addAnalysisResult(result);
            this.updateStatus('Analysis complete', 'ready');

        } catch (error) {
            console.error('Error analyzing frame:', error);
            this.updateStatus('Analysis failed', 'error');
        }
    }

    async startAnalysis() {
        if (!this.currentVideo) {
            alert('Please select a video first');
            return;
        }

        this.isAnalyzing = true;
        this.startAnalysisBtn.style.display = 'none';
        this.stopAnalysisBtn.style.display = 'inline-flex';
        this.updateStatus('Starting analysis...', 'analyzing');
        this.showLoading('Starting video analysis...');

        try {
            const interval = parseInt(this.analysisIntervalSelect.value);
            console.log('Starting analysis with interval:', interval, 'seconds');
            console.log('Video duration:', this.currentVideo.duration, 'seconds');
            
            const response = await fetch('/api/analyze/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    start_time: 0,
                    end_time: this.currentVideo.duration,
                    interval: interval
                })
            });

            if (!response.ok) {
                throw new Error('Failed to start analysis');
            }

            const result = await response.json();
            console.log('Analysis start response:', result);

            this.hideLoading();
            this.updateStatus('Analysis in progress...', 'analyzing');
            this.startPollingAnalysis();

            // Start video playback automatically
            if (this.videoPlayer.paused) {
                this.videoPlayer.play();
                this.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                console.log('Video playback started automatically');
            }

        } catch (error) {
            console.error('Error starting analysis:', error);
            this.updateStatus('Failed to start analysis', 'error');
            this.hideLoading();
            this.stopAnalysis();
        }
    }

    stopAnalysis() {
        this.isAnalyzing = false;
        this.startAnalysisBtn.style.display = 'inline-flex';
        this.stopAnalysisBtn.style.display = 'none';
        this.updateStatus('Analysis stopped', 'ready');

        // Pause video when analysis is stopped
        if (!this.videoPlayer.paused) {
            this.videoPlayer.pause();
            this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            console.log('Video playback paused when analysis stopped');
        }

        fetch('/api/analyze/stop', { method: 'POST' })
            .catch(error => console.error('Error stopping analysis:', error));
    }

    startPollingAnalysis() {
        this.analysisInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/analyze/status');
                const data = await response.json();

                console.log('Polling response:', data);
                this.framesAnalyzedEl.textContent = data.total_analyzed;

                if (data.results && data.results.length > 0) {
                    console.log(`Found ${data.results.length} results, checking for new ones...`);
                    // Add new results (check for new ones)
                    data.results.forEach(result => {
                        if (!this.analysisResults.querySelector(`[data-timestamp="${result.timestamp}"]`)) {
                            console.log('Adding new result:', result.timestamp, result.description.substring(0, 50));
                            this.addAnalysisResult(result);
                        }
                    });
                }

                // Update status based on analysis state
                if (data.is_analyzing) {
                    this.updateStatus(`Analyzing... (${data.total_analyzed} frames)`, 'analyzing');
                } else if (this.isAnalyzing) {
                    this.stopAnalysis();
                    this.updateStatus('Analysis complete', 'ready');
                    console.log('Analysis completed successfully');
                }

            } catch (error) {
                console.error('Error polling analysis status:', error);
            }
        }, 1000);
    }

    addAnalysisResult(result) {
        // Remove empty state if it exists
        const emptyState = this.analysisResults.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const analysisItem = document.createElement('div');
        analysisItem.className = 'analysis-item';
        analysisItem.setAttribute('data-timestamp', result.timestamp);
        
        analysisItem.innerHTML = `
            <div class="analysis-item-header">
                <span class="analysis-timestamp">${result.time_formatted}</span>
            </div>
            <div class="analysis-description">${result.description}</div>
        `;

        // Insert at the top
        this.analysisResults.insertBefore(analysisItem, this.analysisResults.firstChild);
    }

    updateStatus(message, type) {
        this.statusIndicator.className = `status-indicator ${type}`;
        this.statusIndicator.querySelector('span').textContent = message;
    }

    showLoading(text) {
        this.loadingText.textContent = text;
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    handleKeyboard(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }

        switch (event.code) {
            case 'Space':
                event.preventDefault();
                this.togglePlayPause();
                break;
            case 'Escape':
                this.hideVideoModal();
                break;
        }
    }
}

// Initialize the application
const videoAnalyzer = new VideoAnalyzer();
