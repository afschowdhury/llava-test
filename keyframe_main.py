from keyframe_extraction_claude import VideoKeyframeExtractor

extractor = VideoKeyframeExtractor(
    motion_threshold=0.2,  # Camera motion sensitivity
    frame_diff_threshold=0.3,  # Scene change sensitivity
    scene_change_threshold=0.4,  # Feature matching sensitivity
    skip_frames=2,  # Process every 2nd frame
)

# Extract keyframes
keyframes = extractor.extract_keyframes_from_video(
    video_path="video/demo_video.mp4",
    output_dir="keyframes_output",  # Saves keyframe images here
)

# Results
for kf in keyframes:
    print(f"Keyframe at {kf.timestamp:.2f}s - {kf.reason}")
