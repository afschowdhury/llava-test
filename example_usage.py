#!/usr/bin/env python3
"""
Example usage of the enhanced VideoKeyframeExtractor with FPS control
and organized output by change type.
"""

from keyframe_extractor_clip import VideoKeyframeExtractor, analyze_output_folder, main


def example_1_basic_usage():
    """Basic usage with FPS control"""
    print("Example 1: Basic usage with FPS control")
    print("-" * 40)

    # Process video at 15 FPS
    main(
        video_path="your_video.mp4",  # Update this path
        output_dir="keyframes_15fps",
        target_fps=15.0,
    )


def example_2_direct_class_usage():
    """Direct class usage with custom parameters"""
    print("Example 2: Direct class usage with custom parameters")
    print("-" * 40)

    # Create extractor with custom settings
    extractor = VideoKeyframeExtractor(
        motion_threshold=0.3,
        frame_diff_threshold=0.4,
        scene_change_threshold=0.5,
        skip_frames=1,
        target_fps=10.0,  # Process at 10 FPS
    )

    # Extract keyframes
    keyframes = extractor.extract_keyframes_from_video(
        video_path="your_video.mp4", output_dir="keyframes_10fps"  # Update this path
    )

    print(f"Extracted {len(keyframes)} keyframes")
    return keyframes


def example_3_multiple_fps_comparison():
    """Compare results at different FPS values"""
    print("Example 3: Multiple FPS comparison")
    print("-" * 40)

    fps_values = [None, 30.0, 15.0, 10.0, 5.0]
    results = {}

    for fps in fps_values:
        fps_name = f"original" if fps is None else f"{fps}fps"
        output_dir = f"keyframes_{fps_name}"

        print(f"Processing at {fps_name} FPS...")

        try:
            keyframes = main(
                video_path="your_video.mp4",  # Update this path
                output_dir=output_dir,
                target_fps=fps,
            )
            results[fps_name] = len(keyframes) if keyframes else 0
        except Exception as e:
            print(f"Error processing at {fps_name}: {e}")
            results[fps_name] = 0

    # Print comparison
    print("\nFPS Comparison Results:")
    for fps_name, count in results.items():
        print(f"  {fps_name}: {count} keyframes")


def example_4_analyze_change_types():
    """Analyze the distribution of different change types"""
    print("Example 4: Analyze change type distribution")
    print("-" * 40)

    # Process video
    keyframes = main(
        video_path="your_video.mp4",  # Update this path
        output_dir="keyframes_analysis",
        target_fps=15.0,
    )

    if keyframes:
        # Count change types
        change_types = {}
        for kf in keyframes:
            change_type = kf.reason.split("_")[0]  # Get first part of reason
            change_types[change_type] = change_types.get(change_type, 0) + 1

        print("Change type distribution:")
        for change_type, count in sorted(change_types.items()):
            print(f"  {change_type}: {count} keyframes")


def example_5_analyze_existing_output():
    """Analyze an existing output folder"""
    print("Example 5: Analyze existing output folder")
    print("-" * 40)

    # Analyze a specific output folder
    output_folder = "keyframes_output"  # Update this path

    print(f"Analyzing output folder: {output_folder}")
    analyze_output_folder(output_folder)


if __name__ == "__main__":
    print("VideoKeyframeExtractor - Enhanced Usage Examples")
    print("=" * 50)

    print("\nAvailable examples:")
    print("1. Basic usage with FPS control")
    print("2. Direct class usage with custom parameters")
    print("3. Multiple FPS comparison")
    print("4. Analyze change type distribution")
    print("5. Analyze existing output folder")

    choice = input("\nSelect example to run (1-5) or 'all' for all examples: ").strip()

    if choice == "1":
        example_1_basic_usage()
    elif choice == "2":
        example_2_direct_class_usage()
    elif choice == "3":
        example_3_multiple_fps_comparison()
    elif choice == "4":
        example_4_analyze_change_types()
    elif choice == "5":
        example_5_analyze_existing_output()
    elif choice.lower() == "all":
        example_1_basic_usage()
        example_2_direct_class_usage()
        example_3_multiple_fps_comparison()
        example_4_analyze_change_types()
        example_5_analyze_existing_output()
    else:
        print("Invalid choice. Please run the script again and select 1-5 or 'all'.")

    print(
        "\nNote: Update the video_path in the examples to point to your actual video file."
    )
