#!/usr/bin/env python3
"""
Test script to demonstrate the new FPS and organized output features
of the VideoKeyframeExtractor.
"""

import os
import sys

from keyframe_extractor_clip import main


def test_different_fps():
    """Test the script with different FPS values"""

    # Example video path (update this to your actual video)
    video_path = "video/demo_video.mp4"  # Update this path

    # Test with different FPS values
    fps_tests = [
        None,  # Use original FPS
        30.0,  # 30 FPS
        15.0,  # 15 FPS
        10.0,  # 10 FPS
        5.0,  # 5 FPS
    ]

    for fps in fps_tests:
        print(f"\n{'='*60}")
        print(f"Testing with FPS: {fps if fps is not None else 'Original'}")
        print(f"{'='*60}")

        # Create output directory for this FPS test
        if fps is not None:
            output_dir = f"keyframes_output_fps_{fps}"
        else:
            output_dir = "keyframes_output_original_fps"

        try:
            # Run the main function with specified parameters
            main(video_path=video_path, output_dir=output_dir, target_fps=fps)

            # Check if output directories were created
            if os.path.exists(output_dir):
                print(f"\nOutput directory created: {output_dir}")

                # List subdirectories (change types)
                subdirs = [
                    d
                    for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d))
                ]
                print(f"Change type subdirectories: {subdirs}")

                # Count files in each subdirectory
                for subdir in subdirs:
                    subdir_path = os.path.join(output_dir, subdir)
                    files = [f for f in os.listdir(subdir_path) if f.endswith(".jpg")]
                    print(f"  {subdir}: {len(files)} keyframes")
            else:
                print(f"Warning: Output directory {output_dir} was not created")

        except FileNotFoundError:
            print(f"Video file not found: {video_path}")
            print("Please update the video_path variable in this script")
            break
        except Exception as e:
            print(f"Error processing with FPS {fps}: {e}")


def show_usage_examples():
    """Show usage examples for the new features"""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)

    print("\n1. Process video with original FPS:")
    print("   main(video_path='my_video.mp4', output_dir='output', target_fps=None)")

    print("\n2. Process video at 15 FPS:")
    print("   main(video_path='my_video.mp4', output_dir='output', target_fps=15.0)")

    print("\n3. Process video at 5 FPS:")
    print("   main(video_path='my_video.mp4', output_dir='output', target_fps=5.0)")

    print("\n4. Direct class usage:")
    print("   from keyframe_extractor_clip import VideoKeyframeExtractor")
    print("   extractor = VideoKeyframeExtractor(target_fps=10.0)")
    print(
        "   keyframes = extractor.extract_keyframes_from_video('video.mp4', 'output')"
    )

    print("\n" + "=" * 60)
    print("OUTPUT ORGANIZATION")
    print("=" * 60)
    print("Keyframes are now organized into subdirectories by change type:")
    print("- scene_change/")
    print("- object_composition_change/")
    print("- optical_flow_motion/")
    print("- frame_difference/")
    print("- vgg_similarity/")
    print("- first_frame/")
    print("- first_vgg_features/")


if __name__ == "__main__":
    print("VideoKeyframeExtractor - FPS and Organized Output Test")
    print("=" * 60)

    # Show usage examples
    show_usage_examples()

    # Ask user if they want to run tests
    response = input("\nDo you want to run FPS tests? (y/n): ").lower().strip()

    if response == "y":
        test_different_fps()
    else:
        print("Skipping tests. Update video_path in the script and run again to test.")
