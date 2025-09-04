#!/usr/bin/env python3
"""
Script to analyze existing keyframe output folders and display statistics.
"""

import os
import sys

from keyframe_extractor_clip import analyze_output_folder


def analyze_multiple_folders():
    """Analyze multiple output folders"""
    print("Keyframe Output Folder Analyzer")
    print("=" * 50)

    # Common output folder patterns
    common_folders = [
        "keyframes_output",
        "keyframes_output_original_fps",
        "keyframes_output_fps_30",
        "keyframes_output_fps_15",
        "keyframes_output_fps_10",
        "keyframes_output_fps_5",
        "keyframes_15fps",
        "keyframes_10fps",
        "keyframes_analysis",
    ]

    # Find existing folders
    existing_folders = []
    for folder in common_folders:
        if os.path.exists(folder):
            existing_folders.append(folder)

    if not existing_folders:
        print("No common output folders found.")
        print("Available folders in current directory:")
        all_folders = [d for d in os.listdir(".") if os.path.isdir(d)]
        for folder in sorted(all_folders):
            print(f"  - {folder}")
        return

    print(f"Found {len(existing_folders)} output folders:")
    for i, folder in enumerate(existing_folders, 1):
        print(f"  {i}. {folder}")

    print("\nAnalyzing all found folders...")
    print("=" * 50)

    for folder in existing_folders:
        analyze_output_folder(folder)
        print()  # Add spacing between analyses


def analyze_specific_folder():
    """Analyze a specific folder provided by user"""
    print("Keyframe Output Folder Analyzer")
    print("=" * 50)

    folder_path = input("Enter the path to the output folder: ").strip()

    if not folder_path:
        print("No folder path provided.")
        return

    analyze_output_folder(folder_path)


def interactive_mode():
    """Interactive mode to choose analysis type"""
    print("Keyframe Output Folder Analyzer")
    print("=" * 50)
    print("Choose an option:")
    print("1. Analyze all common output folders")
    print("2. Analyze a specific folder")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            analyze_multiple_folders()
            break
        elif choice == "2":
            analyze_specific_folder()
            break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line argument provided
        folder_path = sys.argv[1]
        analyze_output_folder(folder_path)
    else:
        # Interactive mode
        interactive_mode()
