import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import clip
import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine


@dataclass
class DetectedObject:
    """Represents a detected object with ID and class"""

    object_id: int
    object_class: str


@dataclass
class KeyframeResult:
    """Represents a keyframe with its metadata"""

    frame_idx: int
    timestamp: float
    reason: str
    frame_data: np.ndarray
    detected_objects: List[DetectedObject]


class VideoKeyframeExtractor:
    """
    Video-based WorldScribe keyframe extraction system that processes video files
    and identifies keyframes using visual analysis and object composition consistency.
    """

    def __init__(
        self,
        n: int = 5,  # frames to check for consistency
        m: int = 3,  # consecutive keyframes for detailed descriptions
        similarity_threshold: float = 0.6,
        motion_threshold: float = 0.3,
        frame_diff_threshold: float = 0.4,
        scene_change_threshold: float = 0.5,
        skip_frames: int = 1,  # Process every nth frame for efficiency
        target_fps: Optional[float] = None,  # Target FPS for processing
    ):
        """
        Initialize the video keyframe extractor.

        Args:
            n: Number of consecutive frames to check for object consistency
            m: Number of consecutive keyframes needed for detailed descriptions
            similarity_threshold: Cosine similarity threshold for frame comparison
            motion_threshold: Threshold for detecting significant camera motion
            frame_diff_threshold: Threshold for frame difference-based scene changes
            scene_change_threshold: Threshold for feature-based scene change detection
            skip_frames: Process every nth frame (1 = every frame, 2 = every other frame, etc.)
            target_fps: Target FPS for processing (None = use original video FPS)
        """
        self.n = n
        self.m = m
        self.similarity_threshold = similarity_threshold
        self.motion_threshold = motion_threshold
        self.frame_diff_threshold = frame_diff_threshold
        self.scene_change_threshold = scene_change_threshold
        self.skip_frames = skip_frames
        self.target_fps = target_fps

        # Initialize CLIP for feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=self.device
        )
        self.clip_model.eval()

        print(f"Using CLIP ViT-B/32 model on {self.device}")

        # Initialize simple object detection (placeholder - replace with your detector)
        self.object_detector = self._init_simple_detector()

        # State tracking
        self.reset()

    def _init_simple_detector(self):
        """Initialize a simple background subtractor as placeholder object detection"""
        return cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    def _simple_object_detection(
        self, frame: np.ndarray, frame_idx: int
    ) -> List[DetectedObject]:
        """
        Simple object detection using background subtraction.
        Replace this with your preferred object detection method (YOLO, etc.)
        """
        try:
            # Apply background subtraction
            fg_mask = self.object_detector.apply(frame)

            # Find contours
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            objects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small objects
                    objects.append(
                        DetectedObject(object_id=i, object_class="moving_object")
                    )

            return objects

        except Exception:
            return []

    def _get_object_composition(
        self, detected_objects: List[DetectedObject]
    ) -> Set[Tuple[int, str]]:
        """Convert detected objects to a set representation for comparison"""
        return {(obj.object_id, obj.object_class) for obj in detected_objects}

    def _calculate_optical_flow_motion(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> float:
        """Calculate camera motion using optical flow"""
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(magnitude) / 255.0
            return min(motion_score, 1.0)
        except Exception:
            return 0.0

    def _calculate_frame_difference(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> float:
        """Calculate frame difference as a measure of visual change"""
        try:
            prev_gray = (
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                if len(prev_frame.shape) == 3
                else prev_frame
            )
            curr_gray = (
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                if len(curr_frame.shape) == 3
                else curr_frame
            )

            diff = cv2.absdiff(prev_gray, curr_gray)
            diff_score = np.mean(diff) / 255.0
            return min(diff_score, 1.0)
        except Exception:
            return 0.0

    def _detect_scene_change_via_features(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> float:
        """Detect scene changes using ORB features and matching"""
        try:
            prev_gray = (
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                if len(prev_frame.shape) == 3
                else prev_frame
            )
            curr_gray = (
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                if len(curr_frame.shape) == 3
                else curr_frame
            )

            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)

            if des1 is None or des2 is None:
                return 1.0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            max_features = max(len(kp1), len(kp2))
            if max_features == 0:
                return 1.0

            match_ratio = len(matches) / max_features
            scene_change_score = 1.0 - match_ratio
            return min(scene_change_score, 1.0)
        except Exception:
            return 1.0

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract CLIP features from a frame"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess the image for CLIP
            image_input = self.clip_preprocess(frame_rgb).unsqueeze(0).to(self.device)

            # Extract features using CLIP
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                # Normalize features (CLIP embeddings are typically normalized)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                return image_features.squeeze().cpu().numpy()

        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return np.zeros(512)  # CLIP ViT-B/32 embedding size

    def _calculate_cosine_similarity(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        try:
            return 1 - cosine(features1, features2)
        except:
            return 0.0

    def _is_keyframe_visual(
        self,
        curr_frame: np.ndarray,
        prev_frame: np.ndarray,
        curr_objects: List[DetectedObject],
        prev_objects: List[DetectedObject],
    ) -> Tuple[bool, str]:
        """Determine if current frame is keyframe based on visual analysis"""
        if prev_frame is None:
            return True, "first_frame"

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Optical flow motion detection
        motion_score = self._calculate_optical_flow_motion(prev_gray, curr_gray)
        if motion_score > self.motion_threshold:
            return True, f"optical_flow_motion_{motion_score:.3f}"

        # Method 2: Frame difference analysis
        diff_score = self._calculate_frame_difference(prev_frame, curr_frame)
        if diff_score > self.frame_diff_threshold:
            return True, f"frame_difference_{diff_score:.3f}"

        # Method 3: Feature-based scene change detection
        scene_change_score = self._detect_scene_change_via_features(
            prev_frame, curr_frame
        )
        if scene_change_score > self.scene_change_threshold:
            return True, f"scene_change_{scene_change_score:.3f}"

        # Method 4: Object composition changes
        prev_composition = self._get_object_composition(prev_objects)
        curr_composition = self._get_object_composition(curr_objects)

        if prev_composition != curr_composition:
            if len(prev_composition) == 0 and len(curr_composition) == 0:
                change_ratio = 0.0
            elif len(prev_composition) == 0 or len(curr_composition) == 0:
                change_ratio = 1.0
            else:
                intersection = prev_composition & curr_composition
                union = prev_composition | curr_composition
                change_ratio = 1.0 - (len(intersection) / len(union))

            if change_ratio > 0.5:
                return True, f"object_composition_change_{change_ratio:.3f}"

        return False, "no_significant_change"

    def _get_change_type(self, reason: str) -> str:
        """Extract the change type from the reason string for organizing outputs"""
        if reason.startswith("scene_change"):
            return "scene_change"
        elif reason.startswith("object_composition_change"):
            return "object_composition_change"
        elif reason.startswith("optical_flow_motion"):
            return "optical_flow_motion"
        elif reason.startswith("frame_difference"):
            return "frame_difference"
        elif reason.startswith("vgg_similarity"):
            return "vgg_similarity"
        elif reason == "first_frame":
            return "first_frame"
        elif reason == "first_vgg_features":
            return "first_vgg_features"
        else:
            return "other"

    def extract_keyframes_from_video(
        self, video_path: str, output_dir: Optional[str] = None
    ) -> List[KeyframeResult]:
        """
        Extract keyframes from a video file.

        Args:
            video_path: Path to the video file
            output_dir: Optional directory to save keyframe images

        Returns:
            List of KeyframeResult objects containing keyframe information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use target FPS if specified, otherwise use original FPS
        fps = self.target_fps if self.target_fps is not None else original_fps

        # Calculate frame skip based on target FPS
        if self.target_fps is not None and self.target_fps < original_fps:
            fps_skip = int(original_fps / self.target_fps)
        else:
            fps_skip = 1

        print(f"Processing video: {video_path}")
        print(
            f"Original FPS: {original_fps}, Target FPS: {fps}, Total frames: {total_frames}"
        )
        if self.target_fps is not None:
            print(f"Frame skip for target FPS: {fps_skip}")

        # Initialize tracking variables
        keyframes = []
        frame_idx = 0
        processed_frame_idx = 0
        prev_frame = None
        prev_objects = []
        prev_keyframe_features = None

        # Create output directories if specified
        change_type_dirs = {}
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # Create subdirectories for different change types
            change_types = [
                "scene_change",
                "object_composition_change",
                "optical_flow_motion",
                "frame_difference",
                "vgg_similarity",
                "first_frame",
                "first_vgg_features",
            ]
            for change_type in change_types:
                change_dir = os.path.join(output_dir, change_type)
                Path(change_dir).mkdir(parents=True, exist_ok=True)
                change_type_dirs[change_type] = change_dir

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames based on both skip_frames and FPS requirements
                if frame_idx % self.skip_frames != 0 or frame_idx % fps_skip != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / fps

                # Detect objects
                detected_objects = self._simple_object_detection(
                    frame.copy(), processed_frame_idx
                )

                # Check if this is a keyframe
                is_keyframe, reason = self._is_keyframe_visual(
                    frame, prev_frame, detected_objects, prev_objects
                )

                # Additional check using VGG features for empty object compositions
                if (
                    not is_keyframe
                    and len(detected_objects) == 0
                    and len(prev_objects) == 0
                ):
                    current_features = self._extract_features(frame)
                    if prev_keyframe_features is not None:
                        similarity = self._calculate_cosine_similarity(
                            current_features, prev_keyframe_features
                        )
                        if similarity < self.similarity_threshold:
                            is_keyframe = True
                            reason = f"vgg_similarity_{similarity:.3f}"
                            prev_keyframe_features = current_features
                    else:
                        is_keyframe = True
                        reason = "first_vgg_features"
                        prev_keyframe_features = current_features

                if is_keyframe:
                    keyframe_result = KeyframeResult(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        reason=reason,
                        frame_data=frame.copy(),
                        detected_objects=detected_objects,
                    )
                    keyframes.append(keyframe_result)

                    # Save keyframe image if output directory specified
                    if output_dir:
                        # Determine the change type for organizing files
                        change_type = self._get_change_type(reason)

                        filename = (
                            f"keyframe_{frame_idx:06d}_{timestamp:.2f}s_{reason}.jpg"
                        )

                        # Save to appropriate subdirectory
                        if change_type in change_type_dirs:
                            filepath = os.path.join(
                                change_type_dirs[change_type], filename
                            )
                        else:
                            # Fallback to main directory for unknown change types
                            filepath = os.path.join(output_dir, filename)

                        cv2.imwrite(filepath, frame)

                    print(
                        f"Keyframe detected: Frame {frame_idx} ({timestamp:.2f}s) - {reason}"
                    )

                # Update for next iteration
                prev_frame = frame.copy()
                prev_objects = detected_objects
                frame_idx += 1
                processed_frame_idx += 1

                # Progress update
                if processed_frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")

        finally:
            cap.release()

        print(f"Extraction complete. Found {len(keyframes)} keyframes.")

        # Display summary of keyframes in each output folder
        if output_dir and os.path.exists(output_dir):
            self._display_output_summary(output_dir)

        return keyframes

    def _display_output_summary(self, output_dir: str):
        """Display summary of keyframes in each output folder"""
        print("\n" + "=" * 60)
        print("OUTPUT FOLDER SUMMARY")
        print("=" * 60)

        total_keyframes = 0

        # Get all subdirectories (change types)
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]

        if not subdirs:
            print("No subdirectories found in output folder.")
            return

        # Sort subdirectories for consistent display
        subdirs.sort()

        print(f"{'Change Type':<25} {'Keyframes':<10} {'Percentage':<10}")
        print("-" * 50)

        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            # Count JPG files in the subdirectory
            jpg_files = [
                f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")
            ]
            count = len(jpg_files)
            total_keyframes += count

            # Calculate percentage (will be calculated after we know total)
            print(f"{subdir:<25} {count:<10}")

        print("-" * 50)
        print(f"{'TOTAL':<25} {total_keyframes:<10}")

        # Display percentages
        if total_keyframes > 0:
            print("\nPercentage Distribution:")
            print("-" * 30)
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, subdir)
                jpg_files = [
                    f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")
                ]
                count = len(jpg_files)
                percentage = (count / total_keyframes) * 100
                print(f"{subdir:<25} {percentage:>6.1f}%")

        print("=" * 60)

    def reset(self):
        """Reset the extractor state"""
        if hasattr(self, "object_detector"):
            self.object_detector = self._init_simple_detector()


def analyze_output_folder(output_dir: str):
    """
    Analyze and display summary of keyframes in an existing output folder.

    Args:
        output_dir: Path to the output directory containing keyframe subdirectories
    """
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return

    print(f"Analyzing output folder: {output_dir}")

    total_keyframes = 0

    # Get all subdirectories (change types)
    subdirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]

    if not subdirs:
        print("No subdirectories found in output folder.")
        return

    # Sort subdirectories for consistent display
    subdirs.sort()

    print("\n" + "=" * 60)
    print("OUTPUT FOLDER ANALYSIS")
    print("=" * 60)
    print(f"{'Change Type':<25} {'Keyframes':<10}")
    print("-" * 40)

    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        # Count JPG files in the subdirectory
        jpg_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")]
        count = len(jpg_files)
        total_keyframes += count

        print(f"{subdir:<25} {count:<10}")

    print("-" * 40)
    print(f"{'TOTAL':<25} {total_keyframes:<10}")

    # Display percentages
    if total_keyframes > 0:
        print("\nPercentage Distribution:")
        print("-" * 30)
        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            jpg_files = [
                f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")
            ]
            count = len(jpg_files)
            percentage = (count / total_keyframes) * 100
            print(f"{subdir:<25} {percentage:>6.1f}%")

    print("=" * 60)


def main(
    video_path: str = None,
    output_dir: str = None,
    target_fps: float = None,
    scene_change_threshold: float = 0.5,
):
    """
    Example usage of the VideoKeyframeExtractor with CLIP

    Args:
        video_path: Path to the video file to process
        output_dir: Directory to save keyframe images (organized by change type)
        target_fps: Target FPS for processing (None = use original video FPS)
    """

    print("VideoKeyframeExtractor with CLIP")
    print("=" * 50)
    print("Requirements: pip install clip-by-openai")
    print("              pip install torch torchvision")
    print("              pip install opencv-python scipy")
    print("=" * 50)
    print("Features:")
    print("- Configurable FPS processing")
    print(
        "- Organized output by change type (scene_change, object_composition_change, etc.)"
    )
    print("=" * 50)

    # Initialize extractor
    extractor = VideoKeyframeExtractor(
        motion_threshold=0.2,  # Adjust based on your video content
        frame_diff_threshold=0.3,  # Adjust based on your video content
        scene_change_threshold=scene_change_threshold,  # Adjust based on your video content
        skip_frames=2,  # Process every 2nd frame for efficiency
        target_fps=target_fps,  # Target FPS for processing
    )

    # Example video processing
    if video_path is None:
        video_path = "video/demo_video.mp4"  # Replace with your video path
    if output_dir is None:
        output_dir = "keyframes_output"  # Directory to save keyframe images

    try:
        keyframes = extractor.extract_keyframes_from_video(video_path, output_dir)

        # Print results
        print(f"\nSummary:")
        print(f"Total keyframes extracted: {len(keyframes)}")

        for i, kf in enumerate(keyframes):
            print(
                f"Keyframe {i+1}: Frame {kf.frame_idx} at {kf.timestamp:.2f}s - {kf.reason}"
            )
            print(f"  Objects detected: {len(kf.detected_objects)}")

        # Analyze keyframe distribution
        if keyframes:
            times = [kf.timestamp for kf in keyframes]
            print("\nKeyframe timing analysis:")
            print(f"First keyframe: {min(times):.2f}s")
            print(f"Last keyframe: {max(times):.2f}s")
            print(
                f"Average interval: {(max(times) - min(times)) / max(1, len(times)-1):.2f}s"
            )

        print("\nUsing CLIP ViT-B/32 for semantic visual understanding")
        print("CLIP provides better scene understanding compared to VGG16")

    except FileNotFoundError:
        print("Please update video_path to point to your video file")
        print("Example usage:")
        print("extractor = VideoKeyframeExtractor()")
        print(
            "keyframes = extractor.extract_keyframes_from_video('path/to/video.mp4', 'output_dir')"
        )
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install clip-by-openai torch torchvision opencv-python scipy")


def advanced_clip_analysis(
    extractor: VideoKeyframeExtractor,
    keyframes: List[KeyframeResult],
    text_queries: List[str] = None,
):
    """
    Advanced analysis using CLIP's text-image understanding capabilities.
    This demonstrates how CLIP can be used for semantic keyframe analysis.
    """
    if text_queries is None:
        text_queries = [
            "a person walking",
            "indoor scene",
            "outdoor scene",
            "close up view",
            "wide angle view",
            "moving objects",
            "static scene",
        ]

    print("\nAdvanced CLIP Analysis")
    print("=" * 30)

    # Encode text queries
    text_inputs = clip.tokenize(text_queries).to(extractor.device)

    with torch.no_grad():
        text_features = extractor.clip_model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Analyze each keyframe
    for i, kf in enumerate(keyframes[:5]):  # Analyze first 5 keyframes as example
        print(f"\nKeyframe {i+1} (Frame {kf.frame_idx}, {kf.timestamp:.2f}s):")

        # Extract image features
        image_features = torch.tensor(extractor._extract_features(kf.frame_data)).to(
            extractor.device
        )

        # Calculate similarities
        similarities = torch.cosine_similarity(
            text_features, image_features.unsqueeze(0)
        )

        # Find best matches
        top_matches = similarities.argsort(descending=True)[:3]

        for j, idx in enumerate(top_matches):
            score = similarities[idx].item()
            print(f"  {j+1}. {text_queries[idx]}: {score:.3f}")


if __name__ == "__main__":
    main(target_fps=1, scene_change_threshold=0.8)
    analyze_output_folder("keyframes_output")
    
