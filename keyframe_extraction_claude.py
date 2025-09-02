import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
import os
from pathlib import Path


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
        skip_frames: int = 1,
    ):  # Process every nth frame for efficiency
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
        """
        self.n = n
        self.m = m
        self.similarity_threshold = similarity_threshold
        self.motion_threshold = motion_threshold
        self.frame_diff_threshold = frame_diff_threshold
        self.scene_change_threshold = scene_change_threshold
        self.skip_frames = skip_frames

        # Initialize VGG16 for feature extraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor.eval()
        self.feature_extractor.classifier = self.feature_extractor.classifier[:-1]
        self.feature_extractor.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
        """Extract VGG16 features from a frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(tensor)
                return features.squeeze().cpu().numpy()
        except Exception:
            return np.zeros(4096)  # VGG16 FC2 size

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
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")

        # Initialize tracking variables
        keyframes = []
        frame_idx = 0
        processed_frame_idx = 0
        prev_frame = None
        prev_objects = []
        prev_keyframe_features = None

        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if specified
                if frame_idx % self.skip_frames != 0:
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
                        filename = (
                            f"keyframe_{frame_idx:06d}_{timestamp:.2f}s_{reason}.jpg"
                        )
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
        return keyframes

    def reset(self):
        """Reset the extractor state"""
        if hasattr(self, "object_detector"):
            self.object_detector = self._init_simple_detector()


def main():
    """Example usage of the VideoKeyframeExtractor"""

    # Initialize extractor
    extractor = VideoKeyframeExtractor(
        motion_threshold=0.2,  # Adjust based on your video content
        frame_diff_threshold=0.3,  # Adjust based on your video content
        scene_change_threshold=0.4,  # Adjust based on your video content
        skip_frames=2,  # Process every 2nd frame for efficiency
    )

    # Example video processing
    video_path = "your_video.mp4"  # Replace with your video path
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
            print(f"\nKeyframe timing analysis:")
            print(f"First keyframe: {min(times):.2f}s")
            print(f"Last keyframe: {max(times):.2f}s")
            print(
                f"Average interval: {(max(times) - min(times)) / max(1, len(times)-1):.2f}s"
            )

    except FileNotFoundError:
        print(f"Please update video_path to point to your video file")
        print("Example usage:")
        print("extractor = VideoKeyframeExtractor()")
        print(
            "keyframes = extractor.extract_keyframes_from_video('path/to/video.mp4', 'output_dir')"
        )


if __name__ == "__main__":
    main()
