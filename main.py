import base64
import threading
import time

import cv2
import numpy as np
import requests

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ANALYSIS_INTERVAL = 4  # seconds
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)
LINE_TYPE = 2
CAMERA_TIMEOUT = 5  # seconds

# --- Global Variables ---
last_description = "Starting..."
is_running = True
last_frame = None
frame_lock = threading.Lock()
camera_working = False


def create_test_image():
    """Create a test image when camera is not available."""
    # Create a colorful test pattern
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Add some colored rectangles
    cv2.rectangle(
        img, (0, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT // 2), (255, 0, 0), -1
    )  # Blue
    cv2.rectangle(
        img, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH, FRAME_HEIGHT // 2), (0, 255, 0), -1
    )  # Green
    cv2.rectangle(
        img, (0, FRAME_HEIGHT // 2), (FRAME_WIDTH // 2, FRAME_HEIGHT), (0, 0, 255), -1
    )  # Red
    cv2.rectangle(
        img,
        (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
        (FRAME_WIDTH, FRAME_HEIGHT),
        (255, 255, 0),
        -1,
    )  # Yellow

    # Add text
    cv2.putText(
        img,
        "Test Image - Camera Not Available",
        (50, FRAME_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "Press 'q' to quit",
        (50, FRAME_HEIGHT // 2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return img


def get_llava_description(base64_image):
    """Sends an image to the LLaVA model and returns the description."""
    global last_description
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Describe the scene in this image.",
                "stream": False,
                "images": [base64_image],
            },
        )
        response.raise_for_status()
        data = response.json()
        description = data.get("response", "No description found.")
        last_description = description.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        last_description = "Error: Could not connect to Ollama."


def analyze_frames():
    """Periodically captures a frame and sends it for analysis."""
    global last_frame
    while is_running:
        time.sleep(ANALYSIS_INTERVAL)
        with frame_lock:
            if last_frame is not None:
                _, buffer = cv2.imencode(".jpg", last_frame)
                base64_image = base64.b64encode(buffer).decode("utf-8")
                threading.Thread(
                    target=get_llava_description, args=(base64_image,)
                ).start()


def test_camera_with_timeout():
    """Test camera with a timeout to handle WSL2 issues."""
    global camera_working

    print("Testing camera access with timeout...")

    # Try different approaches with timeout
    approaches = [
        ("Direct index 0", lambda: cv2.VideoCapture(0)),
        ("Direct index 0 with V4L2", lambda: cv2.VideoCapture(0, cv2.CAP_V4L2)),
        ("Device path", lambda: cv2.VideoCapture("/dev/video0")),
    ]

    for name, approach in approaches:
        print(f"Trying: {name}")
        try:
            cap = approach()
            if cap.isOpened():
                print(f"✓ {name} - Camera opened successfully")

                # Set a shorter timeout for frame capture
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Try to read a frame with timeout
                start_time = time.time()
                ret = False
                frame = None

                while time.time() - start_time < CAMERA_TIMEOUT:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(
                            f"✓ {name} - Frame captured successfully, shape: {frame.shape}"
                        )
                        cap.release()
                        camera_working = True
                        return cap, name
                    time.sleep(0.1)

                print(
                    f"✗ {name} - Frame capture timed out after {CAMERA_TIMEOUT} seconds"
                )
            else:
                print(f"✗ {name} - Failed to open camera")
            cap.release()
        except Exception as e:
            print(f"✗ {name} - Exception: {e}")
            if "cap" in locals():
                cap.release()

    print("No working camera found. Using test image mode.")
    camera_working = False
    return None, "test_image"


def main():
    """Main function to capture video, display it, and manage analysis."""
    global is_running, last_frame, camera_working

    # Try to find a working camera
    cap, camera_method = test_camera_with_timeout()

    if not camera_working:
        print("\nRunning in test image mode.")
        print("This means the camera is not accessible in WSL2.")
        print("The script will show a test image and analyze it.")
        print("To use a real camera, you may need to:")
        print("1. Use Windows directly instead of WSL2")
        print("2. Set up camera forwarding in WSL2 (complex)")
        print("3. Use a different camera access method")
        print("\nPress 'q' to quit the test mode.\n")

        # Create test image
        test_img = create_test_image()
        last_frame = test_img.copy()

        analysis_thread = threading.Thread(target=analyze_frames)
        analysis_thread.start()

        while True:
            # Display test image with description
            display_img = test_img.copy()

            # Add a semi-transparent background for the text
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 50), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

            # Add the description text
            cv2.putText(
                display_img,
                last_description,
                (10, 30),
                FONT,
                FONT_SCALE,
                FONT_COLOR,
                LINE_TYPE,
            )

            cv2.imshow("Test Image Mode - Real-time Scene Description", display_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                is_running = False
                break

        cv2.destroyAllWindows()
        analysis_thread.join()
        return

    # Camera is working, proceed with normal operation
    print(f"Using camera method: {camera_method}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize latency

    analysis_thread = threading.Thread(target=analyze_frames)
    analysis_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        with frame_lock:
            last_frame = frame.copy()

        # --- Add text to the frame ---
        # Add a semi-transparent background for the text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 50), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add the description text
        cv2.putText(
            frame,
            last_description,
            (10, 30),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            LINE_TYPE,
        )

        cv2.imshow("Real-time Scene Description", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    analysis_thread.join()


if __name__ == "__main__":
    main()
