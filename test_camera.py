import time

import cv2


def test_camera():
    print("Testing camera access...")

    # Try different approaches
    approaches = [
        ("Direct index 0", lambda: cv2.VideoCapture(0)),
        ("Direct index 0 with V4L2", lambda: cv2.VideoCapture(0, cv2.CAP_V4L2)),
        ("Direct index 0 with ANY", lambda: cv2.VideoCapture(0, cv2.CAP_ANY)),
        ("Device path", lambda: cv2.VideoCapture("/dev/video0")),
        (
            "Device path with V4L2",
            lambda: cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2),
        ),
    ]

    for name, approach in approaches:
        print(f"\nTrying: {name}")
        try:
            cap = approach()
            if cap.isOpened():
                print(f"✓ {name} - Camera opened successfully")
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(
                        f"✓ {name} - Frame captured successfully, shape: {frame.shape}"
                    )
                    cap.release()
                    return True
                else:
                    print(f"✗ {name} - Camera opened but frame capture failed")
            else:
                print(f"✗ {name} - Failed to open camera")
            cap.release()
        except Exception as e:
            print(f"✗ {name} - Exception: {e}")

    return False


if __name__ == "__main__":
    success = test_camera()
    if success:
        print("\nCamera test successful! The camera is working.")
    else:
        print("\nCamera test failed. This might be a WSL2 camera access issue.")
