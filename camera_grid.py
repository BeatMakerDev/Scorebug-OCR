"""
camera_grid.py
================

This script enumerates camera devices (webcams) on your computer and opens
each one concurrently, displaying their live video feeds in a single grid
window. Each feed is labelled with its index number so you can easily
identify which physical camera corresponds to which index. Use this to
discover internal and external video inputs on your machine.

Requirements:
    python -m pip install opencv-python

Usage:
    python camera_grid.py

Press 'q' in the window to exit and release all cameras.

Note:
    The script attempts to open camera indices from 0 up to MAX_CAMS-1. If
    you have more than MAX_CAMS cameras, increase the MAX_CAMS constant. On
    some systems, trying to open indices that don't exist may cause a slight
    delay; this is normal. Only successfully opened cameras will be used.

Author: ChatGPT – GPT-5 Thinking
License: MIT
"""

import cv2
import math
import numpy as np


def enumerate_cameras(max_cams: int = 10):
    """Try to open camera indices from 0 to max_cams-1.

    Returns a list of (index, VideoCapture) tuples for cameras that were
    successfully opened. Cameras that fail to open are ignored.

    Args:
        max_cams: The number of indices to probe. Increase if you expect
            more cameras.
    """
    cams = []
    for idx in range(max_cams):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            cams.append((idx, cap))
        else:
            # Clean up failed capture to avoid resource leaks
            if cap is not None:
                cap.release()
    return cams


def draw_label(frame, text: str):
    """Overlay a semi-transparent label on the top-left corner of the frame.

    Args:
        frame: The image on which to draw (BGR numpy array).
        text: The string to draw.
    """
    # Position and styling parameters
    x, y = 5, 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color_bg = (0, 0, 0)  # black background for text
    color_fg = (0, 255, 0)  # green text
    thickness = 2

    # Draw background rectangle for better contrast
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x - 2, y - text_h - 2), (x + text_w + 2, y + 4), color_bg, -1)
    # Draw the text itself
    cv2.putText(frame, text, (x, y), font, scale, color_fg, thickness, cv2.LINE_AA)


def main():
    # Adjust this constant if you expect more than 10 cameras.
    MAX_CAMS = 10
    cams = enumerate_cameras(MAX_CAMS)
    if not cams:
        print("No cameras found.")
        return

    n = len(cams)
    # Compute grid size: columns and rows (roughly square)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Desired size for each camera preview (adjust to taste)
    preview_width = 320
    preview_height = 240

    window_name = "Camera Grid (press 'q' to quit)"

    while True:
        frames = []
        # Read a frame from each camera. If reading fails, use a black placeholder.
        for idx, cap in cams:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
            # Resize to the desired preview size
            resized = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            # Draw the index label
            draw_label(resized, f"Cam {idx}")
            frames.append(resized)

        # Compose the grid: create a blank canvas and place each preview
        grid_h = rows * preview_height
        grid_w = cols * preview_width
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for i, preview in enumerate(frames):
            r = i // cols
            c = i % cols
            y0 = r * preview_height
            x0 = c * preview_width
            grid[y0:y0 + preview_height, x0:x0 + preview_width] = preview

        cv2.imshow(window_name, grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release all cameras and close the window
    for _, cap in cams:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()