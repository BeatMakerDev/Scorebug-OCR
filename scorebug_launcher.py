"""
scorebug_launcher.py
====================

This launcher presents a simple start menu allowing the operator to choose
between running the Scorebug Reader directly or first identifying all
available camera devices via the Camera Grid viewer. Once the camera grid
window is closed, the Scorebug Reader will launch automatically. The menu
and subsequent windows use the same clean ttk styling as the main program.

Usage:
    python scorebug_launcher.py

Dependencies:
    The launcher assumes that `scorebug_readerV6.py` and `camera_grid.py`
    are located in the same directory as this script. Both scripts require
    the dependencies listed in their respective docstrings.

Author: ChatGPT – GPT-5 Thinking
License: MIT
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# Import the two application modules.  When packaged into a single EXE
# (e.g. via PyInstaller), these imports will reference the frozen
# modules bundled within the executable.  When running as plain
# scripts, they will import the corresponding .py files from the same
# directory.  We perform the imports at module load time so they are
# available for the callback functions below.
try:
    import camera_grid
except ImportError:
    camera_grid = None
try:
    import scorebug_readerV6
except ImportError:
    scorebug_readerV6 = None


def run_camera_then_main():
    """Run the camera grid viewer and then the Scorebug Reader in-process.

    This function is invoked by the launcher UI.  It first closes the
    launcher window (handled externally), then calls the camera grid's
    ``main()`` function to display the camera feeds.  When the user
    closes that window, it calls the Scorebug Reader's ``main()``
    function to launch the OCR application.  If either module fails to
    import, it silently returns.
    """
    if camera_grid is not None and hasattr(camera_grid, 'main'):
        try:
            camera_grid.main()
        except Exception:
            # ignore errors to avoid crashing the launcher
            pass
    if scorebug_readerV6 is not None and hasattr(scorebug_readerV6, 'main'):
        try:
            # Provide empty argument namespace if the program expects one
            parser = None
            # Run the main program
            scorebug_readerV6.main()
        except Exception:
            pass


def run_main_only():
    """Run only the Scorebug Reader without launching the camera grid."""
    if scorebug_readerV6 is not None and hasattr(scorebug_readerV6, 'main'):
        try:
            scorebug_readerV6.main()
        except Exception:
            pass


def create_launcher():
    """Create and run the launcher GUI."""
    # Determine the base directory of the bundled scripts.  When running
    # from an unpacked directory, __file__ will refer to this file.  In
    # a PyInstaller one-file build, __file__ still points to the temp
    # extraction directory.  We no longer need base_dir for subprocess
    # since we import modules instead.
    root = tk.Tk()
    root.title("Scorebug OCR Launcher")
    root.resizable(False, False)
    # Apply a modern ttk theme if available
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    # Main frame with padding
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill='both', expand=True)
    # Heading label
    ttk.Label(frame, text="Scorebug OCR", font=("Segoe UI", 18, "bold")).pack(pady=(0, 12))
    ttk.Label(frame, text="Choose how to start:", font=("Segoe UI", 12)).pack(pady=(0, 16))
    # Buttons
    btn_width = 28
    ttk.Button(frame, text="Identify Cameras then Launch Scorebug Reader",
               width=btn_width,
               command=lambda: [root.destroy(), run_camera_then_main()]).pack(pady=6)
    ttk.Button(frame, text="Launch Scorebug Reader Now",
               width=btn_width,
               command=lambda: [root.destroy(), run_main_only()]).pack(pady=6)
    # Footer or credit
    ttk.Label(frame, text="© 2025 Scorebug OCR", font=("Segoe UI", 8)).pack(pady=(20, 0))
    root.mainloop()


if __name__ == '__main__':
    create_launcher()