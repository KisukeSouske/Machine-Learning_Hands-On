"""Application entry point.

Pick the UI theme here - any name registered in kai.themes.THEMES:
    "default"   the Windows 7 / Aero look
    "retro_os"  the classic OS / retro style guide look

You can also override it from the command line without editing this file:
    python main.py retro_os
"""
import sys

from kai.gui import TrainingApp

THEME = "retro_os"

if __name__ == "__main__":
    theme = sys.argv[1] if len(sys.argv) > 1 else THEME
    TrainingApp(theme=theme).mainloop()
