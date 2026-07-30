"""Windows 7 (Aero) flavoured widgets.

ttk has no gradient support, so the glossy pieces - panel headers, the primary
button and the chart tabs - are drawn on plain Canvases. Everything else relies
on the native `vista` ttk theme, which already renders Aero-style comboboxes,
entries, scrollbars and treeviews.
"""
import tkinter as tk

WINDOW_BG = "#D6E5F5"
PANEL_BG = "#FBFDFE"
PANEL_BORDER = "#9AB8D6"
HEADER_TOP = "#F3F9FE"
HEADER_BOTTOM = "#D3E5F7"
HEADER_FG = "#15428B"

BUTTON_TOP = "#FCFDFE"
BUTTON_UPPER = "#EAF4FC"
BUTTON_LOWER = "#D3EAFA"
BUTTON_BOTTOM = "#BEDFF6"
BUTTON_BORDER = "#7FA3C8"
BUTTON_HOVER_TOP = "#F5FBFE"
BUTTON_HOVER_UPPER = "#E3F4FD"
BUTTON_HOVER_LOWER = "#C4E7FC"
BUTTON_HOVER_BOTTOM = "#A8DCFA"
BUTTON_HOVER_BORDER = "#3C7FB1"
BUTTON_DISABLED_TOP = "#F7F7F7"
BUTTON_DISABLED_BOTTOM = "#E3E3E3"
BUTTON_DISABLED_BORDER = "#BCBCBC"
BUTTON_FG = "#0B335E"
BUTTON_DISABLED_FG = "#9A9A9A"

TAB_ACTIVE_TOP = "#5A9BD8"
TAB_ACTIVE_BOTTOM = "#2A6FB5"
TAB_ACTIVE_FG = "#FFFFFF"
TAB_INACTIVE_TOP = "#F2F8FD"
TAB_INACTIVE_BOTTOM = "#D6E6F6"
TAB_INACTIVE_FG = "#15428B"
TAB_BORDER = "#8FB2D4"

STATUSBAR_TOP = "#F2F7FC"
STATUSBAR_BOTTOM = "#DCE9F7"
SELECT_BG = "#CBE4F8"


def _blend(canvas: tk.Canvas, color_a: str, color_b: str, ratio: float) -> str:
    r1, g1, b1 = canvas.winfo_rgb(color_a)
    r2, g2, b2 = canvas.winfo_rgb(color_b)
    r = int(r1 + (r2 - r1) * ratio) >> 8
    g = int(g1 + (g2 - g1) * ratio) >> 8
    b = int(b1 + (b2 - b1) * ratio) >> 8
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_gradient(canvas: tk.Canvas, width: int, height: int, stops, tag: str = "aero_bg") -> None:
    """Paint a vertical gradient. `stops` is a list of colours distributed
    evenly from top to bottom; two colours give a plain fade, four give the
    classic Aero gloss (bright top half, deeper bottom half)."""
    canvas.delete(tag)
    if width <= 0 or height <= 0 or len(stops) < 2:
        return
    segments = len(stops) - 1
    for y in range(height):
        position = (y / max(height - 1, 1)) * segments
        index = min(int(position), segments - 1)
        color = _blend(canvas, stops[index], stops[index + 1], position - index)
        canvas.create_line(0, y, width, y, fill=color, tags=tag)
    canvas.tag_lower(tag)


class GradientHeader(tk.Canvas):
    """The title strip of a panel: Aero gloss plus a hairline bottom border."""

    def __init__(self, parent, text: str, height: int = 26):
        super().__init__(parent, height=height, highlightthickness=0, bd=0, bg=HEADER_TOP)
        self._text = text
        self.bind("<Configure>", lambda _e: self._redraw())

    def _redraw(self) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        draw_gradient(self, width, height, [HEADER_TOP, HEADER_BOTTOM])
        self.delete("content")
        self.create_line(0, height - 1, width, height - 1, fill=PANEL_BORDER, tags="content")
        self.create_text(
            9, height // 2, text=self._text, anchor="w",
            font=("Segoe UI", 9, "bold"), fill=HEADER_FG, tags="content",
        )


class AeroButton(tk.Canvas):
    """A glossy Aero push button. Mirrors the bits of the ttk.Button API the
    app uses (`configure(state=...)` / `cget("state")`) so it can stand in for one."""

    def __init__(self, parent, text: str, command=None, height: int = 30):
        super().__init__(parent, height=height, highlightthickness=0, bd=0, bg=PANEL_BG)
        self._text = text
        self._command = command
        self._state = "normal"
        self._hovered = False
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, _event=None) -> None:
        self._hovered = True
        self._redraw()

    def _on_leave(self, _event=None) -> None:
        self._hovered = False
        self._redraw()

    def _on_click(self, _event=None) -> None:
        if self._state == "normal" and self._command is not None:
            self._command()

    def configure(self, **kwargs):  # type: ignore[override]
        if "state" in kwargs:
            self._state = kwargs.pop("state")
            self.configure_cursor()
            self._redraw()
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            self._redraw()
        if kwargs:
            super().configure(**kwargs)

    config = configure

    def configure_cursor(self) -> None:
        super().configure(cursor="hand2" if self._state == "normal" else "arrow")

    def cget(self, key):  # type: ignore[override]
        if key == "state":
            return self._state
        if key == "text":
            return self._text
        return super().cget(key)

    def _redraw(self) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        if self._state != "normal":
            stops = [BUTTON_DISABLED_TOP, BUTTON_DISABLED_BOTTOM]
            border, fg = BUTTON_DISABLED_BORDER, BUTTON_DISABLED_FG
        elif self._hovered:
            stops = [BUTTON_HOVER_TOP, BUTTON_HOVER_UPPER, BUTTON_HOVER_LOWER, BUTTON_HOVER_BOTTOM]
            border, fg = BUTTON_HOVER_BORDER, BUTTON_FG
        else:
            stops = [BUTTON_TOP, BUTTON_UPPER, BUTTON_LOWER, BUTTON_BOTTOM]
            border, fg = BUTTON_BORDER, BUTTON_FG

        draw_gradient(self, width, height, stops)
        self.delete("content")
        self.create_rectangle(0, 0, width - 1, height - 1, outline=border, tags="content")
        self.create_text(
            width // 2, height // 2, text=self._text, anchor="center",
            font=("Segoe UI", 9, "bold"), fill=fg, tags="content",
        )


class AeroTab(tk.Canvas):
    """A chart tab: glossy blue when active, pale when not."""

    def __init__(self, parent, text: str, on_click, width: int = 150, height: int = 26):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bd=0,
                         bg=TAB_INACTIVE_TOP, cursor="hand2")
        self._text = text
        self._active = False
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Button-1>", lambda _e: on_click())

    def set_active(self, active: bool) -> None:
        self._active = active
        # keep `bg` in sync with the state so cget("bg") reports it
        super().configure(bg=TAB_ACTIVE_TOP if active else TAB_INACTIVE_TOP)
        self._redraw()

    def _redraw(self) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        if self._active:
            stops, fg = [TAB_ACTIVE_TOP, TAB_ACTIVE_BOTTOM], TAB_ACTIVE_FG
        else:
            stops, fg = [TAB_INACTIVE_TOP, TAB_INACTIVE_BOTTOM], TAB_INACTIVE_FG
        draw_gradient(self, width, height, stops)
        self.delete("content")
        self.create_rectangle(0, 0, width - 1, height - 1, outline=TAB_BORDER, tags="content")
        self.create_text(
            width // 2, height // 2, text=self._text, anchor="center",
            font=("Segoe UI", 9, "bold" if self._active else "normal"), fill=fg, tags="content",
        )


class StatusBar(tk.Canvas):
    """The thin gradient strip along the bottom of the window."""

    def __init__(self, parent, height: int = 22):
        super().__init__(parent, height=height, highlightthickness=0, bd=0, bg=STATUSBAR_TOP)
        self._text = ""
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_text(self, text: str) -> None:
        self._text = text
        self._redraw()

    def _redraw(self) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        draw_gradient(self, width, height, [STATUSBAR_TOP, STATUSBAR_BOTTOM])
        self.delete("content")
        self.create_line(0, 0, width, 0, fill=PANEL_BORDER, tags="content")
        self.create_text(8, height // 2, text=self._text, anchor="w",
                         font=("Segoe UI", 8), fill="#3C5A78", tags="content")
