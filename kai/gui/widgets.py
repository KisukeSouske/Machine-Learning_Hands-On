"""Theme-aware custom widgets.

Nothing here hardcodes a color, font or border: every widget receives the
active `Theme` at construction and reads what it needs from it. Gradient
pieces are drawn on plain Canvases (ttk has no gradient support); a stops
tuple whose colors are all identical renders as a flat fill, which is how the
retro theme opts out of gloss without special-casing any widget.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable

from kai.themes.base import FontSpec, Stops, Theme


def _blend(canvas: tk.Canvas, color_a: str, color_b: str, ratio: float) -> str:
    r1, g1, b1 = canvas.winfo_rgb(color_a)
    r2, g2, b2 = canvas.winfo_rgb(color_b)
    r = int(r1 + (r2 - r1) * ratio) >> 8
    g = int(g1 + (g2 - g1) * ratio) >> 8
    b = int(b1 + (b2 - b1) * ratio) >> 8
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_stops(canvas: tk.Canvas, width: int, height: int, stops: Stops, tag: str = "bg") -> None:
    """Paint a vertical gradient (or a flat fill when all stops match)."""
    canvas.delete(tag)
    if width <= 0 or height <= 0 or not stops:
        return
    if len(set(stops)) == 1:
        canvas.create_rectangle(0, 0, width, height, fill=stops[0], outline=stops[0], tags=tag)
    else:
        segments = len(stops) - 1
        for y in range(height):
            position = (y / max(height - 1, 1)) * segments
            index = min(int(position), segments - 1)
            color = _blend(canvas, stops[index], stops[index + 1], position - index)
            canvas.create_line(0, y, width, y, fill=color, tags=tag)
    canvas.tag_lower(tag)


def _tab_font(base: FontSpec, bold: bool) -> FontSpec:
    family, size = base[0], base[1]
    return (family, size, "bold") if bold else (family, size)


class PanelHeader(tk.Canvas):
    """The title strip of a panel, with a hairline bottom border."""

    def __init__(self, parent, text: str, theme: Theme, height: int = 26):
        super().__init__(parent, height=height, highlightthickness=0, bd=0,
                         bg=theme.palette.header_stops[0])
        self._text = text
        self._theme = theme
        self.bind("<Configure>", lambda _e: self._redraw())

    def _redraw(self) -> None:
        palette, fonts = self._theme.palette, self._theme.fonts
        width, height = self.winfo_width(), self.winfo_height()
        draw_stops(self, width, height, palette.header_stops)
        self.delete("content")
        self.create_line(0, height - 1, width, height - 1, fill=palette.panel_border, tags="content")
        self.create_text(9, height // 2, text=self._text, anchor="w",
                         font=fonts.header, fill=palette.header_fg, tags="content")


class ThemedButton(tk.Canvas):
    """A themed push button with hover and disabled states. Mirrors the bits
    of the ttk.Button API the app uses (configure(state=...)/cget("state"))."""

    def __init__(self, parent, text: str, theme: Theme, command: Callable | None = None,
                 height: int = 30):
        super().__init__(parent, height=height, highlightthickness=0, bd=0,
                         bg=theme.palette.panel_bg)
        self._text = text
        self._theme = theme
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
            super().configure(cursor="hand2" if self._state == "normal" else "arrow")
            self._redraw()
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            self._redraw()
        if kwargs:
            super().configure(**kwargs)

    config = configure

    def cget(self, key):  # type: ignore[override]
        if key == "state":
            return self._state
        if key == "text":
            return self._text
        return super().cget(key)

    def _redraw(self) -> None:
        palette, fonts = self._theme.palette, self._theme.fonts
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        if self._state != "normal":
            stops, border, fg = (palette.button_disabled_stops,
                                 palette.button_disabled_border, palette.button_disabled_fg)
        elif self._hovered:
            stops, border, fg = (palette.button_hover_stops,
                                 palette.button_hover_border, palette.button_fg)
        else:
            stops, border, fg = palette.button_stops, palette.button_border, palette.button_fg

        draw_stops(self, width, height, stops)
        self.delete("content")
        self.create_rectangle(0, 0, width - 1, height - 1, outline=border, tags="content")
        self.create_text(width // 2, height // 2, text=self._text, anchor="center",
                         font=fonts.button, fill=fg, tags="content")


class ThemedTab(tk.Canvas):
    """One chart tab; the bar toggles any number of them independently."""

    def __init__(self, parent, text: str, theme: Theme, on_click: Callable[[], None],
                 width: int = 150, height: int = 26):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bd=0,
                         bg=theme.palette.tab_inactive_stops[0], cursor="hand2")
        self._text = text
        self._theme = theme
        self._active = False
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Button-1>", lambda _e: on_click())

    def set_active(self, active: bool) -> None:
        self._active = active
        palette = self._theme.palette
        # keep `bg` in sync with the state so cget("bg") reports it
        super().configure(bg=(palette.tab_active_stops[0] if active
                              else palette.tab_inactive_stops[0]))
        self._redraw()

    def _redraw(self) -> None:
        palette, fonts = self._theme.palette, self._theme.fonts
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        if self._active:
            stops, fg = palette.tab_active_stops, palette.tab_active_fg
        else:
            stops, fg = palette.tab_inactive_stops, palette.tab_inactive_fg
        draw_stops(self, width, height, stops)
        self.delete("content")
        self.create_rectangle(0, 0, width - 1, height - 1, outline=palette.tab_border, tags="content")
        self.create_text(width // 2, height // 2, text=self._text, anchor="center",
                         font=_tab_font(fonts.tab, self._active), fill=fg, tags="content")


class StatusBar(tk.Canvas):
    """The thin strip along the bottom of the window."""

    def __init__(self, parent, theme: Theme, height: int = 22):
        super().__init__(parent, height=height, highlightthickness=0, bd=0,
                         bg=theme.palette.statusbar_stops[0])
        self._text = ""
        self._theme = theme
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_text(self, text: str) -> None:
        self._text = text
        self._redraw()

    def _redraw(self) -> None:
        palette, fonts = self._theme.palette, self._theme.fonts
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1:
            return
        draw_stops(self, width, height, palette.statusbar_stops)
        self.delete("content")
        self.create_line(0, 0, width, 0, fill=palette.panel_border, tags="content")
        self.create_text(8, height // 2, text=self._text, anchor="w",
                         font=fonts.small, fill=palette.statusbar_fg, tags="content")


class MultiSelectDropdown(ttk.Frame):
    """A dropdown holding a multi-select Listbox: rows are toggled by clicking
    them directly, and the popup stays open until an outside click, Escape, or
    Close - unlike ttk.Combobox, which collapses on every pick."""

    def __init__(self, parent, theme: Theme, placeholder: str = "Select...",
                 on_change: Callable[[], None] | None = None):
        super().__init__(parent)
        self._theme = theme
        self._option_names: list[str] = []
        self._selected: list[str] = []
        self._on_change = on_change
        self._placeholder = placeholder
        self._popup: tk.Toplevel | None = None
        self._listbox: tk.Listbox | None = None
        self._enabled = False

        self._button = ThemedButton(self, f"{placeholder}  ▾", theme, command=self.toggle, height=26)
        self._button.configure(state="disabled")
        self._button.pack(fill="x")

    def configure_state(self, enabled: bool) -> None:
        self._enabled = enabled
        self._button.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self.close()

    def set_options(self, options: list[str]) -> None:
        self._option_names = list(options)
        self._selected = [c for c in self._selected if c in self._option_names]
        self._refresh_button_text()
        if self.is_open():
            self.close()
            self.open()

    def get_selected(self) -> list[str]:
        return list(self._selected)

    def set_selected(self, values: list[str]) -> None:
        self._selected = [v for v in values if v in self._option_names]
        self._refresh_button_text()
        self._sync_listbox_selection()

    def is_open(self) -> bool:
        return self._popup is not None and self._popup.winfo_exists()

    def toggle(self) -> None:
        self.close() if self.is_open() else self.open()

    def open(self) -> None:
        if not self._enabled or self.is_open() or not self._option_names:
            return
        palette = self._theme.palette
        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.configure(bg=palette.panel_border)
        x = self._button.winfo_rootx()
        y = self._button.winfo_rooty() + self._button.winfo_height()
        popup.geometry(f"{max(self._button.winfo_width(), 200)}x200+{x}+{y}")

        container = tk.Frame(popup, bg=palette.cell_bg)
        container.pack(fill="both", expand=True, padx=1, pady=1)

        listbox = tk.Listbox(
            container, selectmode="multiple", exportselection=False,
            activestyle="none", highlightthickness=0, borderwidth=0,
            bg=palette.cell_bg, fg=palette.text_fg,
            selectbackground=palette.feature_highlight, selectforeground=palette.text_fg,
            font=self._theme.fonts.body,
        )
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for option in self._option_names:
            listbox.insert(tk.END, option)
        listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox = listbox
        self._sync_listbox_selection()

        tk.Button(container, text="Close", relief="flat", bg=palette.preview_header_bg,
                  fg=palette.text_fg, font=self._theme.fonts.small,
                  command=self.close).pack(side="bottom", fill="x")

        # grab_set routes every click to the popup; clicks landing on the
        # listbox are handled by it, anything else reaches this binding
        # and means "the user clicked outside", so we close.
        popup.bind("<Button-1>", self._maybe_close_on_outside_click)
        popup.bind("<Escape>", lambda _e: self.close())
        popup.grab_set()
        self._popup = popup

    def _sync_listbox_selection(self) -> None:
        if self._listbox is None or not self._listbox.winfo_exists():
            return
        self._listbox.selection_clear(0, tk.END)
        for index, option in enumerate(self._option_names):
            if option in self._selected:
                self._listbox.selection_set(index)

    def _on_listbox_select(self, _event=None) -> None:
        if self._listbox is None:
            return
        self._selected = [self._option_names[i] for i in self._listbox.curselection()]
        self._refresh_button_text()
        if self._on_change is not None:
            self._on_change()

    def _maybe_close_on_outside_click(self, event) -> None:
        popup = self._popup
        if popup is None:
            return
        inside = (
            popup.winfo_rootx() <= event.x_root <= popup.winfo_rootx() + popup.winfo_width()
            and popup.winfo_rooty() <= event.y_root <= popup.winfo_rooty() + popup.winfo_height()
        )
        if not inside:
            self.close()

    def _refresh_button_text(self) -> None:
        count = len(self._selected)
        text = self._placeholder if count == 0 else f"{count} column{'s' if count > 1 else ''} selected"
        self._button.configure(text=f"{text}  ▾")

    def close(self) -> None:
        if self.is_open():
            self._popup.grab_release()
            self._popup.destroy()
        self._popup = None
        self._listbox = None


class ChartTabBar(tk.Frame):
    """Tab-looking toggles: any number can be active at once (unlike a real
    Notebook, which is exclusive)."""

    def __init__(self, parent, theme: Theme, keys_and_labels, active_keys, on_change):
        super().__init__(parent, bg=theme.palette.panel_bg)
        self._on_change = on_change
        self._active = set(active_keys)
        self._tabs: dict[str, ThemedTab] = {}
        for key, label in keys_and_labels:
            tab = ThemedTab(self, label, theme, on_click=lambda k=key: self._toggle(k))
            tab.pack(side="left", padx=(0, 2))
            self._tabs[key] = tab
        self._repaint()

    def _toggle(self, key: str) -> None:
        if key in self._active:
            self._active.remove(key)
        else:
            self._active.add(key)
        self._repaint()
        self._on_change()

    def _repaint(self) -> None:
        for key, tab in self._tabs.items():
            tab.set_active(key in self._active)

    def get_active(self) -> list[str]:
        return [key for key in self._tabs if key in self._active]
