"""Theme contracts.

A theme is a frozen bundle of visual decisions - colors, gradients, fonts and
chart styling. Widgets and panels never hardcode any of these values: they
receive a `Theme` and read what they need from it, which is what makes themes
swappable from `main.py` without touching component code.

Gradients are expressed as `Stops`: a tuple of colors distributed evenly from
top to bottom. Two or more distinct colors produce a fade; a tuple whose
colors are all identical is rendered as a flat fill (how retro themes opt out
of gloss without a special case in every widget).
"""
from dataclasses import dataclass

Stops = tuple[str, ...]
FontSpec = tuple  # ("Segoe UI", 9) or ("Segoe UI", 9, "bold")


@dataclass(frozen=True)
class Palette:
    """Every color a widget is allowed to use, grouped by role."""

    window_bg: str
    panel_bg: str
    panel_border: str
    cell_bg: str
    text_fg: str

    header_stops: Stops
    header_fg: str
    preview_header_bg: str

    button_stops: Stops
    button_border: str
    button_fg: str
    button_hover_stops: Stops
    button_hover_border: str
    button_disabled_stops: Stops
    button_disabled_border: str
    button_disabled_fg: str

    tab_active_stops: Stops
    tab_active_fg: str
    tab_inactive_stops: Stops
    tab_inactive_fg: str
    tab_border: str

    statusbar_stops: Stops
    statusbar_fg: str

    select_bg: str
    select_fg: str
    label_highlight: str
    feature_highlight: str
    chip_bg: str


@dataclass(frozen=True)
class FontSet:
    """Named font roles; widgets pick by role, never by literal family name."""

    body: FontSpec
    body_bold: FontSpec
    small: FontSpec
    header: FontSpec
    button: FontSpec
    tab: FontSpec
    stopwatch: FontSpec
    mono: FontSpec


@dataclass(frozen=True)
class ChartStyle:
    """Styling consumed by kai.visualization when drawing matplotlib charts."""

    figure_bg: str
    panel_bg: str
    grid_color: str
    loss_color: str
    accent_color: str
    scatter_color: str
    font_family: str


@dataclass(frozen=True)
class Theme:
    name: str
    # ttk built-in themes to try, in order; first available wins
    ttk_theme_candidates: tuple[str, ...]
    palette: Palette
    fonts: FontSet
    charts: ChartStyle
