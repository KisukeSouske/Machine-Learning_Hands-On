"""Theme registry.

Adding a theme = writing a module that builds a `Theme` and listing it here.
The GUI resolves themes exclusively through `get_theme`, so `main.py` can pick
one by name without importing any theme module directly.
"""
from kai.themes.base import ChartStyle, FontSet, Palette, Stops, Theme
from kai.themes.default import DEFAULT_THEME
from kai.themes.retro_os import RETRO_OS_THEME

THEMES: dict[str, Theme] = {
    DEFAULT_THEME.name: DEFAULT_THEME,
    RETRO_OS_THEME.name: RETRO_OS_THEME,
}

__all__ = [
    "ChartStyle",
    "FontSet",
    "Palette",
    "Stops",
    "Theme",
    "THEMES",
    "DEFAULT_THEME",
    "RETRO_OS_THEME",
    "get_theme",
]


def get_theme(theme: str | Theme) -> Theme:
    """Resolve a theme by name, or pass a ready-made `Theme` through."""
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        available = ", ".join(sorted(THEMES))
        raise ValueError(f"Unknown theme {theme!r}. Available themes: {available}") from None
