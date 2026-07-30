"""The `retro_os` theme: classic OS / retro style guide.

Follows the Retro-OS UI kit tokens: gray #A0A0A0, white #F0F0F0, blue #0078D7
and red #E81123, flat fills (single-color stops - no gloss), 1px solid
borders, monospace bitmap-flavored headings over a modern sans body.
"""
from kai.themes.base import ChartStyle, FontSet, Palette, Theme

_BLUE = "#0078D7"
_RED = "#E81123"
_GRAY = "#A0A0A0"
_WHITE = "#F0F0F0"

RETRO_OS_THEME = Theme(
    name="retro_os",
    # clam renders flat widgets; the native vista theme would reintroduce gloss
    ttk_theme_candidates=("clam",),
    palette=Palette(
        window_bg="#DEDEDE",
        panel_bg=_WHITE,
        panel_border="#202020",
        cell_bg="#FFFFFF",
        text_fg="#000000",
        header_stops=(_BLUE,),
        header_fg="#FFFFFF",
        preview_header_bg="#C8C8C8",
        button_stops=(_BLUE,),
        button_border="#005A9E",
        button_fg="#FFFFFF",
        button_hover_stops=("#2B96E8",),
        button_hover_border="#003C6C",
        button_disabled_stops=("#C8C8C8",),
        button_disabled_border=_GRAY,
        button_disabled_fg="#7A7A7A",
        tab_active_stops=("#FFFFFF",),
        tab_active_fg="#000000",
        tab_inactive_stops=("#C8C8C8",),
        tab_inactive_fg="#5A5A5A",
        tab_border="#202020",
        statusbar_stops=(_BLUE,),
        statusbar_fg="#FFFFFF",
        select_bg=_BLUE,
        select_fg="#FFFFFF",
        label_highlight="#F6C6CD",   # light tint of the red token
        feature_highlight="#BFDDF5",  # light tint of the blue token
        chip_bg="#BFDDF5",
    ),
    fonts=FontSet(
        body=("Segoe UI", 9),
        body_bold=("Segoe UI", 9, "bold"),
        small=("Segoe UI", 8),
        header=("Consolas", 10, "bold"),
        button=("Consolas", 10, "bold"),
        tab=("Consolas", 9),
        stopwatch=("Consolas", 26, "bold"),
        mono=("Consolas", 8),
    ),
    charts=ChartStyle(
        figure_bg=_WHITE,
        panel_bg="#FFFFFF",
        grid_color="#C8C8C8",
        loss_color=_BLUE,
        accent_color=_RED,
        scatter_color=_BLUE,
        font_family="Consolas",
    ),
)
