"""The `default` theme: the Windows 7 / Aero look the app shipped with.

Glossy multi-stop gradients, light blue chrome, Segoe UI text and the
Roboto-based chart styling used since the first dashboard.
"""
from kai.themes.base import ChartStyle, FontSet, Palette, Theme

DEFAULT_THEME = Theme(
    name="default",
    ttk_theme_candidates=("vista", "xpnative", "winnative", "clam"),
    palette=Palette(
        window_bg="#D6E5F5",
        panel_bg="#FBFDFE",
        panel_border="#9AB8D6",
        cell_bg="#FBFDFE",
        text_fg="#15428B",
        header_stops=("#F3F9FE", "#D3E5F7"),
        header_fg="#15428B",
        preview_header_bg="#D3E5F7",
        button_stops=("#FCFDFE", "#EAF4FC", "#D3EAFA", "#BEDFF6"),
        button_border="#7FA3C8",
        button_fg="#0B335E",
        button_hover_stops=("#F5FBFE", "#E3F4FD", "#C4E7FC", "#A8DCFA"),
        button_hover_border="#3C7FB1",
        button_disabled_stops=("#F7F7F7", "#E3E3E3"),
        button_disabled_border="#BCBCBC",
        button_disabled_fg="#9A9A9A",
        tab_active_stops=("#5A9BD8", "#2A6FB5"),
        tab_active_fg="#FFFFFF",
        tab_inactive_stops=("#F2F8FD", "#D6E6F6"),
        tab_inactive_fg="#15428B",
        tab_border="#8FB2D4",
        statusbar_stops=("#F2F7FC", "#DCE9F7"),
        statusbar_fg="#3C5A78",
        select_bg="#CBE4F8",
        select_fg="#000000",
        label_highlight="#FFE0B2",
        feature_highlight="#E1D5F7",
        chip_bg="#E1D5F7",
    ),
    fonts=FontSet(
        body=("Segoe UI", 9),
        body_bold=("Segoe UI", 9, "bold"),
        small=("Segoe UI", 8),
        header=("Segoe UI", 9, "bold"),
        button=("Segoe UI", 9, "bold"),
        tab=("Segoe UI", 9),
        stopwatch=("Consolas", 26, "bold"),
        mono=("Consolas", 8),
    ),
    charts=ChartStyle(
        figure_bg="#FFFFFF",
        panel_bg="#E5ECF6",
        grid_color="#FFFFFF",
        loss_color="#0057E7",
        accent_color="#D62D20",
        scatter_color="tab:blue",
        font_family="Roboto",
    ),
)
