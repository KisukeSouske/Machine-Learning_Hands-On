import dataclasses

import pytest

from kai.themes import DEFAULT_THEME, RETRO_OS_THEME, THEMES, Theme, get_theme


def test_get_theme_by_name_returns_theme_instances():
    assert get_theme("default") is DEFAULT_THEME
    assert get_theme("retro_os") is RETRO_OS_THEME


def test_get_theme_passes_theme_instances_through():
    assert get_theme(DEFAULT_THEME) is DEFAULT_THEME


def test_get_theme_unknown_name_raises_with_available_names():
    with pytest.raises(ValueError) as excinfo:
        get_theme("does_not_exist")
    message = str(excinfo.value)
    assert "does_not_exist" in message
    assert "default" in message
    assert "retro_os" in message


def test_registry_names_match_theme_names():
    for name, theme in THEMES.items():
        assert isinstance(theme, Theme)
        assert theme.name == name


def test_every_theme_fills_every_palette_field():
    for theme in THEMES.values():
        for field in dataclasses.fields(theme.palette):
            value = getattr(theme.palette, field.name)
            assert value, f"{theme.name}.palette.{field.name} is empty"


def test_retro_theme_uses_styleguide_tokens():
    palette = RETRO_OS_THEME.palette
    assert palette.header_stops == ("#0078D7",)  # flat blue, no gradient
    assert RETRO_OS_THEME.charts.accent_color == "#E81123"


def test_default_theme_keeps_aero_gradients():
    palette = DEFAULT_THEME.palette
    assert len(set(palette.button_stops)) > 1  # glossy multi-stop gradient
    assert palette.header_stops == ("#F3F9FE", "#D3E5F7")
