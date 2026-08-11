from pathlib import Path
import streamlit as st

from config.frontend import (
    BACKGROUND,
    BORDER,
    CARD,
    GRADIENT_FROM,
    GRADIENT_TO,
    PRIMARY,
    PRIMARY_DARK,
    ACCENT_PRODUCT,
    ACCENT_POLICY,
    ACCENT_OTHER,
    SUBTEXT,
    TEXT,
)


def load_css():
    css_path = Path(__file__).parent / "styles.css"
    css = css_path.read_text(encoding="utf-8")

    variables = {
        "PRIMARY": PRIMARY,
        "PRIMARY_DARK": PRIMARY_DARK,
        "GRADIENT_FROM": GRADIENT_FROM,
        "GRADIENT_TO": GRADIENT_TO,
        "BACKGROUND": BACKGROUND,
        "CARD": CARD,
        "BORDER": BORDER,
        "TEXT": TEXT,
        "SUBTEXT": SUBTEXT,
        "ACCENT_PRODUCT": ACCENT_PRODUCT,
        "ACCENT_POLICY": ACCENT_POLICY,
        "ACCENT_OTHER": ACCENT_OTHER,
    }

    for name, value in variables.items():
        css = css.replace(f"{{{name}}}", value)

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )