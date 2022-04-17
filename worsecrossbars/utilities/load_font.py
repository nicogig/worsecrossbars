"""load_font:
A utility module used to load fonts for plotting
"""
import os
from pathlib import Path

from matplotlib.font_manager import FontProperties


def load_font() -> FontProperties:
    """This function loads the computern modern font used in the plots.

    Returns:
      fpath: Font path object pointing to the computern modern font.
    """

    if os.path.exists(Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf")):
        fpath = FontProperties(
            fname=Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf"), size=18
        )
    else:
        fpath = FontProperties(family="sans-serif", size=18)

    return fpath
