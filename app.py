"""Standalone launcher. The UI itself is built in spx/app.py and is also
displayed and launched inline from the notebook, so both paths stay identical."""
from spx.app import launch

if __name__ == "__main__":
    launch()
