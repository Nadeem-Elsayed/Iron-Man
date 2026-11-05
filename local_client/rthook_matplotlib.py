import sys
import types

# MediaPipe draws landmarks with matplotlib if available, but falls back gracefully if stubbed.
# Stub matplotlib and matplotlib.pyplot to avoid packaging heavy plotting libraries.
if "matplotlib" not in sys.modules:
    dummy_mpl = types.ModuleType("matplotlib")
    dummy_mpl.pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"] = dummy_mpl
    sys.modules["matplotlib.pyplot"] = dummy_mpl.pyplot

