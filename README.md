# Foundations of Quantum Information Science (QIS101)

## Using VSCode, JupyterLab, and IBM Qiskit

[QIS101 Slides](https://brookhavenlab.sharepoint.com/:f:/s/QIS101/IgBODKjMlnDCSLFAF7FVrMxLASDbYGwVHOfwfGlg_8bEIX4?e=HaMC2H)

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the project folder run:

```sh
uv sync
```

This creates a `.venv` with everything needed. The courseware runs on Windows,
macOS (Apple Silicon and Intel), and Linux.

### Linux notes

Standalone `.py` scripts open an interactive matplotlib window via Qt (PySide6).
On Linux this requires:

- **Ubuntu 22.04+ / Linux Mint 21+** (or another distro with glibc ≥ 2.34) so
  `uv sync` can install PySide6.
- The Qt `xcb` platform plugin's system library. If a script fails with
  *"Could not load the Qt platform plugin 'xcb'"*, install it once with:

  ```sh
  sudo apt install libxcb-cursor0
  ```

The Jupyter notebooks render plots inline and need none of the above — they run
on any platform with no extra system packages.
