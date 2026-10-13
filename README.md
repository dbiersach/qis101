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

### Git setup for contributors

If you will commit changes, run these two commands once from the project
folder after cloning:

```sh
git config core.hooksPath githooks
git config filter.nbclean.clean "uv run --quiet --no-project python nb_clean.py"
```

The first registers the pre-commit hook, which runs `nb_check.py` and refuses
a commit containing a notebook without `%matplotlib inline` or a script that
reads or writes data files outside its own folder. The second registers the
filter that strips the interpreter version and execution timestamps from
notebooks as they are staged, so notebooks nobody edited do not show up as
changed.

Git never runs hooks or filters that arrive with a clone until they are
registered locally, so skipping this step causes no errors. The checks simply
do not run. See [nb_check vs nb_clean.md](<nb_check vs nb_clean.md>) for how
both tools work.

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
