# QIS101 Python and Jupyter Notebook Style Guide

These instructions define the expected coding and documentation style for all Python scripts (`.py`) and Jupyter notebooks (`.ipynb`) in this repository.

The goal is clarity, consistency, and strong pedagogical value.

---

## General Principles

- Code should be **clear, explicit, and readable**.
- Prefer **teaching-oriented explanations** over compact or clever code.
- Write as if the reader is a **student learning the concept for the first time**.
- Avoid unnecessary abstraction unless it improves understanding.

---

## File Naming

- Use lowercase `snake_case` for all files.
- File names should be **descriptive and topic-based**.

Examples:

- `basel_series.ipynb`
- `quantum_circuit_intro.ipynb`
- `qis101_utils.py`

---

## Jupyter Notebook Structure

### First Code Cell

The first code cell must begin with a short docstring containing the notebook filename:

```python
"""example_notebook.ipynb"""
```

If the notebook imports matplotlib anywhere, an inline backend must be
selected after the docstring and the cell-label comment, before any import:

```python
"""example_notebook.ipynb"""

# Cell 01 - Import packages

%matplotlib inline

import matplotlib.pyplot as plt
```

This is required, not optional, and it must be in the **first** code cell so
that it runs before any plot is made. Without it the kernel inherits the
venv's `qtagg` backend, `plt.show()` opens a window that blocks the kernel,
and the cell spins forever. A notebook that needs a live animation uses
`%matplotlib widget` instead. `nb_check.py` enforces this, and
`githooks/pre-commit` refuses a commit that violates it.

---

### Cell Labeling

Each code cell should be labeled with a structured comment:

```python
# Cell 01 - Import packages
# Cell 02 - Define helper functions
# Cell 03 - Run simulation
```

Guidelines:

- Use two-digit numbering (`01`, `02`, etc.)
- Keep descriptions short and meaningful

---

### Markdown Cell Structure

Every markdown cell after the first one in a notebook must begin with a
horizontal rule on its own line, followed by a `###` header:

```markdown
---
### Setup: simulation parameters

The simulation runs from ...
```

The rule draws a visible line between the cell and the output of the code cell
above it, which otherwise run together in the notebook view.

Header rules:

- Start at `###`. Never use `#` or `##`, which render so large that a short
  heading eats a disproportionate amount of vertical space.
- `###` is the only level used for section headers. Sub-points inside a section
  are made with bold lead-ins or lists, not `####`.
- The very first cell of the notebook is the only exception to the rule: it
  opens directly with its `###` title, since there is no output above it to
  separate.
- A markdown cell that follows another markdown cell still takes the rule. The
  separator doubles as a section break, so it stays even where there is no code
  output above it.

Do not open a markdown cell with a bolded run-in sentence such as
`**Simulation parameters.**`. Write a real `###` header instead.

---

### Every Code Cell Must Display Output

Never write a code cell that produces no visible output. A cell containing
only imports, constants, or function definitions gives the student no feedback
that they ran it. It is easy to skip a silent cell and then hit a `NameError`
in the next one.

When a cell exists mainly to define things, end it with a short check that
exercises what was just defined. Call the new functions on a simple case and
`print()` or `display()` the result next to the expected answer:

```python
# Quick check that circuit works as expected
out = circuit(t, t, t, t)
print(f"circuit(1, 1, 1, 1) = {out[1, 0]}  (expected 0)")
```

This doubles as a worked example and as proof the cell ran.

Stale saved output is the related hazard. A cell whose code was edited but
never rerun still shows its old result, which reads as if it passed. Rerun
the notebook after editing it.

---

### Markdown + Code Balance

- Use markdown cells to explain:
  - What the code does
  - Why the method is used
  - What the results mean
- Keep explanations **plain, direct, and instructional**
- Avoid overly formal or verbose writing

---

## Python Code Style

### Type Hints

- Use type hints for all reusable functions and classes
- Prefer modern Python 3.13 syntax:

```python
float | np.ndarray
list[str]
tuple[np.ndarray, ...]
```

---

### Docstrings

- Use **NumPy-style docstrings** for reusable functions in `.py` files

Example:

```python
def compute_energy(x: np.ndarray) -> float:
    """
    Compute the total energy of the system.

    Parameters
    ----------
    x : np.ndarray
        Input state vector.

    Returns
    -------
    float
        Computed energy value.
    """
```

- Short helper functions may use one-line docstrings:

```python
def square(x: float) -> float:
    """Return x squared."""
```

---

## File Input and Output

Every file a script reads or writes lives in the **same folder as the script
itself**. Data written by one lab is read back by its partner in that folder,
so a student can open any session folder and find the inputs and the outputs
together.

A bare filename does not do this. Python resolves a relative path against the
current working directory, which is wherever the terminal happened to be when
the script was launched. Run the same script from the repository root and the
output lands in the repository root, where its partner script will not find
it.

Anchor the path to the script instead:

```python
file_name = "samples.csv"
file_path = Path(__file__).parent / file_name
np.savetxt(file_path, samples, fmt="%3.6f", delimiter=",")
print(f"Saved file {file_path}")
```

Reading uses the same anchor, so the writer and the reader always agree:

```python
file_name = "samples.csv"
file_path = Path(__file__).parent / file_name
times, volts = np.genfromtxt(file_path, delimiter=",", unpack=True)
```

Two rules follow, and the second is the one that actually gets broken:

- Build `file_path` from `Path(__file__).parent`, never from the bare name.
- Pass `file_path` to **every** call that touches the file. Computing
  `file_path` and then handing `file_name` to `np.savetxt` looks correct at a
  glance and silently writes to the wrong folder.

Print the full `file_path` rather than the bare name when reporting a save.
A student who sees the whole path can tell at once where the file went.

`nb_check.py` enforces this, and `githooks/pre-commit` refuses a commit that
breaks it.

### Notebooks

A notebook has no `__file__`, so it cannot use that anchor, and writing
`Path(__file__)` in a cell raises `NameError`. Notebook file names resolve
against the kernel's working directory, which is the folder holding the
notebook. ``qis101.code-workspace`` pins this with

    "jupyter.notebookFileRoot": "${fileDirname}"

so a bare `"ray.csv"` in a notebook means the `ray.csv` sitting next to it.
That setting is the extension's own default; stating it in the workspace keeps
a machine-level override from breaking a lab.

Notebooks that write files should say so plainly. Set the destination once,
near the top, and name it in the output:

```python
# Notebooks have no __file__, so anchor output to the working directory
LAB_DIR = Path.cwd()
print(f"Files will be written to {LAB_DIR}")
```

---

## Environment Notes

These are properties of the development machine, not style rules.

### Force the inline matplotlib backend in notebooks

The venv's default backend is `qtagg`, which hangs a Jupyter kernel. Two
layers guard against this, and both should stay in place:

- Every notebook that imports matplotlib puts `%matplotlib inline` in its
  first code cell, right after the docstring and cell-label comment. This
  travels with the `.ipynb` file, so it works for students on any machine.
- `nb_check.py` enforces that rule, so a notebook cannot quietly ship
  without it. `githooks/pre-commit` runs the check against staged
  notebooks and refuses the commit if one is missing the magic. Register
  the hook directory once per clone:

      git config core.hooksPath githooks

  Run the same check over the whole repository at any time with
  `uv run --no-project python nb_check.py`.

A VS Code setting cannot serve as the backstop here. `jupyter.runStartupCommands`
is declared by the Jupyter extension with `"scope": "application"`, which means
VS Code reads it only from user settings and ignores it in a workspace or folder
file, where it renders grayed out.

A notebook that needs a live animation uses `%matplotlib widget`, never the Qt
backend. Standalone `.py` scripts are unaffected and keep their interactive Qt
window. See `.claude/skills/machine-environment-notes/SKILL.md` for why the Qt
backend blocks the kernel and how to diagnose a cell that never finishes.

### Notebooks that will not stop showing as modified

`.gitattributes` routes every `.ipynb` through the `nbclean` clean filter, which
drops `metadata.language_info.version` and the per-cell `metadata.execution`
timestamps on the way into the index. The working copy keeps those fields. That
is deliberate, and it is also why a notebook can get stuck showing as modified
when nothing about it changed.

The filter has no smudge half, so the file git checks out is *smaller* than the
file Jupyter writes back the first time the notebook is opened. Git records the
size it checked out, sees a different size on disk, and concludes the file
changed. It reports the notebook modified without ever running the filter to
compare the cleaned content. Once that happens the notebook stays modified
forever.

Nothing about the filter is broken, which is what makes this so slow to
diagnose. Every obvious check comes back innocent, because none of them go
through the path git is skipping:

- `git diff` prints nothing.
- `git hash-object <notebook>` equals the blob in the index and in `HEAD`.
- `git -c filter.nbclean.required=true status` raises no error, since a filter
  that is never invoked cannot fail.

Confirm it by comparing the size git cached against the size on disk:

```bash
git ls-files --debug -- "<notebook>"   # the "size:" line
stat -c %s "<notebook>"                # Linux
(Get-Item "<notebook>").Length         # Windows PowerShell
```

If the cached size is the smaller, cleaned size, this is it. Fix it with

```bash
git add --renormalize .
```

which runs the filter, records the right size, and stages nothing, because the
blob it produces is the one already committed. `git update-index
--really-refresh` does the same job.

It is a once-per-notebook event, not a recurring tax. In practice the only field
written back is `language_info.version`, whose length is fixed, so the notebook
grows by the same few bytes the first time a kernel attaches and then keeps that
exact size on every later open. After one `git add --renormalize .` the cached
size is the grown size and the notebook stays clean. Only a fresh clone,
checkout, or branch switch resets it, and so does moving to a new Python patch
release, since that changes the length of the version string.

Nothing writes the per-cell `metadata.execution` timestamps here. The VS Code
Jupyter extension does not record them, and no notebook in the repository
carries any. `nb_clean.py` still strips them, because JupyterLab and `nbconvert
--execute` do write them, and a notebook that arrives from either would
otherwise carry timing noise into the index.

`git add --renormalize .` is safe to run at any point and is the first thing to
try when a notebook you never edited appears in `git status`.

---

## Reference Material Loaded On Demand

Two longer references used to live in this file. They are now skills under
`.claude/skills/`, so Claude Code loads them only when the task calls for
them instead of on every session. Other tools should read the files directly.

- `.claude/skills/office-latex/SKILL.md` - Office-compatible LaTeX for the
  Microsoft 365 Equation Editor (PowerPoint and Word), including Dirac
  bra-ket notation.
- `.claude/skills/machine-environment-notes/SKILL.md` - diagnosing a notebook
  cell that hangs or never finishes, clearing orphaned kernel processes,
  reloading VS Code after a `uv sync`, and why quantum chemistry packages
  cannot be installed on this machine.
