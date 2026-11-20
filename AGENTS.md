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

## Code Screenshots: Actual VS Code, Dark Modern

- Use actual captures of the current source file displayed in VS Code with the
  **Dark Modern** theme. Verify the active theme and the source text before
  capturing. Do not substitute Dark 2026, VS Code Dark, or another theme.
- Do not generate lookalike code images or recreate editor text with a renderer,
  syntax highlighter, drawing library, or image generator. Do not recolor an older
  screenshot to simulate the required theme; retake it in VS Code.
- Prefer lossless PNG captures. Crop the actual capture without retouching its
  text, and preserve its aspect ratio with equal horizontal and vertical scaling.
  Check the actual file encoding, saved image and installed PowerPoint rendering.
- Tell the user immediately before taking control of their machine, including
  when resuming control after a pause. Explain which application will be used;
  avoid selecting or editing source text merely to capture it. Tell the user
  when mouse and keyboard interaction is finished.
- Preserve existing compound arrows, highlights, shape IDs, equations, animation
  targets and build sequences. Move arrow-and-number groups intact and fit the
  existing highlights to the captured source; follow the callout rules below.
  These protections apply to retained content. Apply the justified-removal rules
  below when revised code or the teaching sequence makes an annotation redundant.
- As work reaches each deck after Session 04, check its source-code screenshots
  against the current source and the **Dark Modern** scheme. Retake any that do
  not match when working on that deck. Do not change later decks in advance.

## PowerPoint Code Callouts: Preserve Compound Objects

Treat each numbered arrow as one compound object. Use the matching callout in
`Session 00 - Template.pptx` as the reference for its construction and formatting.
This reference is not permission to normalize or reconstruct an existing object.
Never alter the internal geometry of a red arrow and its number as a side effect
of updating code, replacing screenshots, or aligning annotations.

- Move the entire arrow-and-number group as a unit. Do not independently move,
  resize, shorten, stretch, regroup, or rebuild its children unless the user
  explicitly requests a repair of the compound object itself.
- Preserve the template's arrow length, stroke, arrowhead, number font and size,
  internal spacing, text justification, margins, horizontal and vertical anchors,
  and relative positions. Do not apply generic alignment or sizing defaults.
- Position the arrowhead tip just to the right of the last visible character on
  the target code line, leaving a small, consistent horizontal gap. Align the tip
  vertically with the actual visible character, not the line box, text-box center,
  nominal baseline, or an approximate fraction of the font size. For a closing
  parenthesis, the tip points at the vertical center of the parenthesis's bulge.
- Use the rendered code to check this alignment. Move the whole group to correct
  its target; do not change its internal geometry to make it fit. Resolve or report
  collisions without silently distorting the callout.
- Center red, white, and yellow highlight rectangles vertically around the visible
  characters of the enclosed code snippet. Nudge the rectangle up or down as
  needed; do not center it on the line box or an assumed font baseline. Preserve
  its size, horizontal placement, color, stroke, and other formatting unless the
  user requests those changes too.
- Preserve shape IDs, group membership, stacking order, animation targets, builds,
  and transitions. Preserve animation order, click steps, triggers, delays,
  durations, entrance/exit effects, motion paths, interactive sequences, and slide
  transition settings. Do not recreate or flatten animated objects, reset their
  timing, or change their targets while replacing screenshots. Preserve these
  properties for retained content. The narrow exception is removing effects that
  belong to an obsolete or redundant annotation under the rules below; unrelated
  animation or transition changes require an explicit user request.
- Group bounds and child coordinates must describe the same compound object, so
  selecting or aligning the group behaves correctly. If an inconsistency exists,
  report it; do not silently repair or normalize its geometry during another edit.
- Verify both the rendered appearance and the native group bounds and properties
  in PowerPoint. A correct-looking screenshot alone does not verify the compound
  object's geometry or editing behavior.

## Replacing Code Screenshots: Preserve Annotation Meaning and Builds

- Read and understand the actual source before choosing screenshot boundaries.
  Read the enclosing function or algorithm, not just isolated lines. Identify
  each stage's inputs, operation, result and relationship to the next stage.
- Review the preceding, current and following slides together. State what the
  current slide teaches and where the next teaching step begins. Choose the
  excerpt to serve that instructional flow before considering how many existing
  callouts it contains. Do not add code or comments merely to retain a callout.
- Replacing or overlaying a code screenshot with an updated actual VS Code capture
  is allowed. Align and scale the capture proportionally, and preserve the
  surrounding slide structure and its animation behavior. Prefer replacing the
  existing image content in place so its identity and animation targets survive.
  An overlay must not hide callouts, equations or other content, or show revised
  code at a different build step from the image it replaces.
- Before editing, inspect the original slide and its builds. Record what each
  numbered callout, arrow and highlight identifies: the specific condition,
  expression, statement, output or teaching point. Use an earlier deck when the
  current screenshot or annotation is already mismatched.
- After updating the screenshot, preserve each annotation's original meaning
  wherever that concept remains relevant to this slide's teaching purpose. Follow
  renamed variables, moved statements and equivalent conditions to their current
  locations. Do not align an annotation merely to the nearest line or attach it
  to unrelated code to retain a number or a build count.
- Move arrow-and-number groups only as intact units. Reposition or resize an
  existing highlight rectangle only as needed to enclose its intended revised
  expression, retaining its identity, style and animation target. Preserve all
  unrelated object geometry and formatting.
- Preservation is not a requirement to keep every object or the same build count.
  During an authorized deck revision, remove an annotation when reading the code
  and neighboring slides establishes that its teaching point is obsolete,
  redundant, or belongs on another slide. Explain the specific instructional
  reason in the change record. Such a removal is justified, not arbitrary.
- Remove the redundant screenshot content and its whole arrow-and-number group
  together. Remove only animation effects and build references targeting the
  removed objects; leave no dangling references or empty click steps. Preserve
  the relative order, triggers and timing of the retained teaching sequence, and
  preserve all unrelated transitions and geometry. Verify the intended resulting
  sequence rather than demanding an identical object or click count.
- Do not duplicate a section-heading comment across adjacent screenshots unless
  the repetition has a clear teaching purpose. For example, when slide 139 teaches
  choosing `best_m` and slide 140 teaches updating `(a, b, k)`, end slide 139 before
  `# Update (a, b, k) ...`. Remove slide 139's arrow 7 and its associated effects;
  keep the update comment and arrow 1 on slide 140. Do not append the comment to
  slide 139 merely to give arrow 7 a target.
- If the instructional purpose remains ambiguous after reading the code and
  adjacent slides, report the uncertainty and ask a focused question. Do not
  invent a new meaning for a callout. Clear, justified removals within the
  authorized revision do not require a separate approval merely because they
  change the number of callouts or build steps.
- Preserve a recoverable pre-edit copy. Compare before and after renders and
  object/build properties, and replay affected builds in PowerPoint. Check the
  installed deck, not only a staging copy. Report the exact slides changed and
  any unresolved source-to-screenshot or annotation mismatch. Document justified
  removals and the resulting build sequence separately from accidental damage.

## Review Code Slides as a Teaching Sequence

- Audit adjacent code slides together against the enclosing function in the
  current workspace. Give each slide a distinct teaching purpose: prerequisites,
  initialization, selection, update, result, or another coherent stage. The next
  slide should advance that explanation, not restart the same statements under
  new callout numbers.
- Check both ends of every excerpt. A section-heading comment belongs with the
  statements it introduces. Avoid carrying it onto the preceding slide, and
  avoid repeating setup statements just to fill space or preserve old callouts.
  Retain repeated code only when its specific teaching purpose is clear.
- Consider combining adjacent slides when they explain one coherent operation
  and the combined code remains readable at the existing scale. For example,
  the Chakravala selection conditions and the code that chooses best_m form one
  step; updating (a, b, k) is the following step. Remove a redundant equation
  panel when it merely restates the highlighted conditions and obstructs that
  layout. Do not compress unrelated stages or shrink code merely to save a slide.
- For an authorized merge, reuse actual VS Code captures and move complete
  callout groups with their code. Renumber labels coherently without changing
  their font, margins, justification, internal geometry, or group dimensions.
  Preserve highlights and their meaning. Remap shape and timing IDs only where
  needed to prevent collisions, updating every reference consistently.
- Plan the merged build sequence explicitly. A callout that appeared
  automatically at the start of the second slide may need to become the next
  click in the combined slide. Preserve unrelated effect types and timing;
  remove effects for omitted content and avoid empty clicks. Replay every step
  through the transition to the following slide in PowerPoint.
- Verify all other slides by stable slide identity and package content, because
  deleting an absorbed slide shifts later slide numbers. Report the old-to-new
  numbering and distinguish automatic footer renumbering from content edits.
- In a read-only audit, recommend exact slide-specific edits before changing
  the deck. Separate confirmed source mismatches, layout/build problems,
  pedagogical recommendations, and capture-provenance uncertainty. A similar
  palette alone does not prove an image came from VS Code or used Dark Modern.

---

## Code Comments

Comments describe the code **as it stands now**, in the present tense. The reader
sees only the current version, so anything about the code's past is confusing noise.

- **Never refer to a prior version of the code**, an earlier approach, or how the
  code used to behave.
- **Never describe a past mistake or bug**, or frame the current code as a "fix,"
  "correction," or improvement over something that came before.
- Do not use words like *now*, *again*, *still*, *no longer*, *previously*, *used
  to*, or *the old way* to contrast with an earlier state.
- State what the code does and **why it is written this way**, on its own terms. If
  a design choice needs justifying, justify it by the requirement it meets, not by
  the alternative it replaced.

Example — instead of "The n4 test is the cure: the old (1,1)-only method missed a
third of cases, so we added it," write "Two tests detect a factor: the (1,1) test
and the n4 test. Both are needed because each alone leaves some semiprimes
unfactored."

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
- Do not suppress type-checker diagnostics with `# pyright: ignore`, `# type: ignore`, or equivalent settings unless no viable type-correct solution exists. Investigate and verify proper typing alternatives first. If suppression is unavoidable, explain why and limit it to the specific diagnostic and line.
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
