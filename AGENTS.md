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

## Imports

Follow this order:

1. Standard library
2. Third-party packages
3. Local modules

Use standard aliases:

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Comments and Writing Style

- Comments must be **functional and explanatory**
- Focus on:
  - Purpose of the code
  - Mathematical meaning
  - Instructions to the reader/student

### Avoid

- Decorative or stylistic comments
- Redundant comments that restate obvious code
- Em dashes or long dashes

Instead:

- Use normal hyphens `-`
- Or rewrite the sentence for clarity

---

## Variable Naming

- Use **clear, descriptive names**
- Avoid overly short or cryptic variables unless standard (e.g., `x`, `t`)
- Prefer readability over brevity

---

## Notebook Teaching Style

When writing notebooks:

- Break work into logical steps
- Explain transitions between steps
- Clearly interpret results

Good pattern:

1. Introduce concept
2. Show implementation
3. Run code
4. Interpret output

---

## Formatting

- Code must be compatible with:
  - Ruff
  - Black

- Follow consistent spacing and formatting
- Avoid overly dense code blocks

---

## LaTeX for PowerPoint / Word Equation Editor

When I ask for LaTeX to paste into the **Microsoft 365 Equation Editor**
(PowerPoint or Word: Insert -> Equation -> type LaTeX -> Convert to Math /
"build up"), produce **Office-compatible** LaTeX, not general LaTeX. The
Office build-up engine has stricter delimiter rules than a normal LaTeX
compiler, and expressions that render fine in LaTeX can "fail miserably" here.

### Core rule: delimiters must be balanced by count

Office pairs every opening delimiter (`(`, `[`, `|`, `\langle`, `\lfloor`, ...)
with a matching closer, then builds one auto-sizing bracket object between them.
An **unmatched opener escapes its group** and swallows surrounding content
(e.g. it eats across a fraction bar), producing a mangled result.

- Bad: `\frac{\lvert 1}{2}` - lone `\lvert` has no closer; the bar escapes the
  numerator and wraps the whole fraction.
- Good: `\frac{|1|}{2}` or `\frac{\left|1\right|}{2}` - balanced.

Office does **not** require the two sides to be the *same glyph* - only that
they form one matched `\left ... \right` pair. That is what makes
mixed-delimiter brackets (kets, bras, floors) possible.

### Use `\left ... \right`, not the fixed `\lvert/\rvert` pairs

`\lvert`/`\rvert` (and `\lfloor/\rfloor`, etc.) are **dedicated fixed pairs**:
`\lvert` is hard-wired to seek a matching `\rvert` and will *not* mate with a
different closer. So `\lvert\psi\rangle` fails - `\lvert` wants `\rvert`,
`\rangle` wants `\langle`, and neither finds its partner.

Any bracket whose two sides differ in shape **must** use the generic
`\left ... \right` mechanism, where `\left`/`\right` open/close with whatever
glyph follows and only the count has to balance.

### Dirac (bra-ket) notation

| Notation | Office-compatible LaTeX | Renders |
|---|---|---|
| Ket | `\left|\psi\right\rangle` | \|psi> |
| Bra | `\left\langle\psi\right|` | <psi\| |
| Braket / inner product | `\left\langle\phi\middle|\psi\right\rangle` | <phi\|psi> |
| Matrix element | `\left\langle\phi\right|A\left|\psi\right\rangle` | <phi\|A\|psi> |
| Ket in a fraction | `\frac{\left|\psi\right\rangle}{\sqrt{2}}` | |

Never write a ket with `\lvert` - always `\left|`.

### Other Office gotchas

- Absolute value: `\left|x\right|` (stretchy) or `|x|` (fixed size, fine for
  short contents).
- Unsupported LaTeX keywords in Office: `\eqarray`, `\Middle`, `\ldiv`,
  `\dsmash`. Note capital `\Middle` is unsupported; lowercase `\middle` usually
  works but if it misbehaves, fall back to a plain separator, e.g.
  `\langle\phi|\psi\rangle` (fixed-size brackets are fine for single symbols).
- Prefer fixed-size brackets (`\langle...|...\rangle`) for single letters;
  reach for `\left...\right` when contents are tall (fractions, sums, big
  operators) and the brackets need to grow.
- Recommended reference: Microsoft's "Linear format equations using UnicodeMath
  and LaTeX in Word" support page.

---

## Summary

All code in this repository should:

- Be easy to read
- Be easy to teach from
- Clearly explain both **how** and **why**
- Follow consistent structure across notebooks and scripts
