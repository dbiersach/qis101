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
compiler and supports no packages at all, so expressions that render fine in
a real LaTeX compiler can "fail miserably" here.

Assume the equation is going into the Equation Editor in **LaTeX input
mode**, and return the raw source in a code block so it can be copied
directly.

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

### Never use package-dependent macros

Office has no package system. Anything that a normal LaTeX document would
pull in from `amsmath`, `braket`, or `physics` simply does not exist in the
build-up engine, and the equation fails.

Never emit these:

```latex
\ket{\psi}
\bra{\psi}
\braket{\phi|\psi}
\lvert\psi\rangle
\langle\psi\rvert
```

Write every bracket out longhand with `\left` and `\right` instead.

### Dirac (bra-ket) notation

| Notation | Office-compatible LaTeX |
|---|---|
| Ket | `\left\|\psi\right\rangle` |
| Bra | `\left\langle\psi\right\|` |
| Inner product | `\left\langle\phi\middle\|\psi\right\rangle` |
| Matrix element | `\left\langle\phi\middle\|\hat{A}\middle\|\psi\right\rangle` |
| Ket in a fraction | `\frac{\left\|\psi\right\rangle}{\sqrt{2}}` |

Never write a ket with `\lvert` - always `\left|`.

Use `\middle|` for a bar that sits *inside* a bracket pair, as in an inner
product or a matrix element. Splitting the same expression into two separate
pairs, `\left\langle\phi\right|\hat{A}\left|\psi\right\rangle`, also builds
correctly, but `\middle|` keeps it as one group so every glyph grows to the
same height.

Keep the delimiters explicit inside fractions, where a lone bar does the most
damage:

```latex
\frac{\left\langle\psi\middle|\hat{H}\middle|\psi\right\rangle}
{\left\langle\psi\middle|\psi\right\rangle}
```

### Composite states, outer products, and operators

Write a composite ket as one bracket pair:

```latex
\left|00\right\rangle
```

Keep both pairs when the product structure is what matters:

```latex
\left|0\right\rangle\left|1\right\rangle
```

Write outer products out in full, and wrap the whole outer product in
parentheses when it acts on a ket. Add the parentheses even where they are
not mathematically required - they make the operator-action structure
unambiguous to a reader:

```latex
(\left|0\right\rangle\left\langle1\right|)\left|0\right\rangle
```

Preserve that grouping when expanding the operation:

```latex
(\left|0\right\rangle\left\langle1\right|)\left|0\right\rangle
=
\left|0\right\rangle
\left(\left\langle1\middle|0\right\rangle\right)
=
0
```

Parenthesize a compound operator whenever adjacency could be misread:

```latex
(\hat{A}+\hat{B})\left|\psi\right\rangle
```

A single named operator needs no parentheses:

```latex
\hat{U}\left|\psi\right\rangle
```

### Tensor products

Use `\otimes` when the tensor product should be explicit:

```latex
\left|\psi\right\rangle\otimes\left|\phi\right\rangle
```

Do not silently collapse an explicit tensor product into juxtaposition
unless a shorter form was requested.

### Other Office gotchas

- Absolute value: `\left|x\right|` (stretchy) or `|x|` (fixed size, fine for
  short contents).
- Unsupported LaTeX keywords in Office: `\eqarray`, `\Middle`, `\ldiv`,
  `\dsmash`. Capital `\Middle` is unsupported; lowercase `\middle` is the one
  to use. In the rare case it misbehaves, the fallback is all fixed-size
  brackets with a plain separator, `\langle\phi|\psi\rangle`, which keeps the
  delimiter count balanced.
- Recommended reference: Microsoft's "Linear format equations using UnicodeMath
  and LaTeX in Word" support page.

### Output conventions

When asked for "PowerPoint LaTeX", "Microsoft LaTeX", or "Equation Editor
LaTeX":

1. Put the copyable source in a fenced `latex` code block, raw and
   unrendered, so it can be pasted straight into the equation field.
2. Use explicit `\left ... \right` delimiters and explicit parentheses.
3. Use no package-dependent commands.
4. Do not convert the expression to UnicodeMath unless UnicodeMath was
   specifically requested.
5. Where practical, also show the equation rendered normally so the result
   can be checked by eye.

---

## Environment Notes

These are properties of the development machine, not style rules. They are
recorded here so that time is not lost rediscovering them.

### Reload VS Code after a `uv sync` that changes packages

After any `uv sync` that adds, removes, or upgrades a package, the VS Code
Jupyter extension can keep a handle on the pre-sync environment. Notebook
cells then hang on the first run - stuck while connecting to the kernel,
with no error message.

- Fix: Command Palette -> **Developer: Reload Window** (restarting the kernel
  alone is sometimes enough).
- The cause is the running extension host being pinned to the old
  environment while the contents of `.venv` are swapped underneath it. The
  notebook, the venv, `ipykernel`, and the matplotlib inline config are all
  fine.
- To tell a real hang from this one, execute the notebook outside VS Code
  with `jupyter nbconvert --execute` or drive a kernel directly through
  `jupyter_client.start_new_kernel`. If those succeed, the problem is the
  extension host, not the code.

### Quantum chemistry packages cannot be installed on this machine

`pyscf` has no Windows wheel for Python 3.13, so installing it falls back to
compiling from source, which fails (no C/C++ compiler, and CMake cannot find
`nmake`). `qiskit-nature` and `openfermion` both need `pyscf` or an
equivalent driver to produce molecular integrals, so they are unavailable
too. There is no WSL fallback on this machine.

To get a real molecular qubit Hamiltonian, either:

1. Compute it from first principles in a self-contained script - STO-3G
   Gaussian integrals -> RHF -> MO transform -> Jordan-Wigner via
   `SparsePauliOp.from_operator` on the JW-mapped matrix, or
2. Use a vetted set of literature coefficients.

Either way, verify before trusting the result: check that
`<HF|H|HF> == E_RHF` and that the FCI minimum eigenvalue matches the known
value.

Two traps to watch for:

- The MO two-electron transform must contract the AO axes against the
  **first** axis of the coefficient matrix `C`, i.e.
  `ip,jq,kr,ls,ijkl->pqrs`.
- `SparsePauliOp.from_operator` reads matrices in Qiskit little-endian
  order. Build the Jordan-Wigner operators with mode 0 as the **rightmost**
  tensor factor so they match a table where `q0` is the first orbital.

---

## Summary

All code in this repository should:

- Be easy to read
- Be easy to teach from
- Clearly explain both **how** and **why**
- Follow consistent structure across notebooks and scripts
