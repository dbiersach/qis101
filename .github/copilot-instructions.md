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

### Main Guard

- Use `if __name__ == "__main__":` only if the script contains user-defined functions (via `def` statements)
- If a script contains only direct procedural code with no function definitions, the main guard is not required
- When used, wrap all main logic in a `main()` function called from the guard

Example (script with functions):

```python
def compute_value(x: float) -> float:
    """Return computed result."""
    return x ** 2

def main() -> None:
    result = compute_value(5)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
```

Example (script without functions - no guard needed):

```python
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
print(f"Mean: {np.mean(y)}")
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

## Summary

All code in this repository should:

- Be easy to read
- Be easy to teach from
- Clearly explain both **how** and **why**
- Follow consistent structure across notebooks and scripts
