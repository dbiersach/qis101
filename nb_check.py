"""nb_check.py

Check that every notebook which uses matplotlib also selects an inline
backend in its first code cell.

The venv's default backend is `qtagg`, because a Qt binding is installed so
that standalone `.py` scripts can open an interactive plot window. That
default is wrong inside a Jupyter kernel: `plt.show()` opens a Qt window that blocks
the kernel, and the cell never returns. The fix is a `%matplotlib inline`
line in the notebook itself, which travels with the `.ipynb` file and so
works for students on any machine.

It is easy to forget that line on a new notebook, and the symptom appears
much later as a cell that spins forever. This script turns the omission
into an immediate, obvious error.

Run it over the whole repository:

    uv run --no-project python nb_check.py

Or over specific notebooks, which is what the pre-commit hook does:

    uv run --no-project python nb_check.py path/to/one.ipynb

`githooks/pre-commit` runs this automatically against staged notebooks.
Git will not run a hook directory a clone has not registered, so enable it
once per clone:

    git config core.hooksPath githooks

Skipping that step is harmless: the check is then simply never run.

The `--hook` flag reads a Claude Code PostToolUse payload on stdin and
checks whichever file the tool just wrote. It exits 2 when the magic is
missing, which is the exit code that hands the message back to the model,
so a notebook gets corrected as soon as it is written rather than at commit
time. `.claude/settings.json` wires it up.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# A notebook needs the magic only if it actually pulls matplotlib in.
USES_MATPLOTLIB = re.compile(
    r"^\s*(?:import\s+matplotlib|from\s+matplotlib\b)", re.MULTILINE
)

# `inline` is the normal choice. `widget` is the documented alternative for
# a notebook that needs a live animation. Either one avoids the Qt window.
SELECTS_BACKEND = re.compile(r"^\s*%matplotlib\s+(?:inline|widget)\b", re.MULTILINE)

# Directories that hold copies of notebooks rather than sources we maintain.
SKIP_DIRS = {".ipynb_checkpoints", ".venv", ".git"}


def cell_source(cell: dict[str, Any]) -> str:
    """Return a notebook cell's source as one string."""
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def check(path: Path) -> str | None:
    """
    Check one notebook for a missing inline backend declaration.

    Parameters
    ----------
    path : Path
        The notebook to inspect.

    Returns
    -------
    str | None
        A description of the problem, or None when the notebook is fine.
    """
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"could not be parsed as a notebook ({exc})"

    code_cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        return None

    sources = [cell_source(c) for c in code_cells]

    # A notebook that never imports matplotlib has nothing to declare.
    if not any(USES_MATPLOTLIB.search(s) for s in sources):
        return None

    if not any(SELECTS_BACKEND.search(s) for s in sources):
        return "imports matplotlib but never runs %matplotlib inline"

    # The magic has to run before any plotting, so it belongs in the first
    # code cell. Further down it may execute after a plot was already made.
    if not SELECTS_BACKEND.search(sources[0]):
        return "has %matplotlib inline, but not in the first code cell"

    return None


def notebooks_to_check(argv: list[str]) -> list[Path]:
    """Return the notebooks named on the command line, or every one in the tree."""
    if argv:
        return [Path(a) for a in argv if a.endswith(".ipynb")]

    return sorted(
        p for p in Path().rglob("*.ipynb") if not SKIP_DIRS.intersection(p.parts)
    )


def hook_main() -> int:
    """
    Check the one file named by a Claude Code PostToolUse payload on stdin.

    Returns
    -------
    int
        2 when the notebook is missing the magic, which is the exit code
        that feeds the message back to the model. 0 in every other case,
        so an unrelated edit never interrupts the session.
    """
    try:
        payload = json.loads(sys.stdin.read())
    # UnicodeDecodeError and json.JSONDecodeError both subclass ValueError.
    # Catching the base class keeps this file valid on Python 3.13 as well:
    # the unparenthesized "except A, B" form is 3.14 and later only.
    except ValueError:
        return 0

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response") or {}
    raw = (
        tool_input.get("file_path")
        or tool_input.get("notebook_path")
        or tool_response.get("filePath")
        or ""
    )

    if not raw.endswith(".ipynb"):
        return 0

    path = Path(raw)
    if not path.is_file():
        return 0

    problem = check(path)
    if problem is None:
        return 0

    message = [
        f"{path.name} {problem}.",
        "",
        "Add `%matplotlib inline` to the first code cell, right after the",
        "docstring and the cell-label comment. Without it the kernel uses",
        "the Qt backend and plt.show() hangs the cell. See the First Code",
        "Cell section of AGENTS.md.",
    ]
    for line in message:
        print(line, file=sys.stderr)
    return 2


def main() -> int:
    """Check the requested notebooks and report any that are missing the magic."""
    if "--hook" in sys.argv[1:]:
        return hook_main()

    problems: list[tuple[Path, str]] = []

    paths = notebooks_to_check(sys.argv[1:])
    for path in paths:
        # A staged deletion leaves a path that no longer exists on disk.
        if not path.is_file():
            continue
        problem = check(path)
        if problem is not None:
            problems.append((path, problem))

    if not problems:
        print(f"nb_check: {len(paths)} notebook(s) checked, all fine")
        return 0

    print("nb_check: the inline matplotlib backend is missing\n", file=sys.stderr)
    for path, problem in problems:
        print(f"  {path}\n      {problem}", file=sys.stderr)

    print(
        "\nAdd this to the first code cell, right after the docstring and"
        "\nthe cell-label comment:\n"
        "\n    %matplotlib inline\n"
        "\nWithout it the kernel uses the Qt backend, and plt.show() hangs"
        "\nthe cell instead of drawing.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
