"""nb_check.py

Check two things that are easy to get wrong and slow to diagnose:

1. Every notebook which uses matplotlib selects an inline backend in its
   first code cell.
2. Every script reads and writes its data files in the folder that holds
   the script, rather than in whatever directory it was launched from.

The venv's default backend is `qtagg`, because a Qt binding is installed so
that standalone `.py` scripts can open an interactive plot window. That
default is wrong inside a Jupyter kernel: `plt.show()` opens a Qt window that blocks
the kernel, and the cell never returns. The fix is a `%matplotlib inline`
line in the notebook itself, which travels with the `.ipynb` file and so
works for students on any machine.

The path rule matters for the same reason. Python resolves a bare filename
against the current working directory, so a script launched from the
repository root writes its output there instead of next to itself, and the
partner script that reads the file then cannot find it. Anchoring the name
to `Path(__file__).parent` makes a lab work from any directory.

Both mistakes look fine on the page and surface much later, as a cell that
spins forever or as a file that is not where it should be. This script
turns them into an immediate, obvious error.

Run it over the whole repository:

    uv run --no-project python nb_check.py

Or over specific files, which is what the pre-commit hook does:

    uv run --no-project python nb_check.py path/to/one.ipynb path/to/two.py

`githooks/pre-commit` runs this automatically against staged files.
Git will not run a hook directory a clone has not registered, so enable it
once per clone:

    git config core.hooksPath githooks

Skipping that step is harmless: the check is then simply never run.

The `--hook` flag reads a Claude Code PostToolUse payload on stdin and
checks whichever file the tool just wrote. It exits 2 when a rule is
broken, which is the exit code that hands the message back to the model,
so the file gets corrected as soon as it is written rather than at commit
time. `.claude/settings.json` wires it up.
"""

import ast
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

# Directories that hold copies of files rather than sources we maintain.
SKIP_DIRS = {".ipynb_checkpoints", ".venv", ".git", "__pycache__"}

# Calls whose first positional argument names a file on disk. A call that
# takes an already-open handle, such as json.load, is deliberately absent:
# the path was decided by the open() above it, and that is what gets checked.
PATH_CALLS = {
    "np.genfromtxt",
    "np.loadtxt",
    "np.savetxt",
    "np.load",
    "np.save",
    "np.savez",
    "np.savez_compressed",
    "np.fromfile",
    "pd.read_csv",
    "pd.read_json",
    "pd.read_excel",
    "pd.read_table",
    "plt.savefig",
    "plt.imread",
    "plt.imsave",
    "open",
    "lzma.open",
    "gzip.open",
    "bz2.open",
}

# Path methods where the receiver, not an argument, names the file:
# file_path.open(...) rather than open(file_path, ...).
PATH_METHODS = {"open", "read_text", "write_text", "read_bytes", "write_bytes"}


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


def strip_magics(source: str) -> str:
    """Blank out the IPython magic and shell lines, which are not valid Python."""
    lines = source.splitlines()
    kept = ["" if line.lstrip().startswith(("%", "!")) else line for line in lines]
    return "\n".join(kept)


def dotted_name(node: ast.AST) -> str:
    """Return a call target written out, so np.savetxt reads as "np.savetxt"."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return base + "." + node.attr if base else node.attr
    return ""


def mentions_script_folder(node: ast.AST) -> bool:
    """Return True if the expression reaches for __file__ anywhere inside."""
    return any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node))


def anchored_names(tree: ast.AST) -> set[str]:
    """
    Collect the variables that hold a path which is already pinned down.

    A script normally writes ``file_path = Path(__file__).parent / file_name``
    and then uses ``file_path`` further down, so following the assignment is
    what lets the check see the anchor. A spelled-out absolute path counts
    too, since it names one place and no other. The loop repeats because one
    such name can be built out of another.

    Parameters
    ----------
    tree : ast.AST
        The parsed module.

    Returns
    -------
    set[str]
        Every variable name that traces back to __file__ or to an
        absolute path.
    """
    names: set[str] = set()

    for _ in range(6):
        found_more = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                targets, value = [node.optional_vars], node.context_expr
            else:
                continue

            anchored = (
                mentions_script_folder(value)
                or is_absolute_literal(value)
                or any(
                    isinstance(n, ast.Name) and n.id in names for n in ast.walk(value)
                )
            )
            if not anchored:
                continue

            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    found_more = True

        if not found_more:
            break

    return names


def parameter_names(tree: ast.AST) -> set[str]:
    """Return every function parameter name in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            names.add(arg.arg)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)
    return names


def is_absolute_literal(node: ast.AST) -> bool:
    """Return True for a spelled-out absolute path, which is always deliberate."""
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Constant) or not isinstance(inner.value, str):
            continue
        text = inner.value
        if text.startswith("/") or text[1:3] in ("://", ":\\", ":/"):
            return True
    return False


def path_argument(node: ast.Call) -> ast.AST | None:
    """Return the expression naming the file, or None if this is not file I/O."""
    name = dotted_name(node.func)

    # file_path.open(), file_path.read_text(): the receiver is the path.
    # Path.open(name) calls the method unbound, so there the path is the
    # argument and the receiver is only the class.
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in PATH_METHODS
        and name not in PATH_CALLS
    ):
        receiver = node.func.value
        if isinstance(receiver, ast.Name) and receiver.id == "Path":
            return node.args[0] if node.args else None
        return receiver

    if name in PATH_CALLS and node.args:
        return node.args[0]

    return None


def check_paths(path: Path) -> list[str]:
    """
    Check one script for file I/O that is not anchored to the script's folder.

    Parameters
    ----------
    path : Path
        The .py file to inspect.

    Returns
    -------
    list[str]
        One description per unanchored call. Empty when the script is fine.
    """
    # These two try blocks stay separate, each catching one class. The
    # combined "except (A, B)" form is what a py314 formatter rewrites into
    # the unparenthesized "except A, B", which is 3.14 and later only and
    # would stop this file running on 3.13.
    try:
        source = path.read_text(encoding="utf-8")
    except ValueError:
        return []

    try:
        tree = ast.parse(source)
    # A syntax error means an exercise left deliberately incomplete for a
    # student to finish. That is not this script's business to report.
    except SyntaxError:
        return []

    anchored = anchored_names(tree)
    parameters = parameter_names(tree)
    problems: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        target = path_argument(node)
        if target is None:
            continue
        if mentions_script_folder(target) or is_absolute_literal(target):
            continue
        if any(isinstance(n, ast.Name) and n.id in anchored for n in ast.walk(target)):
            continue
        # A path handed in by the caller was already resolved where it was built.
        if isinstance(target, ast.Name) and target.id in parameters:
            continue

        call = dotted_name(node.func) or "this call"
        problems.append(
            f"line {node.lineno}: {call} is given {ast.unparse(target)}, "
            "which resolves against the working directory"
        )

    return problems


def check_notebook_paths(path: Path) -> list[str]:
    """
    Check that a notebook does not reach for __file__, which it does not have.

    Parameters
    ----------
    path : Path
        The notebook to inspect.

    Returns
    -------
    list[str]
        One description per use of __file__. Empty when the notebook is fine.
    """
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    # UnicodeDecodeError and json.JSONDecodeError both subclass ValueError,
    # and catching the base class avoids a tuple that a py314 formatter
    # would rewrite into syntax 3.13 cannot read.
    except ValueError:
        return []

    problems: list[str] = []
    for number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        # Parsing rather than searching the text keeps a comment that merely
        # mentions __file__, explaining why a notebook cannot use it, from
        # being read as a use of it.
        try:
            tree = ast.parse(strip_magics(cell_source(cell)))
        except SyntaxError:
            continue
        if mentions_script_folder(tree):
            problems.append(
                f"cell {number}: uses __file__, which a notebook does not define"
            )
    return problems


def files_to_check(argv: list[str]) -> list[Path]:
    """Return the files named on the command line, or every one in the tree."""
    if argv:
        return [Path(a) for a in argv if a.endswith((".ipynb", ".py"))]

    found = [p for p in Path().rglob("*.ipynb") if not SKIP_DIRS.intersection(p.parts)]
    found += [p for p in Path().rglob("*.py") if not SKIP_DIRS.intersection(p.parts)]
    return sorted(found)


def problems_for(path: Path) -> list[tuple[str, str]]:
    """
    Return every rule this one file breaks.

    Parameters
    ----------
    path : Path
        The notebook or script to inspect.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of rule name and description. The rule name selects which
        explanation to print, so a path mistake is not answered with
        advice about matplotlib.
    """
    if path.suffix == ".ipynb":
        found: list[tuple[str, str]] = []
        backend = check(path)
        if backend is not None:
            found.append(("backend", backend))
        found += [("path", p) for p in check_notebook_paths(path)]
        return found
    return [("path", p) for p in check_paths(path)]


BACKEND_ADVICE = (
    "A notebook that imports matplotlib needs this in its first code cell,"
    "\nright after the docstring and the cell-label comment:"
    "\n"
    "\n    %matplotlib inline"
    "\n"
    "\nWithout it the kernel uses the Qt backend, and plt.show() hangs the"
    "\ncell instead of drawing."
    "\n"
    "\nSee the First Code Cell section of AGENTS.md."
)

PATH_ADVICE = (
    "A script names its data files relative to itself, so that it works no"
    "\nmatter which directory it was launched from:"
    "\n"
    '\n    file_name = "samples.csv"'
    "\n    file_path = Path(__file__).parent / file_name"
    '\n    np.savetxt(file_path, samples, delimiter=",")'
    "\n"
    "\nPass file_path to every call that touches the file. Building it and"
    "\nthen passing the bare file_name is the usual slip: it looks right and"
    "\nquietly writes to whatever directory the script was launched from."
    "\n"
    "\nA notebook has no __file__ and does not need one: its working"
    "\ndirectory is already the folder that holds the notebook, so a bare"
    "\nfile name there already means the file sitting beside it."
    "\n"
    "\nSee the File Input and Output section of AGENTS.md."
)


def advice_for(rules: set[str]) -> str:
    """Return the explanations for the rules that were actually broken."""
    parts = []
    if "backend" in rules:
        parts.append(BACKEND_ADVICE)
    if "path" in rules:
        parts.append(PATH_ADVICE)
    return "\n\n".join(parts)


def hook_main() -> int:
    """
    Check the one file named by a Claude Code PostToolUse payload on stdin.

    Returns
    -------
    int
        2 when the file breaks a rule, which is the exit code that feeds
        the message back to the model. 0 in every other case, so an
        unrelated edit never interrupts the session.
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

    if not raw.endswith((".ipynb", ".py")):
        return 0

    path = Path(raw)
    if not path.is_file():
        return 0

    problems = problems_for(path)
    if not problems:
        return 0

    for _, problem in problems:
        print(f"{path.name} {problem}.", file=sys.stderr)
    print(file=sys.stderr)
    print(advice_for({rule for rule, _ in problems}), file=sys.stderr)
    return 2


def main() -> int:
    """Check the requested files and report every rule they break."""
    if "--hook" in sys.argv[1:]:
        return hook_main()

    problems: list[tuple[Path, str, str]] = []

    checked = 0
    for path in files_to_check(sys.argv[1:]):
        # A staged deletion leaves a path that no longer exists on disk.
        if not path.is_file():
            continue
        checked += 1
        for rule, problem in problems_for(path):
            problems.append((path, rule, problem))

    if not problems:
        # Count what was actually opened, not what was asked for. A name
        # that reached this script mangled would otherwise be reported as
        # a file that passed.
        print(f"nb_check: {checked} file(s) checked, all fine")
        return 0

    print("nb_check: these files break a rule\n", file=sys.stderr)
    for path, _, problem in problems:
        print(f"  {path}\n      {problem}", file=sys.stderr)

    print(file=sys.stderr)
    print(advice_for({rule for _, rule, _ in problems}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
