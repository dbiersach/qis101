"""nb_clean.py

Git clean filter that strips volatile metadata out of Jupyter notebooks.

Jupyter rewrites two fields every time a notebook is saved, neither of
which says anything about the content:

- `metadata.language_info.version`, the interpreter version of whatever
  machine happened to run the notebook last.
- `execution` in each cell's metadata, a block of iopub timestamps left
  behind by a notebook run.

Both produce diffs on files nobody edited. This filter removes them on
the way into the git index, so the working copy keeps whatever Jupyter
wrote and the committed file stays clean.

`.gitattributes` points `*.ipynb` at this filter, but git refuses to run a
filter that the local repository has not registered, because a filter is
executable code arriving with a clone. Register it once per clone:

    git config filter.nbclean.clean "uv run --quiet --no-project python nb_clean.py"

Until that command is run the filter is simply skipped and notebooks are
staged verbatim, so a fresh clone still works.
"""

import json
import sys
from typing import Any


def clean(notebook: dict[str, Any]) -> dict[str, Any]:
    """
    Remove the volatile metadata fields from a parsed notebook.

    Parameters
    ----------
    notebook : dict[str, Any]
        The notebook, already parsed from JSON.

    Returns
    -------
    dict[str, Any]
        The same notebook, modified in place and returned for convenience.
    """
    notebook.get("metadata", {}).get("language_info", {}).pop("version", None)

    for cell in notebook.get("cells", []):
        cell.get("metadata", {}).pop("execution", None)

    return notebook


def main() -> None:
    """Read a notebook on stdin and write the cleaned form to stdout."""
    raw = sys.stdin.buffer.read()

    # A filter must never destroy content it does not understand. Anything
    # that is not a parseable notebook passes straight through untouched.
    try:
        notebook = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        sys.stdout.buffer.write(raw)
        return

    # indent=1 with sorted keys is the format nbformat itself writes, so a
    # cleaned file is byte identical to what Jupyter would have saved
    text = json.dumps(clean(notebook), indent=1, sort_keys=True, ensure_ascii=False)
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
