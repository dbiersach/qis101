# nb_check.py vs nb_clean.py

The repository root holds two small Python tools whose names look alike but
whose jobs do not overlap at all:

- **`nb_check.py` is a linter.** It looks for two specific mistakes, reports
  them, and refuses to let them through. It never changes a file.
- **`nb_clean.py` is a git filter.** It quietly rewrites notebooks on their
  way into git, removing metadata that changes on every run. It never
  rejects anything.

One protects **correctness**. The other keeps **history clean**.

---

## At a glance

| | `nb_check.py` | `nb_clean.py` |
|---|---|---|
| **Kind of tool** | Linter / validator | Git clean filter |
| **Purpose** | Catch two mistakes that are easy to make and slow to diagnose | Strip run-to-run metadata so it never shows up as a diff |
| **Files it handles** | `.ipynb` and `.py` | `.ipynb` only |
| **Changes files?** | No. Reports and exits nonzero. | Yes, but only the copy stored in git. The working file is untouched. |
| **Who runs it** | The pre-commit hook, the Claude Code hook, or you by hand | Git itself, during `git add` |
| **One-time setup per clone** | `git config core.hooksPath githooks` | `git config filter.nbclean.clean "uv run --quiet --no-project python nb_clean.py"` |
| **If the setup is skipped** | Nothing checks for the problems | Notebooks are committed with the extra metadata |
| **Size** | About 530 lines (parses Python with `ast`) | About 70 lines, standard library only |

---

## nb_check.py

### What it checks

**Rule 1: a notebook that uses matplotlib must set an inline backend.**
If a notebook imports matplotlib anywhere, its *first* code cell must contain
`%matplotlib inline` (or `%matplotlib widget` for a live animation). The
venv's default backend is `qtagg`, so without the magic `plt.show()` opens a
Qt window that blocks the Jupyter kernel, and the cell spins forever.

**Rule 2: a script must anchor its data files to its own folder.**
A script that writes `np.savetxt("samples.csv", ...)` puts the file in
whatever directory the terminal was in when it was launched, not next to the
script. The partner script that reads the file then cannot find it. The fix
is to build the path from `Path(__file__).parent / file_name` and pass that
path to every call that touches the file. Notebooks get a matching check,
since a notebook has no `__file__` at all.

Both mistakes look fine on the page and only show up later, as a hung cell
or a missing file. `nb_check.py` turns them into an immediate, obvious error.

### How and when it is called

It has three entry points. They all run the same checks.

**1. The git pre-commit hook (`githooks/pre-commit`).**
Runs automatically every time you run `git commit`, once the hook directory
is registered with `git config core.hooksPath githooks`. The hook asks git
for the staged `.ipynb` and `.py` files that were added, copied, modified, or
renamed (deletions are skipped), and passes them to the checker:

```sh
git diff --cached --name-only --diff-filter=ACMR -z -- '*.ipynb' '*.py' \
    | xargs -0 uv run --quiet --no-project python nb_check.py
```

The list is NUL-separated (`-z` and `xargs -0`) because every session folder
has spaces in its name. If any file breaks a rule, the script exits with
code 1 and git aborts the commit. To commit anyway, for instance while a lab
is still a work in progress, use `git commit --no-verify`.

**2. The Claude Code hook (`.claude/settings.json`).**
A `PostToolUse` hook runs after every `Write`, `Edit`, or `NotebookEdit`
that Claude Code makes:

```sh
uv run --quiet --no-project python nb_check.py --hook
```

With `--hook`, the script reads Claude Code's JSON payload from standard
input, pulls out the path of the file just written, and checks only that
file. If a rule is broken it exits with code **2**, which is the code that
sends the message back to the model, so the file gets fixed in the same turn
instead of at commit time. Any other file, or any problem reading the
payload, exits 0 so an unrelated edit never interrupts the session.

**3. By hand.**
Check the whole repository:

```sh
uv run --no-project python nb_check.py
```

With no arguments it walks the repository for every `.ipynb` and `.py`,
skipping `.venv`, `.git`, `__pycache__`, and `.ipynb_checkpoints`. You can
also name specific files:

```sh
uv run --no-project python nb_check.py path/to/one.ipynb path/to/two.py
```

On success it prints how many files it actually opened, for example
`nb_check: 298 file(s) checked, all fine`. On failure it lists each file and
problem, then prints advice on the fix.

`--no-project` is used everywhere so the check keeps working even partway
through a `uv sync`, when the project venv may be in an inconsistent state.

---

## nb_clean.py

### What it does

Jupyter rewrites two fields that say nothing about a notebook's content:

- `metadata.language_info.version`, the Python version of whichever machine
  last ran the notebook.
- `metadata.execution` on each cell, a block of timestamps that JupyterLab
  and `nbconvert --execute` write. The VS Code Jupyter extension does not
  write these, but the filter still removes them in case a notebook comes
  from one of those tools.

Left alone, these fields make notebooks nobody edited show up as changed.
`nb_clean.py` deletes them and writes the notebook back out in the same
format Jupyter uses (`indent=1`, sorted keys). Anything that is not valid
notebook JSON passes through unchanged, because a filter must never destroy
content it does not understand.

### How and when it is called

You never run `nb_clean.py` yourself. Git runs it.

`.gitattributes` contains this line:

```text
*.ipynb text eol=lf filter=nbclean
```

That tells git that notebooks go through a filter named `nbclean`. The
command behind the name comes from local git config, set once per clone:

```sh
git config filter.nbclean.clean "uv run --quiet --no-project python nb_clean.py"
```

A git **clean** filter runs whenever git moves a file from your working
folder into the index (the staging area). That happens on `git add`, on
`git commit -a`, and whenever git has to compare a file's contents with the
index. Git sends the notebook to the script on standard input and stores
whatever comes back on standard output.

The result:

- **Your working copy** keeps exactly what Jupyter wrote, version string
  included.
- **The committed copy** has those fields removed.

There is no matching **smudge** filter, which would run on checkout to put
the fields back. That is intentional, and it causes the one known side
effect described next.

Git refuses to run a filter the local clone has not registered, since a
filter is code that arrives with a clone. Until the `git config` line is run
the filter is simply skipped and notebooks are staged verbatim, so a fresh
clone still works.

---

## The side effect: notebooks stuck showing as modified

Because the filter removes bytes and nothing puts them back, the notebook
git checks out is *smaller* than the file Jupyter writes the first time a
kernel attaches to it.

For speed, git does not reread every file on every `git status`. It keeps a
cache in the index of each file's size and modification time. When the cache
still matches what is on disk, git assumes the file is unchanged. When the
size differs, git reports the file as modified, and it does so **without
running the filter** to see whether the cleaned content actually differs.

So once Jupyter adds the version string back, the notebook shows as
modified, and it stays that way. Every check looks innocent:

- `git diff` prints nothing, because once it runs the filter the content
  matches.
- `git hash-object <notebook>` equals the blob in the index and in `HEAD`.
- Nothing is wrong with the filter. It is just never being asked.

You can confirm it by comparing the cached size with the size on disk:

```sh
git ls-files --debug -- "<notebook>"   # look at the "size:" line
(Get-Item "<notebook>").Length         # Windows PowerShell
```

If the cached size is the smaller one, this is the cause.

---

## What `git add --renormalize .` means and does

**"Normalize"** is git's word for converting a file from its working-folder
form into the form stored in the repository. For this repository that means
two things: line endings converted to LF (from the `eol=lf` settings in
`.gitattributes`), and notebooks passed through `nb_clean.py`.

Normally git normalizes a file only when it has reason to think the file
changed. **`--renormalize`** tells git to do it for every tracked file under
the given path (`.` means the whole repository), whether or not it looks
changed. For each file git:

1. Reads the file from disk.
2. Runs it through the line-ending conversion and any clean filter.
3. Hashes the result and writes it into the index.
4. Refreshes the cached size and modification time to match the file as it
   is now on disk.

For a stuck notebook, step 3 produces the exact blob already in the index,
because the only difference was the version string the filter removes. So
nothing new is staged. What changes is step 4: the cache now records the
larger, real size, so `git status` stops flagging the file.

Some practical points:

- **It is safe to run at any time.** It never changes your working files.
  It only updates the index.
- **It can stage real changes.** If a file really does differ after
  normalization, for instance a file you edited, or one with CRLF line
  endings that should be LF, it gets staged. Check `git status` afterward.
  For the stuck-notebook case it stages nothing.
- **It is usually a one-time fix per notebook.** The version string has a
  fixed length, so after the first kernel attach the file keeps the same
  size. Only a fresh clone, a checkout or branch switch, or moving to a new
  Python patch release (which changes the length of the version string)
  brings it back.
- `git update-index --really-refresh` fixes the same problem in a different
  way: it rechecks every file's contents instead of trusting the cache.

---

## Summary

| Question | `nb_check.py` | `nb_clean.py` |
|---|---|---|
| Does it stop a commit? | Yes, if a rule is broken | Never |
| Does it change what is committed? | No | Yes, it removes two metadata fields |
| Does it change your working files? | No | No |
| When does it run? | On `git commit`, after Claude Code edits a file, or by hand | On `git add` and whenever git compares a notebook with the index |
| What breaks if it is not set up? | Hung cells and misplaced data files get through | Noisy diffs on notebooks nobody edited |
