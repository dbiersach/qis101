---
name: machine-environment-notes
description: Environment limits and failure modes of this development machine. Use when a Jupyter notebook cell hangs, spins forever, or never finishes in VS Code (Qt matplotlib backend blocking the kernel, ipykernel 7 dropping replies on Windows, orphaned kernel processes, a stale extension host after uv sync), and when quantum chemistry work needs pyscf, qiskit-nature, or openfermion, which cannot be installed here.
---

# Environment notes

These are properties of the development machine, not style rules. They are
recorded here so that time is not lost rediscovering them.

## A notebook cell that never finishes has three likely causes

The symptom is identical in all three cases - a cell that spins forever with
no error - so check them in this order rather than guessing:

1. A GUI matplotlib backend blocking the kernel (see below).
2. `ipykernel` 7 dropping reply messages on Windows (see below).
3. A stale VS Code extension host or an orphaned kernel process (see below).

Before any of that, confirm the notebook code is innocent by executing it
outside VS Code:

```sh
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute "path\to\notebook.ipynb"
```

If that succeeds, the fault is in the environment, not the courseware.

## Cause 1: a GUI matplotlib backend blocking the kernel

The venv's default backend is `qtagg`, because PySide6 is installed so that
standalone `.py` scripts can open an interactive plot window. That default is
wrong inside a Jupyter kernel: `plt.show()` opens a Qt window that blocks the
kernel, and the cell never returns.

The two guards against this are documented in `AGENTS.md` under "Force the
inline matplotlib backend in notebooks", and both should stay in place. If a
cell hangs, first check that the notebook actually carries `%matplotlib
inline` in its first code cell.

## Cause 2: `ipykernel` 7 drops replies on Windows

`ipykernel` 7 rewrote the subshell and control-channel handling, and on
Windows it drops `execute_reply` and idle messages. VS Code then shows a cell
as perpetually running even though the kernel already finished and went idle
(vscode-jupyter issues #17228 and #17234).

`pyproject.toml` pins `ipykernel<7`, which resolves to the 6.x line. Nothing
in this repository uses subshells, so there is no functional loss. Do not
lift the pin without rechecking those issues.

## Cause 3a: orphaned kernel processes

Kernel processes accumulate across VS Code sessions and outlive the notebooks
that started them. Each one holds a connection file in
`%APPDATA%\jupyter\runtime`, and a pile of them is what a "zombie kernel"
looks like from the notebook side.

List them, and check the `CommandLine` for `ipykernel_launcher` entries
pointing at this project's `.venv`:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, CommandLine
```

Stop those with `Stop-Process -Force`, then reload the window as described
below.

## Cause 3b: reload VS Code after a `uv sync` that changes packages

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
- Besides the `nbconvert` check above, driving a kernel directly through
  `jupyter_client.start_new_kernel` also isolates this case: it talks to the
  same kernel over ZMQ without the extension host in the way.

## Quantum chemistry packages cannot be installed on this machine

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
