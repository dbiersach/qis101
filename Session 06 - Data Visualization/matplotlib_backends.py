#!/usr/bin/env -S uv run
"""matplotlib_backends.py

List and test available Matplotlib backends

Notes
-----
When using the QtAgg backend, `matplotlib.use("QtAgg")` must be called
before importing `matplotlib.pyplot`. Once `pyplot` is imported, the
backend is effectively locked in and attempting to switch backends may
raise an exception or produce unexpected behavior

This script:
- Prints the active Matplotlib configuration file
- Lists all built-in backend names
- Forces the QtAgg backend and verifies it is active
- Attempts to initialize each available backend
- Reports which backends are usable in the current Python environment

If Ruff reports E402 ("module level import not at top of file") for the
`pyplot` import, that warning is expected and intentional because the
backend must be selected before importing `matplotlib.pyplot`
"""

import matplotlib

print(f"Path to config file: {matplotlib.matplotlib_fname()}", end="\n\n")

all_backends = sorted(matplotlib.backends.backend_registry.list_builtin())
print(f"All backends: {all_backends}", sep=",", end="\n\n")

# To use QtAgg with PyQt6, must set before importing pyplot
matplotlib.use("QtAgg")
print(f"Active Backend: {matplotlib.get_backend()}", end="\n\n")

import matplotlib.pyplot as plt  # noqa: E402

print("After importing pyplot...\n")

available_backends = []
for backend in all_backends:
    try:
        matplotlib.use(backend, force=True)
        available_backends.append(backend)
    except Exception as error:
        print(f"Error occurred while trying backend '{backend}': {error}")

print()
print(f"Available backends: {available_backends}", sep=",", end="\n\n")

matplotlib.use("TkAgg")
print(f"Active Backend: {plt.get_backend()}")
