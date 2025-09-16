#!/usr/bin/env python3
"""Entry-point orchestrating the (placeholder) SCARF pipeline.

Run with either ``--smoke-test`` or ``--full-experiment`` to pick the desired
configuration.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Make top-level project modules importable when ``src/`` is executed directly
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# Also ensure the *src* directory itself is importable for fallback look-ups
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -----------------------------------------------------------------------------
# Local imports (deferred to ensure path fix above is in effect)
# -----------------------------------------------------------------------------
import yaml  # noqa: E402  pylint: disable=wrong-import-position


# -----------------------------------------------------------------------------
# Robust dynamic import helper
# -----------------------------------------------------------------------------

def _lazy_import(name: str) -> ModuleType:
    """Import *name* with multiple fallbacks.

    1. Try as a top-level module (works for editable installs / source checkout).
    2. Try relative to *__package__* (works when ``python -m ...`` is used).
    3. Try relative to the *root* distribution package (works from the installed
       wheel where modules are laid out as ``scarf_experiment.<module>``).
    4. FINAL SAFETY NET: load the *file* directly from ROOT_DIR **or src/** if
       present.
    """
    # (1) plain import first ---------------------------------------------------
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as first_err:  # keep original to re-raise later
        # (2) relative to the current "package" ------------------------------
        if __package__:
            try:
                return importlib.import_module(f"{__package__}.{name}")
            except ModuleNotFoundError:
                pass

        # (3) relative to the *root* package ----------------------------------
        root_pkg = __name__.split(".")[0]  # e.g. "scarf_experiment"
        if root_pkg and root_pkg != name:
            try:
                return importlib.import_module(f"{root_pkg}.{name}")
            except ModuleNotFoundError:
                pass

        # (4) FINAL fallback – direct file import -----------------------------
        for base in (ROOT_DIR, SRC_DIR):
            candidate = base / f"{name}.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(name, candidate)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module  # cache so subsequent imports work
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                    return module

        # Nothing worked – re-raise the *original* error for clarity
        raise first_err


# -----------------------------------------------------------------------------
# Pipeline module loader
# -----------------------------------------------------------------------------

def _load_pipeline_modules() -> None:  # noqa: D401 – imperative mood
    """Dynamically import the pipeline building blocks."""
    global preprocess, train, evaluate  # pylint: disable=global-statement
    preprocess = _lazy_import("preprocess_py").preprocess
    train = _lazy_import("train_py").train
    evaluate = _lazy_import("evaluate_py").evaluate


CONFIG_DIR = ROOT_DIR / "config"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        sys.exit(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCARF experiment runner")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke-test config")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment config")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    args = _parse_args()

    cfg = _load_yaml("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")

    # Import pipeline modules only after CLI args & config were loaded
    _load_pipeline_modules()

    # -------------------- pipeline stages --------------------
    preprocess(cfg)
    train_artifacts = train(cfg)
    evaluate(train_artifacts)


if __name__ == "__main__":
    main()
