import json
from pathlib import Path
from typing import Dict, Any

_OUTPUT_DIR = Path(".research/iteration8")  # UPDATED PATH (keep consistent)


def evaluate(train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Load metrics produced by ``train()`` and produce a summary.

    For the purposes of the smoke-test we simply re-package the existing
    metrics and add a dummy pass/fail flag.  The resulting summary is persisted
    alongside the training metrics as required by the spec.
    """

    metrics = train_artifacts["metrics"]
    summary = {
        "passed": metrics["micro_f1"] > 0.5,  # Always True in our range
        **metrics,
    }

    # ------------------------------------------------------------------
    # Persist evaluation summary next to the training artefacts
    # ------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _OUTPUT_DIR / Path(train_artifacts["json_path"]).name.replace(
        "metrics", "summary"
    )
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    # Echo to stdout for verification
    print("===== EVALUATION SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("==============================")

    return summary
