import json
from pathlib import Path
from typing import Dict, Any


def evaluate(train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Load metrics produced by `train()` and produce a summary.

    For the purposes of the smoke-test we simply re-package the existing
    metrics and add a dummy pass/fail flag.
    """
    metrics = train_artifacts["metrics"]
    summary = {
        "passed": metrics["micro_f1"] > 0.5,  # Always True in our range
        **metrics,
    }

    # Echo to stdout for verification
    print("===== EVALUATION SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("==============================")

    return summary
