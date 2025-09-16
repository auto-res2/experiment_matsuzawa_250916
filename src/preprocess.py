from typing import Dict, Any


def preprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    """No-op placeholder that would normally download and prepare datasets.

    A *fail-fast* policy is enforced: if the configuration requests a real
    dataset (key `requires_real_data`), we abort immediately because such
    datasets are not accessible inside the execution sandbox.
    """
    if config.get("requires_real_data"):
        raise RuntimeError(
            "Required real dataset unavailable – aborting experiment per NO-FALLBACK policy"
        )

    # Return an object to show that preprocessing succeeded.
    return {"status": "preprocessed"}
