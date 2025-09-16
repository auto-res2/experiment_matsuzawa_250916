from typing import Dict, Any


def preprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder preprocessing step enforcing *fail-fast* policy.

    If the configuration requests access to a real dataset (key
    ``requires_real_data``) we abort immediately because such datasets are not
    accessible inside the execution sandbox.
    """
    if config.get("requires_real_data"):
        raise RuntimeError(
            "Required real dataset unavailable – aborting experiment per NO-FALLBACK policy"
        )

    # Return an object to show that preprocessing succeeded.
    return {"status": "preprocessed"}
