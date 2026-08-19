from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

FEATURES = {
    "latitude": "observable_current_state",
    "longitude": "observable_current_state",
    "altitude": "observable_current_state",
    "speed": "observable_current_state",
    "heading": "observable_current_state",
    "vertical_rate": "observable_current_state",
    "track": "observable_current_state",
    "timestamp": "observable_current_state",
    "aircraft_id": "identifier",
    "flight_id": "identifier",
    "trajectory_id": "identifier",
    "distance": "derived_current_state",
    "bearing": "derived_current_state",
    "relative_position": "derived_current_state",
    "relative_velocity": "derived_current_state",
    "relative_speed": "derived_current_state",
    "relative_heading": "derived_current_state",
    "tcpa": "future_derived_or_label_proxy",
    "dcpa": "future_derived_or_label_proxy",
    "future_position": "future_information",
    "future_altitude": "future_information",
    "future_speed": "future_information",
    "future_heading": "future_information",
    "conflict": "target",
    "conflict_label": "target",
    "label": "target",
}

PATTERNS = {
    "tcpa": r"\btcpa\b",
    "dcpa": r"\bdcpa\b",
    "future": r"\bfuture\b|\blookahead\b|\bhorizon\b|\bforecast\b",
    "label": r"\blabel\b|\by\b",
    "shift": r"\.shift\s*\(",
    "rolling": r"\.rolling\s*\(",
    "trajectory": r"\btrajectory\b",
}

rows = []

for feature, classification in FEATURES.items():
    hits = []
    pattern = re.compile(rf"\b{re.escape(feature)}\b", re.I)

    for path in sorted(SCRIPTS.glob("*.py")):
        text = path.read_text(errors="ignore")
        if pattern.search(text):
            hits.append(path.name)

    rows.append({
        "feature": feature,
        "classification": classification,
        "source_scripts": hits,
        "allowed_for_final_predictor":
            classification in {
                "observable_current_state",
                "derived_current_state",
            }
            and feature not in {"aircraft_id", "flight_id", "trajectory_id"},
    })

report = {
    "audit": "feature_leakage_audit",
    "status": "PRELIMINARY_SOURCE_SCAN",
    "features": rows,
    "critical_exclusions": ["tcpa", "dcpa", "future_position",
                            "future_altitude", "future_speed",
                            "future_heading", "conflict_label", "label"],
}

out = ROOT / "research_artifacts" / "feature_leakage_audit.json"
out.write_text(json.dumps(report, indent=2))
print(out)
