"""Local inference with the same feature builder used for training."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from noshow.data import FEATURES, features


def predict(model_path: Path, scheduled_day: str, appointment_day: str) -> dict:
    # Load only a trusted artifact created by this project: joblib can execute code.
    bundle = joblib.load(model_path)
    if bundle["features"] != FEATURES:
        raise ValueError("Incompatible model feature contract")
    row = pd.DataFrame([{"ScheduledDay": scheduled_day, "AppointmentDay": appointment_day}])
    probability = float(bundle["pipeline"].predict_proba(features(row))[0, 1])
    return {"estimated_no_show_probability": probability,
            "model": bundle["metadata"]["selected_model"],
            "use": "Educational retrospective model; no external or operational validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--scheduled", required=True)
    parser.add_argument("--appointment", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.scheduled, args.appointment), indent=2))


if __name__ == "__main__":
    main()
