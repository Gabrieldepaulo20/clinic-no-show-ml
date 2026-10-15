"""Operational-only features and splits that respect outcome availability."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["lead_days", "booking_hour", "booking_weekday", "appointment_weekday", "same_day"]
RAW_COLUMNS = ["AppointmentID", "ScheduledDay", "AppointmentDay", "No-show"]


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only features available when the appointment is booked.

    The source does not document local timezone conversion. Use the UTC dates
    supplied by the dataset consistently; this is a limitation of the study.
    """
    booking = pd.to_datetime(frame["ScheduledDay"], utc=True, errors="coerce")
    appointment = pd.to_datetime(frame["AppointmentDay"], utc=True, errors="coerce")
    if booking.isna().any() or appointment.isna().any():
        raise ValueError("Missing or invalid appointment/booking date")
    lead = (appointment.dt.normalize() - booking.dt.normalize()).dt.days
    if (lead < 0).any():
        raise ValueError("Appointment calendar day cannot precede booking day")
    return pd.DataFrame({
        "lead_days": lead.astype(float),
        "booking_hour": booking.dt.hour.astype(float),
        "booking_weekday": booking.dt.dayofweek.astype(float),
        "appointment_weekday": appointment.dt.dayofweek.astype(float),
        "same_day": (lead == 0).astype(float),
    }, index=frame.index)[FEATURES]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    # Read no patient IDs, demographics, clinical variables or SMS intervention.
    frame = pd.read_csv(path, usecols=RAW_COLUMNS, dtype={"AppointmentID": "string"})
    total = len(frame)
    if frame["AppointmentID"].isna().any() or frame["AppointmentID"].duplicated().any():
        raise ValueError("Missing or duplicate appointment identifier")
    if not frame["No-show"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected outcome; expected Yes (no-show) or No (attended)")
    frame["booking_at"] = pd.to_datetime(frame["ScheduledDay"], utc=True, errors="coerce")
    frame["appointment_at"] = pd.to_datetime(frame["AppointmentDay"], utc=True, errors="coerce")
    valid = frame["booking_at"].notna() & frame["appointment_at"].notna()
    valid &= frame["appointment_at"].dt.normalize() >= frame["booking_at"].dt.normalize()
    clean = frame.loc[valid].copy().sort_values("booking_at", kind="stable").reset_index(drop=True)
    clean["target"] = clean["No-show"].eq("Yes").astype(int)
    return clean, {"source_rows": total, "valid_rows": len(clean), "invalid_date_rows": total - len(clean)}


def temporal_split(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Split booking dates 70/15/15 and purge unobserved training labels.

    A no-show label is assumed available after its appointment calendar day.
    Drop bookings whose appointment has not ended at the next split boundary.
    No selection or fit uses the final test outcomes.
    """
    ordered = frame.sort_values("booking_at", kind="stable")
    if len(ordered) < 20:
        raise ValueError("At least 20 observations are required")
    validation_start = ordered.iloc[int(len(ordered) * .70)]["booking_at"].normalize()
    test_start = ordered.iloc[int(len(ordered) * .85)]["booking_at"].normalize()
    if validation_start >= test_start:
        raise ValueError("Not enough distinct booking days for temporal holdouts")
    train_window = ordered[ordered["booking_at"] < validation_start]
    validation_window = ordered[(ordered["booking_at"] >= validation_start) & (ordered["booking_at"] < test_start)]
    train = train_window[train_window["appointment_at"].dt.normalize() < validation_start].copy()
    validation = validation_window[validation_window["appointment_at"].dt.normalize() < test_start].copy()
    test = ordered[ordered["booking_at"] >= test_start].copy()
    parts = {"train": train, "validation": validation, "test": test}
    if any(part.empty or part["target"].nunique() < 2 for part in parts.values()):
        raise ValueError("Each split must contain observations from both classes")
    meta = {
        "validation_start": validation_start.isoformat(), "test_start": test_start.isoformat(),
        "purged_train_labels": len(train_window) - len(train),
        "purged_validation_labels": len(validation_window) - len(validation),
        "splits": {name: {"rows": len(p), "no_show_rate": float(p["target"].mean()),
                          "booking_min": p["booking_at"].min().isoformat(),
                          "booking_max": p["booking_at"].max().isoformat(),
                          "appointment_max": p["appointment_at"].max().isoformat()} for name, p in parts.items()},
    }
    return parts, meta


def top_capacity_metrics(y, probability, fraction: float = .20) -> dict:
    """Simulate a fixed reminder budget. A score tie uses stable row order.

    This is a retrospective ranking, not an estimate of prevented no-shows.
    """
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not 0 < fraction <= 1 or len(y) != len(probability) or not len(y):
        raise ValueError("Invalid budget or empty/mismatched arrays")
    if not np.isfinite(probability).all():
        raise ValueError("Scores must be finite")
    k = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(-probability, kind="stable")[:k]
    positives = int(y[selected].sum())
    base_rate = float(y.mean())
    precision = positives / k
    return {"budget_fraction": fraction, "selected": k, "captured_no_shows": positives,
            "precision": precision, "recall": positives / int(y.sum()) if y.sum() else 0.,
            "lift_over_random_expectation": precision / base_rate if base_rate else None}
