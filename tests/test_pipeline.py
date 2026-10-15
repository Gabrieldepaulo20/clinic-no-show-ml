"""Meaningful checks of prediction-time availability and the inference contract."""
import numpy as np
import pandas as pd
import pytest

from noshow.data import FEATURES, features, load_dataset, temporal_split, top_capacity_metrics
from noshow.predict import predict
from noshow.train import train


def synthetic_frame(n=200):
    booking = pd.date_range("2016-01-01", periods=n, freq="D", tz="UTC")
    # Artificial labels only for tests, never reported as real model performance.
    return pd.DataFrame({"AppointmentID": [str(i) for i in range(n)],
                         "ScheduledDay": booking,
                         "AppointmentDay": pd.date_range("2016-01-04", periods=n, freq="D", tz="UTC"),
                         "No-show": ["Yes" if i % 3 == 0 else "No" for i in range(n)]})


def test_prediction_does_not_read_outcome_or_sensitive_fields():
    raw = synthetic_frame(2)
    a = features(raw)
    raw["No-show"] = ["No", "Yes"]
    raw["PatientId"] = ["private-1", "private-2"]
    raw["SMS_received"] = [1, 0]
    raw["Diabetes"] = [1, 1]
    pd.testing.assert_frame_equal(a, features(raw))
    assert list(a) == FEATURES


def test_same_day_booking_is_valid_despite_midnight_appointment_timestamp():
    row = pd.DataFrame([{"ScheduledDay": "2016-05-01T18:00:00Z", "AppointmentDay": "2016-05-01T00:00:00Z"}])
    assert features(row).iloc[0]["lead_days"] == 0


@pytest.mark.parametrize("appointment", ["2016-04-01", "not-a-date"])
def test_inference_rejects_past_or_invalid_dates(appointment):
    with pytest.raises(ValueError):
        features(pd.DataFrame([{"ScheduledDay": "2016-05-01", "AppointmentDay": appointment}]))


def test_temporal_split_purges_future_outcomes(tmp_path):
    p = tmp_path / "sample.csv"
    synthetic_frame().to_csv(p, index=False)
    frame, _ = load_dataset(p)
    parts, meta = temporal_split(frame.sample(frac=1, random_state=4))
    validation_start = pd.Timestamp(meta["validation_start"])
    test_start = pd.Timestamp(meta["test_start"])
    assert parts["train"]["booking_at"].max() < parts["validation"]["booking_at"].min()
    assert parts["validation"]["booking_at"].max() < parts["test"]["booking_at"].min()
    assert parts["train"]["appointment_at"].max() < validation_start
    assert parts["validation"]["appointment_at"].max() < test_start
    assert meta["purged_train_labels"] == 3
    assert meta["purged_validation_labels"] == 3


def test_unknown_labels_and_duplicate_appointments_fail(tmp_path):
    frame = synthetic_frame()
    frame.loc[0, "No-show"] = "maybe"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outcome"):
        load_dataset(path)
    frame = synthetic_frame()
    frame.loc[1, "AppointmentID"] = frame.loc[0, "AppointmentID"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


def test_fixed_budget_metrics_match_observed_outcomes():
    result = top_capacity_metrics([1, 0, 1, 0, 0], [.9, .8, .7, .2, .1], .4)
    assert result["selected"] == 2
    assert result["precision"] == .5
    assert result["recall"] == .5
    assert result["lift_over_random_expectation"] == 1.25


def test_training_artifact_and_inference_work_together(tmp_path):
    source = tmp_path / "sample.csv"
    synthetic_frame().to_csv(source, index=False)
    result = train(source, tmp_path / "reports")
    assert result["selected_model"] in result["validation"]
    p = predict(tmp_path / "artifacts/model.joblib", "2016-05-01T12:00:00Z", "2016-05-04T00:00:00Z")
    assert np.isfinite(p["estimated_no_show_probability"])
    assert 0 <= p["estimated_no_show_probability"] <= 1
