"""Reproducible baseline comparison; choose on validation, evaluate once on test."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshow.data import FEATURES, features, load_dataset, temporal_split, top_capacity_metrics


def candidates() -> dict:
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), ["lead_days", "booking_hour"]),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["booking_weekday", "appointment_weekday", "same_day"]),
    ])
    return {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([("preprocess", preprocessing),
                                         ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.,
            early_stopping=False, random_state=42),
    }


def score(y, p) -> dict:
    pred = (p >= .5).astype(int)
    return {"average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)), "accuracy_at_05": float(accuracy_score(y, pred)),
            "precision_at_05": float(precision_score(y, pred, zero_division=0)),
            "recall_at_05": float(recall_score(y, pred, zero_division=0))}


def train(csv_path: Path, output: Path) -> dict:
    frame, cleaning = load_dataset(csv_path)
    parts, split_info = temporal_split(frame)
    x = {k: features(v) for k, v in parts.items()}
    y = {k: v["target"].to_numpy() for k, v in parts.items()}
    fitted, validation = {}, {}
    for name, model in candidates().items():
        model.fit(x["train"], y["train"])
        fitted[name] = model
        validation[name] = score(y["validation"], model.predict_proba(x["validation"])[:, 1])
    # Baseline is eligible: do not force promotion of a weaker model.
    selected = max(validation, key=lambda n: validation[n]["average_precision"])
    test_results = {name: score(y["test"], model.predict_proba(x["test"])[:, 1])
                    for name, model in fitted.items()}
    selected_p = fitted[selected].predict_proba(x["test"])[:, 1]
    result = {
        "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "seed": 42, "target": "1 = appointment no-show", "features": FEATURES,
        "cleaning": cleaning, "temporal_evaluation": split_info,
        "selection_metric": "validation average precision", "selected_model": selected,
        "validation": validation, "test": test_results,
        "test_reminder_simulation": top_capacity_metrics(y["test"], selected_p),
        "limitations": ["Historical 2016 sample; no external clinic validation",
                        "Only calendar features; demographics, patient IDs, clinical flags and SMS excluded",
                        "No causal estimate of reminder impact or reduced no-shows",
                        "UTC calendar interpretation; local timezone mapping not verified",
                        "Existing data may include repeated patients across dates; no new-patient holdout",
                        "Scores not externally calibrated; retrospective educational use only"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    artifact_dir = output.parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump({"pipeline": fitted[selected], "features": FEATURES, "metadata": result},
                artifact_dir / "model.joblib")
    write_report(result, output / "RESULTS.md")
    return result


def write_report(result: dict, destination: Path) -> None:
    lines = ["# Resultados do primeiro experimento", "",
             "Experimento retrospectivo com dados públicos de 2016. Resultados calculados pelo pipeline; sem validação operacional em clínicas.", "",
             f"Modelo escolhido por average precision na validação: **{result['selected_model']}**.", "",
             "| Modelo | AP validação | AP teste | ROC AUC teste | Brier teste | Acurácia a 0,5 | Recall a 0,5 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in result["test"].items():
        lines.append(f"| {name} | {result['validation'][name]['average_precision']:.4f} | "
                     f"{metrics['average_precision']:.4f} | {metrics['roc_auc']:.4f} | {metrics['brier']:.4f} | "
                     f"{metrics['accuracy_at_05']:.4f} | {metrics['recall_at_05']:.4f} |")
    sim = result["test_reminder_simulation"]
    lines += ["", "Acurácia usa o limiar ilustrativo 0,5. Escolher AP na validação prioriza avaliação de ranking; não implica superioridade em todas as métricas.", "",
              "## Simulação de capacidade", "",
              f"Entre os {sim['selected']} agendamentos de maior score (20% do teste), "
              f"{sim['precision']:.1%} foram faltas observadas, capturando {sim['recall']:.1%} das faltas do teste. "
              f"Lift retrospectivo sobre seleção aleatória esperada: {sim['lift_over_random_expectation']:.2f}x.", "",
              "Essa simulação não mede consultas recuperadas, ganho financeiro ou efeito de lembretes. Empates no score usam ordem estável; a comparação aleatória é expectativa analítica, não um teste operacional.", "",
              "## Divisão temporal", "",
              "| Conjunto | Registros | Taxa de faltas |", "| --- | ---: | ---: |"]
    for name, info in result["temporal_evaluation"]["splits"].items():
        lines.append(f"| {name} | {info['rows']} | {info['no_show_rate']:.1%} |")
    temporal = result["temporal_evaluation"]
    lines += ["", f"Início da validação: {temporal['validation_start']}; início do teste: {temporal['test_start']}.", "",
              f"Removidos do treino: {temporal['purged_train_labels']} resultados ainda indisponíveis no início da validação; "
              f"removidos da validação: {temporal['purged_validation_labels']} resultados ainda indisponíveis no início do teste.", "",
              "## Limites", ""] + [f"- {s}" for s in result["limitations"]]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/appointments.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(json.dumps({"selected_model": result["selected_model"],
                      "test": result["test"][result["selected_model"]]}, indent=2))


if __name__ == "__main__":
    main()
