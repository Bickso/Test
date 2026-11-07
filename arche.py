"""Comparaison exploratoire de modèles de réussite sur ARCHE."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
LOG_COLUMNS = ["heure", "pseudo", "contexte", "composant", "evenement"]
COMPONENTS = {
    "systeme": "Système",
    "fichier": "Fichier",
    "test": "Test",
    "forum": "Forum",
}
FEATURES = [
    "n_systeme", "n_fichier", "n_test", "n_forum", "n_autre",
    "jours_actifs", "semaines_actives", "ressources_distinctes",
    "evenements_par_jour_actif", "recence_jours",
]
WARNINGS = [
    "Analyse exploratoire : seuil, périmètre du cours et date d'examen à confirmer.",
    "Sans coupure antérieure à l'examen, les performances ne valident pas une alerte précoce.",
    "Les deux parties du cours sont regroupées : pas de validation par cours distinct.",
    "Une consultation de test n'est pas une mesure de maîtrise.",
    "Petit échantillon d'une seule cohorte : pas de preuve de généralisation ni de causalité.",
]


def read_data(data_dir):
    """Valider les CSV sans modifier les sources ni inventer de notes."""
    data_dir = Path(data_dir)
    logs = pd.read_csv(data_dir / "logs.csv", dtype={"pseudo": "string"})
    notes = pd.read_csv(data_dir / "notes.csv", dtype={"pseudo": "string"})
    for frame, required, name in (
        (logs, LOG_COLUMNS, "logs.csv"),
        (notes, ["pseudo", "note"], "notes.csv"),
    ):
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{name} : colonnes requises {required}")
        if frame[required].isna().any().any():
            raise ValueError(f"{name} : valeurs manquantes dans les colonnes requises")
        for col in set(required) - {"heure", "note"}:
            frame[col] = frame[col].astype("string").str.strip()
            if frame[col].eq("").any():
                raise ValueError(f"{name} : valeur vide dans {col}")
    if notes["pseudo"].duplicated().any():
        raise ValueError("notes.csv : plusieurs notes pour un même pseudo")
    notes["note"] = pd.to_numeric(notes["note"], errors="raise")
    if not np.isfinite(notes["note"]).all() or not notes["note"].between(0, 20).all():
        raise ValueError("Les notes doivent être finies et comprises entre 0 et 20")
    logs["heure"] = pd.to_datetime(logs["heure"], format="ISO8601", errors="raise")
    if logs["heure"].isna().any() or logs["heure"].dt.tz is not None:
        raise ValueError("Les dates doivent être renseignées, sans fuseau horaire")
    audit = {
        "raw_events": len(logs),
        "duplicate_events_removed": int(logs.duplicated(LOG_COLUMNS).sum()),
        "log_learners": int(logs["pseudo"].nunique()),
        "graded_learners": len(notes),
        "graded_without_any_trace": sorted(set(notes["pseudo"]) - set(logs["pseudo"])),
        "trace_without_grade": sorted(set(logs["pseudo"]) - set(notes["pseudo"])),
        "first_trace": logs["heure"].min().isoformat(),
        "last_trace": logs["heure"].max().isoformat(),
    }
    return logs.drop_duplicates(LOG_COLUMNS).copy(), notes, audit


def prepare_features(logs, cutoff=None):
    """Agrégation déterministe, sans note et sans apprentissage de statistiques."""
    reference = pd.Timestamp(cutoff) if cutoff is not None else logs["heure"].max()
    if pd.isna(reference) or reference.tzinfo is not None:
        raise ValueError("La coupure doit être une date valide sans fuseau horaire")
    # Seuls les événements strictement antérieurs à la coupure sont disponibles.
    observed = logs.loc[
        logs["heure"].lt(reference) if cutoff is not None
        else logs["heure"].le(reference)
    ].copy()
    # Garder le fait qu'un apprenant est observé, même si ses seuls événements
    # sont ensuite exclus parce qu'ils concernent directement une évaluation.
    observed_ids = sorted(observed["pseudo"].unique())
    evaluation_event = observed["evenement"].str.contains(
        r"évalu|Rapport", case=False, regex=True, na=False
    )
    usable = observed.loc[~evaluation_event].copy()
    features = pd.DataFrame(0.0, index=observed_ids, columns=FEATURES)
    features.index.name = "pseudo"
    # Pour un apprenant observé mais sans événement utilisable, la récence
    # reste manquante : son historique d'activité ne doit pas être inventé.
    features["recence_jours"] = np.nan
    if not usable.empty:
        usable["jour"] = usable["heure"].dt.normalize()
        usable["semaine"] = usable["heure"].dt.to_period("W")
        grouped = usable.groupby("pseudo", sort=True)
        for feature, component in COMPONENTS.items():
            counts = usable.loc[usable["composant"].eq(component)].groupby("pseudo").size()
            features[f"n_{feature}"] = counts.reindex(features.index, fill_value=0)
        other = usable.loc[~usable["composant"].isin(COMPONENTS.values())]
        features["n_autre"] = other.groupby("pseudo").size().reindex(
            features.index, fill_value=0
        )
        features["jours_actifs"] = grouped["jour"].nunique().reindex(features.index, fill_value=0)
        features["semaines_actives"] = grouped["semaine"].nunique().reindex(
            features.index, fill_value=0
        )
        features["ressources_distinctes"] = grouped["contexte"].nunique().reindex(
            features.index, fill_value=0
        )
        features["evenements_par_jour_actif"] = (
            grouped.size() / grouped["jour"].nunique()
        ).reindex(features.index, fill_value=0)
        features["recence_jours"] = (
            (reference - grouped["heure"].max()).dt.total_seconds() / 86400
        ).reindex(features.index)
    info = {
        "reference_time": reference.isoformat(),
        "cutoff_provided": cutoff is not None,
        "observed_events": len(observed),
        "evaluation_events_excluded": int(evaluation_event.sum()),
        "usable_events": len(usable),
        "observed_learners": len(observed_ids),
    }
    return features, info


def make_dataset(features, notes):
    # L'absence de traces n'est pas assimilée à une absence réelle d'activité.
    joined = notes.set_index("pseudo").join(features, how="inner").sort_index()
    # Sans récence, aucun événement utilisable n'a été observé.
    return joined.dropna(subset=FEATURES)


def make_models(seed=42):
    return {
        "moyenne": DummyRegressor(strategy="mean"),
        "lineaire": make_pipeline(StandardScaler(), LinearRegression()),
        "foret": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            max_features=1.0, random_state=seed, n_jobs=-1,
        ),
    }


def make_splits(labels, folds, repeats, seed):
    if folds < 2 or repeats < 1:
        raise ValueError("Au moins deux plis et une répétition sont nécessaires")
    counts = pd.Series(labels).value_counts()
    if len(counts) != 2 or counts.min() < folds:
        raise ValueError("Chaque classe doit contenir au moins autant d'apprenants que de plis")
    splitter = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def score_predictions(true_grade, predicted_grade, true_pass, predicted_pass):
    scores = {
        "balanced_accuracy": float(balanced_accuracy_score(true_pass, predicted_pass)),
        "rappel_echec": float(recall_score(true_pass, predicted_pass, pos_label=0, zero_division=0)),
        "precision_echec": float(precision_score(true_pass, predicted_pass, pos_label=0, zero_division=0)),
        "f1_echec": float(f1_score(true_pass, predicted_pass, pos_label=0, zero_division=0)),
    }
    if predicted_grade is not None:
        scores["mae"] = float(mean_absolute_error(true_grade, predicted_grade))
        scores["rmse"] = float(root_mean_squared_error(true_grade, predicted_grade))
    return scores


def evaluate(dataset, threshold=10, folds=5, repeats=5, seed=42):
    x, y = dataset[FEATURES], dataset["note"]
    labels = y.ge(threshold).astype(int).to_numpy()
    splits = make_splits(labels, folds, repeats, seed)
    models = make_models(seed)
    records = []
    for split_number, (train, test) in enumerate(splits):
        # Même découpage pour tous les modèles ; jamais deux lignes par pseudo.
        majority = int(labels[train].mean() > 0.5)
        for name, estimator in models.items():
            fitted = clone(estimator).fit(x.iloc[train], y.iloc[train])
            predicted = fitted.predict(x.iloc[test])
            for position, grade in zip(test, predicted):
                records.append({
                    "repeat": split_number // folds, "fold": split_number % folds,
                    "model": name, "pseudo": str(dataset.index[position]),
                    "note": float(y.iloc[position]), "prediction_note": float(grade),
                    "reussite": int(labels[position]),
                    "prediction_reussite": int(grade >= threshold),
                })
        for position in test:
            records.append({
                "repeat": split_number // folds, "fold": split_number % folds,
                "model": "majoritaire", "pseudo": str(dataset.index[position]),
                "note": float(y.iloc[position]), "prediction_note": np.nan,
                "reussite": int(labels[position]), "prediction_reussite": majority,
            })
    predictions = pd.DataFrame(records)
    rows = []
    matrices = {}
    # Chaque répétition fournit une prédiction hors entraînement par apprenant.
    # Ne pas présenter N * répétitions comme un effectif indépendant.
    for (name, repeat), part in predictions.groupby(["model", "repeat"], sort=True):
        row = {"model": name, "repeat": int(repeat)}
        row.update(score_predictions(
            part["note"], None if name == "majoritaire" else part["prediction_note"],
            part["reussite"], part["prediction_reussite"],
        ))
        rows.append(row)
        matrices.setdefault(name, []).append(
            confusion_matrix(part["reussite"], part["prediction_reussite"], labels=[0, 1]).tolist()
        )
    metrics = pd.DataFrame(rows)
    summary = {}
    for name, part in metrics.groupby("model", sort=True):
        summary[name] = {}
        for column in metrics.columns.difference(["model", "repeat"]):
            values = part[column].dropna()
            if not values.empty:
                summary[name][column] = {
                    "mean": float(values.mean()),
                    "std_between_repeats": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()), "max": float(values.max()),
                }
    linear = metrics.loc[metrics.model.eq("lineaire")].set_index("repeat")
    forest = metrics.loc[metrics.model.eq("foret")].set_index("repeat")
    deltas = forest.drop(columns="model") - linear.drop(columns="model")
    paired = {
        column: {
            "mean_forest_minus_linear": float(deltas[column].mean()),
            "min": float(deltas[column].min()), "max": float(deltas[column].max()),
        } for column in deltas.columns
    }
    return predictions, metrics, summary, matrices, paired


def recommendation(summary):
    """Règle descriptive prudente ; pas de test de significativité."""
    linear, forest = summary["lineaire"], summary["foret"]
    best = max(linear["balanced_accuracy"]["mean"], forest["balanced_accuracy"]["mean"])
    baseline = summary["majoritaire"]["balanced_accuracy"]["mean"]
    if best <= baseline + 0.02:
        return "Aucun modèle suffisamment convaincant par rapport à la référence majoritaire."
    if (
        forest["balanced_accuracy"]["min"] > linear["balanced_accuracy"]["max"]
        and forest["balanced_accuracy"]["mean"] > linear["balanced_accuracy"]["mean"] + 0.02
        and forest["rappel_echec"]["mean"] >= linear["rappel_echec"]["mean"] - 0.05
        and forest["mae"]["mean"] <= linear["mae"]["mean"] + 0.1
    ):
        return "Forêt candidate à une validation externe ; avantage descriptif à confirmer."
    return (
        "Conserver provisoirement la régression comme référence simple, sans déploiement : "
        "avantage de la forêt insuffisamment net ou compromis défavorable. "
        "Vérifier aussi la MAE face à la moyenne et valider sur une autre cohorte."
    )


def save_plots(predictions, summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(
        axes, ["balanced_accuracy", "rappel_echec"],
        ["Balanced accuracy", "Rappel des échecs"],
    ):
        ax.bar(names, [summary[n][metric]["mean"] for n in names])
        ax.set_ylim(0, 1)
        ax.set_title(title + " (moyenne des répétitions)")
    fig.tight_layout()
    fig.savefig(output / "classification.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Une seule répétition pour ne pas multiplier artificiellement les points.
    for ax, name in zip(axes, ["lineaire", "foret"]):
        part = predictions.loc[predictions.model.eq(name) & predictions.repeat.eq(0)]
        ax.scatter(part.note, part.prediction_note, alpha=0.7)
        low = min(part.note.min(), part.prediction_note.min())
        high = max(part.note.max(), part.prediction_note.max())
        ax.plot([low, high], [low, high], "--", color="gray")
        ax.set(xlabel="Note réelle", ylabel="Note prédite", title=name)
    fig.tight_layout()
    fig.savefig(output / "notes_predites.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ["lineaire", "foret"]):
        part = predictions.loc[predictions.model.eq(name) & predictions.repeat.eq(0)]
        matrix = confusion_matrix(part.reussite, part.prediction_reussite, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Échec", "Réussite"],
               yticklabels=["Échec", "Réussite"], xlabel="Prédiction",
               ylabel="Réalité", title=name + " — répétition 1")
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)


def load_analysis(args):
    logs, notes, audit = read_data(args.data_dir)
    features, feature_info = prepare_features(logs, args.cutoff)
    dataset = make_dataset(features, notes)
    audit.update(feature_info)
    audit.update({
        "included_learners": len(dataset),
        "excluded_graded_pseudos": sorted(set(notes["pseudo"]) - set(dataset.index)),
        "pass_count": int(dataset["note"].ge(args.threshold).sum()),
        "fail_count": int(dataset["note"].lt(args.threshold).sum()),
    })
    return features, dataset, audit


def compare(args):
    _, dataset, audit = load_analysis(args)
    predictions, metrics, summary, matrices, paired = evaluate(
        dataset, args.threshold, args.folds, args.repeats, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "exploratoire",
        "warnings": WARNINGS,
        "configuration": {
            "threshold": args.threshold, "cutoff": args.cutoff,
            "folds": args.folds, "repeats": args.repeats, "seed": args.seed,
            "hyperparameters": "fixés a priori ; aucun réglage sur les plis externes",
            "forest": {
                "n_estimators": 200, "max_depth": 4,
                "min_samples_leaf": 5, "max_features": 1.0,
            },
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
        },
        "source_sha256": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in ["logs.csv", "notes.csv"]
        },
        "audit": audit, "features": FEATURES, "metrics": summary,
        "confusion_labels": ["échec", "réussite"],
        "confusion_convention": "lignes réelles, colonnes prédites ; une matrice par répétition",
        "confusion_by_repeat": matrices, "paired_differences": paired,
        "uncertainty": "Dispersion descriptive ; répétitions corrélées, pas un intervalle de confiance.",
        "recommendation": recommendation(summary),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    dataset.to_csv(args.output / "features.csv")
    predictions.to_csv(args.output / "predictions_oof.csv", index=False)
    metrics.to_csv(args.output / "metrics_by_repeat.csv", index=False)
    save_plots(predictions, summary, args.output)
    print(f"Mode exploratoire | {len(dataset)} apprenants | seuil {args.threshold:g}")
    print(metrics.groupby("model").mean(numeric_only=True).drop(columns="repeat").round(4).to_string())
    print(report["recommendation"])
    print(f"Résultats : {args.output}")
    for warning in WARNINGS:
        print("ATTENTION :", warning)


def predict_learner(features, dataset, pseudo, model_name, seed=42):
    pseudo = str(pseudo)
    if pseudo not in features.index or features.loc[pseudo, FEATURES].isna().any():
        raise ValueError("Pseudo sans traces utilisables avant la coupure : prédiction refusée")
    # Si sa note existe, ne jamais l'utiliser pour prédire ce même apprenant.
    training = dataset.drop(index=pseudo, errors="ignore")
    if len(training) < 10:
        raise ValueError("Au moins dix autres apprenants notés sont nécessaires")
    model = make_models(seed)[model_name]
    model.fit(training[FEATURES], training["note"])
    grade = float(model.predict(features.loc[[pseudo], FEATURES])[0])
    return grade, training


def predict(args):
    features, dataset, audit = load_analysis(args)
    grade, training = predict_learner(features, dataset, args.pseudo, args.model, args.seed)
    result = {
        "pseudo": args.pseudo, "model": args.model, "mode": "exploratoire",
        "predicted_grade_unclipped": grade, "threshold": args.threshold,
        "predicted_outcome": "réussite" if grade >= args.threshold else "échec",
        "training_learners": len(training), "target_excluded_from_training": True,
        "reference_time": audit["reference_time"],
        "warnings": WARNINGS + [
            "La note prédite n'est pas une probabilité et ne doit pas servir seule à une décision.",
            "Les régressions peuvent produire des notes hors de [0, 20] ; elles ne sont pas tronquées.",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["compare", "predict"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--data-dir", type=Path, default=ROOT)
        sub.add_argument("--threshold", type=float, default=10)
        sub.add_argument("--cutoff", help="Date ISO ; seules les traces strictement antérieures sont incluses")
        sub.add_argument("--seed", type=int, default=42)
        if command == "compare":
            sub.add_argument("--output", type=Path, default=ROOT / "outputs")
            sub.add_argument("--folds", type=int, default=5)
            sub.add_argument("--repeats", type=int, default=5)
        else:
            sub.add_argument("--pseudo", required=True)
            sub.add_argument("--model", choices=["lineaire", "foret"], default="lineaire")
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or not 0 < args.threshold < 20:
        parser.error("Le seuil doit être fini et strictement compris entre 0 et 20")
    try:
        (compare if args.command == "compare" else predict)(args)
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Erreur : {exc}\n")


if __name__ == "__main__":
    main()
