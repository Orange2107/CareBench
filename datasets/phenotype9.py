"""Phenotype-9 label selection utilities."""

PHENOTYPE9_LABELS = [
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Other liver diseases",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Essential hypertension",
    "Acute myocardial infarction",
    "Conduction disorders",
    "Acute cerebrovascular disease",
    "Gastrointestinal hemorrhage",
]

PHENOTYPE9_ABBREVIATIONS = [
    "CHF",
    "CorAth/HD",
    "Liver Dis.",
    "COPD",
    "HTN",
    "AMI",
    "CD",
    "A. CVD",
    "GIB",
]


def select_phenotype9_labels(all_phenotype_cols):
    """Return the configured Phenotype-9 labels and their indices in the source table."""
    missing = [label for label in PHENOTYPE9_LABELS if label not in all_phenotype_cols]
    if missing:
        raise ValueError(
            "Phenotype-9 labels missing from source columns: "
            + ", ".join(missing)
        )

    indices = [all_phenotype_cols.index(label) for label in PHENOTYPE9_LABELS]
    return PHENOTYPE9_LABELS.copy(), indices
