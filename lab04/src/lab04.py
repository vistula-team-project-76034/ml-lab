import hashlib
import json
import logging
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CONFIG = {
    "dataset": {"name": "adult", "version": 2},
    "seed": 42,
    "test_size": 0.3,
    "clip_percentiles": [1, 99],
    "positive_label": ">50K",
    "label_sample_size": 50,
}

RESULTS = Path("lab04/results")
PROCESSED = Path("lab04/data/processed")
RESULTS.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(RESULTS / "pipeline.log", mode="w"), logging.StreamHandler()],
)
log = logging.info

SCHEMA = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", "income": "category",
}
RANGES = {"age": (17, 90), "hours-per-week": (1, 99), "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
ALLOWED = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                  "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                   "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                       "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                       "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                       "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "income": [">50K", "<=50K"],
}
EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7, "12th": 8,
           "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
           "Prof-school": 15, "Doctorate": 16}
NUM_COLS = [c for c, t in SCHEMA.items() if t == "int"]
CAT_COLS = [c for c, t in SCHEMA.items() if t == "category" and c != "income"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_missing(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("object").where(df[c].notna(), "?").str.strip()
        df[c] = df[c].replace("?", np.nan)
    return df


def add_features(df):
    df = df.copy()
    df["capital_net"] = df["capital-gain"] - df["capital-loss"]
    df["hours_per_age"] = df["hours-per-week"] / (df["age"] - 16)
    return df


def build_preprocessor(X_train, num_cols):
    lo, hi = np.nanpercentile(X_train[num_cols].to_numpy(float), CONFIG["clip_percentiles"], axis=0)
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": lo, "a_max": hi}, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, CAT_COLS)])
    bounds = {c: [float(a), float(b)] for c, a, b in zip(num_cols, lo, hi)}
    return pre, bounds


def evaluate(X_train, y_train, X_test, y_test, num_cols):
    pre, bounds = build_preprocessor(X_train, num_cols)
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, bounds, {"accuracy": round(accuracy_score(y_test, pred), 4), "f1": round(f1_score(y_test, pred), 4)}


# step 2
data = fetch_openml(name=CONFIG["dataset"]["name"], version=CONFIG["dataset"]["version"], as_frame=True)
raw = data.frame.rename(columns={"class": "income"})
log(f"ingestion: rows={raw.shape[0]} columns={raw.shape[1]}")
raw_hash = hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
log(f"raw data hash: {raw_hash}")
df = raw.copy()

# step 3
issues = []
for col, t in SCHEMA.items():
    if col not in df.columns:
        issues.append({"check": "column", "column": col, "violations": 1, "detail": "column missing"})
        continue
    s = df[col]
    if t == "int":
        ok = pd.api.types.is_numeric_dtype(s) and bool((s.dropna() % 1 == 0).all())
        issues.append({"check": "type int", "column": col, "violations": 0 if ok else 1,
                       "detail": f"{s.dtype}, whole numbers" if ok else f"{s.dtype}, not whole numbers"})
    else:
        ok = isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s)
        issues.append({"check": "type category", "column": col, "violations": 0 if ok else 1, "detail": str(s.dtype)})
for col, (lo, hi) in RANGES.items():
    bad = int(((df[col] < lo) | (df[col] > hi)).sum())
    issues.append({"check": f"range {lo}-{hi}", "column": col, "violations": bad,
                   "detail": f"observed {df[col].min()}-{df[col].max()}"})
for col in SCHEMA:
    s = df[col]
    n_missing = int(s.isna().sum() + (s.astype("object").astype(str).str.strip() == "?").sum())
    issues.append({"check": "missing ('?' or null)", "column": col, "violations": n_missing,
                   "detail": f"{100 * n_missing / len(df):.2f}%"})
for col, values in ALLOWED.items():
    vals = df[col].dropna().astype(str).str.strip().str.rstrip(".")
    unknown = sorted(set(vals) - set(values) - {"?"})
    issues.append({"check": "allowed categories", "column": col, "violations": int(vals.isin(unknown).sum()),
                   "detail": ", ".join(unknown) if unknown else "ok"})
dup = int(df.duplicated().sum())
issues.append({"check": "exact duplicate rows", "column": "all", "violations": dup, "detail": "reported, not dropped"})
report = pd.DataFrame(issues)
report.to_csv(RESULTS / "validation_report.csv", index=False)
with open(RESULTS / "validation_report.md", "w") as f:
    f.write(f"# Validation report\n\nRows: {len(df)}, columns: {df.shape[1]}. No rows were dropped.\n\n")
    f.write("| check | column | violations | detail |\n|---|---|---|---|\n")
    for r in issues:
        f.write(f"| {r['check']} | {r['column']} | {r['violations']} | {r['detail']} |\n")
log(f"validation: {len(report)} checks, {int((report.violations > 0).sum())} with violations, rows kept={len(df)}")

# step 4
df[NUM_COLS] = df[NUM_COLS].astype("int64")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
y = (df["income"] == CONFIG["positive_label"]).astype(int)
X = df.drop(columns="income")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["seed"])
log(f"split: train={len(X_train)} test={len(X_test)} "
    f"positive rate train={y_train.mean():.4f} test={y_test.mean():.4f}")

X_train, X_test = to_missing(X_train), to_missing(X_test)
train_full = pd.concat([X_train, y_train.rename("income")], axis=1)
n_before = len(train_full)
train_full = train_full.drop_duplicates()
X_train, y_train = train_full.drop(columns="income"), train_full["income"]
log(f"duplicates dropped from train: {n_before - len(train_full)} (test untouched, {len(X_test)} rows)")

# step 5
base_model, base_bounds, base_scores = evaluate(X_train, y_train, X_test, y_test, NUM_COLS)
log(f"baseline (14 features): {base_scores}")
FE_COLS = NUM_COLS + ["capital_net", "hours_per_age"]
Xf_train, Xf_test = add_features(X_train), add_features(X_test)
fe_model, fe_bounds, fe_scores = evaluate(Xf_train, y_train, Xf_test, y_test, FE_COLS)
log(f"with derived features: {fe_scores}")

preprocessor = fe_model.named_steps["pre"]
joblib.dump(preprocessor, RESULTS / "preprocessor.joblib")
log("saved lab04/results/preprocessor.joblib")

# step 6
feat = [c for c in df.columns if c != "income"]
audit = to_missing(df).fillna("?")
conflicts = audit.groupby(feat)["income"].nunique()
conflict_groups = conflicts[conflicts > 1]
conflict_rows = int(audit.set_index(feat).index.isin(conflict_groups.index).sum())
log(f"duplicate-conflict audit: {len(conflict_groups)} feature groups with conflicting labels, {conflict_rows} rows")

sample = to_missing(df.sample(CONFIG["label_sample_size"], random_state=CONFIG["seed"]))


def row_problems(r):
    p = []
    for col, (lo, hi) in RANGES.items():
        if not lo <= r[col] <= hi:
            p.append(f"{col} out of range")
    for col, values in ALLOWED.items():
        if pd.notna(r[col]) and r[col] not in values:
            p.append(f"{col} unknown value")
    if pd.notna(r["education"]) and EDU_NUM.get(r["education"]) != r["education-num"]:
        p.append("education and education-num disagree")
    if (r["relationship"] == "Husband" and r["sex"] != "Male") or (r["relationship"] == "Wife" and r["sex"] != "Female"):
        p.append("relationship and sex disagree")
    if r["capital-gain"] > 0 and r["capital-loss"] > 0:
        p.append("both capital-gain and capital-loss")
    if r["age"] < 18 and r["marital-status"] in ("Married-civ-spouse", "Widowed", "Divorced"):
        p.append("age vs marital-status")
    return "; ".join(p)


sample["issues"] = sample.apply(row_problems, axis=1)
sample.to_csv(RESULTS / "label_sample_50.csv")
n_bad = int((sample["issues"] != "").sum())
log(f"50-row manual check: {n_bad} rows with impossible values")

full_issues = to_missing(df).apply(row_problems, axis=1)
issue_counts = full_issues[full_issues != ""].str.split("; ").explode().value_counts().to_dict()
log(f"same checks on all rows: {issue_counts}")

# step 7
names = list(preprocessor.get_feature_names_out())
train_out = pd.DataFrame(preprocessor.transform(Xf_train), columns=names)
train_out["income"] = y_train.to_numpy()
test_out = pd.DataFrame(preprocessor.transform(Xf_test), columns=names)
test_out["income"] = y_test.to_numpy()
train_out.to_csv(PROCESSED / "train.csv", index=False, float_format="%.6f")
test_out.to_csv(PROCESSED / "test.csv", index=False, float_format="%.6f")
hashes = {"train.csv": sha256_file(PROCESSED / "train.csv"), "test.csv": sha256_file(PROCESSED / "test.csv")}
log(f"processed sha256: {hashes}")

run_config = {
    "config": CONFIG,
    "schema": SCHEMA,
    "ranges": {k: list(v) for k, v in RANGES.items()},
    "clip_bounds_from_train": fe_bounds,
    "raw_data_hash": raw_hash,
    "processed_sha256": hashes,
    "processed_shape": {"train": list(train_out.shape), "test": list(test_out.shape)},
    "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                 "scikit-learn": sklearn.__version__},
}
with open(RESULTS / "config.yaml", "w") as f:
    yaml.safe_dump(run_config, f, sort_keys=False)

results = {
    "rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
    "validation_checks_with_violations": report[report.violations > 0][["check", "column", "violations"]]
    .to_dict("records"),
    "train_rows_after_dedup": int(len(X_train)), "test_rows": int(len(X_test)),
    "duplicates_dropped_train": int(n_before - len(X_train)),
    "baseline": base_scores, "with_features": fe_scores,
    "conflict_groups": int(len(conflict_groups)), "conflict_rows": conflict_rows,
    "sample_rows_with_issues": n_bad, "all_rows_issue_counts": issue_counts,
    "processed_sha256": hashes,
}
with open(RESULTS / "results.json", "w") as f:
    json.dump(results, f, indent=2)
log("Done")
