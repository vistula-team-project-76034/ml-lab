# Lab 4: Data pipeline and quality

Run from the repo root with one command:

```
python lab04/src/lab04.py
```

Outputs go to `lab04/results/`: `pipeline.log`, `validation_report.md` (+ `.csv`), `preprocessor.joblib`,
`label_sample_50.csv`, `config.yaml`, `results.json`. Processed data is written to `lab04/data/processed/`
(not committed because of size, it is fully recreated by the command, and its hash is in `config.yaml`).
Data card: `lab04/data_card.md`.

## 1. Problem definition

The task is to predict if a person earns more than 50K dollars per year, using data from the 1994 US Census
(UCI Adult). It is a binary classification problem.

- **Inputs:** 14 features about a person: 6 numeric (age, fnlwgt, education-num, capital-gain, capital-loss,
  hours-per-week) and 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex,
  native-country).
- **Output:** `income`, 1 if `>50K`, 0 if `<=50K`.
- **Success metrics:** F1 for the positive class and accuracy, measured on a held-out test set. Only 24% of people
  are `>50K`, so a model that always says `<=50K` already has 76% accuracy. F1 shows if the model really finds the
  positive class, so it is the main metric and accuracy is the second one.
- **Constraints:**
  - No data leakage: the test set is split off first, and everything that learns from data (imputer, clip
    bounds, scaler, encoder, model) is fitted only on the training set.
  - Reproducible pipeline: fixed seed, all settings in one config, one command rebuilds everything, and the SHA256
    hash of the processed data proves that two runs give the same result.
  - Data quality problems are reported, not silently removed.

The focus of this lab is the data pipeline, not the model, so a simple LogisticRegression is used as a baseline.

## 2. Ingestion

The data is loaded with `fetch_openml(name='adult', version=2, as_frame=True)`. The raw frame is not changed: all
later steps work on a copy. The only change is renaming the OpenML label column `class` to `income`.

Log: `ingestion: rows=48842 columns=15`. A hash of the raw frame is also logged, so it is possible to check that the
source data did not change between runs.

## 3. Validation

The schema from the manual is written in the code (`SCHEMA`, `RANGES`), plus a list of allowed values for each
categorical column taken from the UCI data dictionary (`adult.names`). Checks:

- type of every column (int columns must contain only whole numbers, OpenML stores them as float64)
- ranges: age 17–90, hours-per-week 1–99, capital-gain 0–99999, capital-loss 0–4356
- missing values: `?` or null count as missing
- categorical values must be from the allowed list (`?` allowed)
- exact duplicate rows

Full table: `results/validation_report.md`. 44 checks, 4 have violations:

| check | column | violations |
|---|---|---|
| missing | workclass | 2,799 (5.73%) |
| missing | occupation | 2,809 (5.75%) |
| missing | native-country | 857 (1.75%) |
| exact duplicates | all columns | 52 |

All types, ranges and categories are correct. No rows were dropped in this step. The problems are only reported
and handled later in the cleaning step on the train set.

## 4. Cleaning and transformation

Order matters here, because doing cleaning before the split is a form of leakage.

1. **Split first:** 70/30 stratified by income, `random_state=42`. Train 34,189 rows, test 14,653 rows. The
   positive rate is 0.2393 in both, so stratification worked.
2. `?` is turned into a real missing value in train and test separately (this rule does not learn anything).
3. **Duplicates:** exact duplicates (features + label) are dropped from train only: 24 rows, train becomes
   34,165. Test is not touched, because test must look like real future data.
4. **Imputation:** median for numeric columns, most frequent value for categorical columns. Both are fitted on
   train only (`SimpleImputer` inside the pipeline).
5. **Outliers:** numeric columns are clipped to the 1st and 99th percentile computed on train. Examples of bounds:
   age 17–74, hours-per-week 8–80, capital-gain 0–15,024, capital-loss 0–1,980. The same bounds are used for test.
   All bounds are saved in `config.yaml`.
6. Numeric columns are scaled with `StandardScaler`, categorical columns are one-hot encoded
   (`handle_unknown='ignore'`, so a category that appears only in test does not crash the pipeline).

All learned steps are one sklearn `ColumnTransformer`, saved with `joblib.dump` to `results/preprocessor.joblib`.
Because it is one object, calling `transform` on new data uses exactly the train statistics.

## 5. Feature engineering

- `capital_net = capital-gain - capital-loss`: one number for the net capital income.
- `hours_per_age = hours-per-week / (age - 16)`: how much a person works compared to how long they could work
  (minimum age is 17, so there is no division by zero).

Both features only use values from the same row, so they can not leak information from other rows. They are
clipped, imputed and scaled with train statistics like the other numeric columns.

LogisticRegression (`max_iter=1000`), same split, same preprocessing:

| features | accuracy | F1 |
|---|---|---|
| 14 original | 0.8540 | 0.6601 |
| 14 + 2 derived | 0.8578 | 0.6696 |

Both metrics go up a little (+0.38 pp accuracy, +0.95 pp F1). It makes sense for a linear model: after clipping,
capital-gain and capital-loss are two separate weak columns, and `capital_net` gives the model one combined signal.
`hours_per_age` adds a non-linear combination of two columns that the linear model can not build by itself.
The gain is small, so the main information was already in the original features.

## 6. Label-quality check

**Duplicate-conflict audit** (on all 48,842 rows, `?` treated as a normal value): 5 groups of rows have identical
14 features but different income labels, 10 rows in total. At least one row in each group has a wrong label, or
the features are not enough to decide the label. This is a very small part of the data (0.02%), so label noise from
this source is low.

**Manual check of 50 random rows** (`results/label_sample_50.csv`, seed 42, 40 `<=50K` and 10 `>50K`). Each row was
checked against the data dictionary: ranges, allowed categories, education vs education-num (for example
HS-grad must be 9), relationship vs sex (Husband must be Male, Wife must be Female), a person can not have both
capital gain and capital loss, and a 17-year-old should not be married or divorced. Result: 0 rows with impossible
values. 2 rows have `?` in workclass and occupation, which is allowed by the null policy.

The same rules on all rows found 4 rows where relationship and sex disagree and 3 rows with a 17-year-old who is
married or divorced. They are reported in `pipeline.log` and the data card, not removed.

Adult has objective labels (from census answers), so the "relabel after 24 hours and compute Cohen's kappa" part
does not apply.

## 7. Hash, config, reproducibility

The processed train and test (after the fitted preprocessor, 107 feature columns + label) are saved as CSV with a
fixed number format, and their SHA256 is computed:

- `train.csv`: `253cc30268f71e65c87aa11abb131c2ce0bea34fca999719277f66f71782b862`
- `test.csv`: `ee20c623b9330724f50ec06c25618be45c1e249ea66752de6719428be3e803b7`

The hashes are stored in `results/config.yaml` together with the config (seed, split, clip percentiles), the schema,
the clip bounds learned from train and the library versions. I ran the command twice and got the same hashes, so the
pipeline is deterministic. With other versions of pandas / scikit-learn the hash can be different, that is why the
versions are saved next to it.
