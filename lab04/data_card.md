# Data card: UCI Adult (processed for Lab 4)

## Source

- UCI Machine Learning Repository, "Adult" (also called "Census Income").
- Extracted by Barry Becker from the 1994 US Census database, donated by Ronny Kohavi and Barry Becker in 1996.
- Loaded with `sklearn.datasets.fetch_openml(name="adult", version=2, as_frame=True)` (OpenML copy, train and test files of the original are merged).

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). It can be shared and adapted with credit to the original authors.

## Size

- Raw: 48,842 rows, 15 columns (14 features + label `income`).
- 6 numeric features: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
- 8 categorical features: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
- Label: `income` (`<=50K` 76.1%, `>50K` 23.9%).
- Missing values (`?`): workclass 2,799, occupation 2,809, native-country 857. 52 exact duplicate rows.
- Processed: train 34,165 rows (after removing 24 duplicates), test 14,653 rows, 108 columns
  (8 scaled numeric, 99 one-hot, 1 label). SHA256 of the processed files is in `results/config.yaml`.

## Processing

Stratified 70/30 split (seed 42) before any cleaning. Imputation (median / mode), clipping to the 1st/99th
percentile, scaling and one-hot encoding are all fitted on train only and saved in `results/preprocessor.joblib`.
Two derived features: `capital_net`, `hours_per_age`. Duplicates removed from train only, test is not changed.

## Intended use

Teaching and benchmarking of tabular ML pipelines. Not for real decisions about people (credit, hiring, pay).

## Known biases

- Old data: incomes are from 1994 and are not adjusted for inflation, so the 50K threshold means something different today.
- Only the USA: about 90% of rows have native-country United-States, other countries have very few rows.
- Sex imbalance: about 67% male, 33% female. `>50K` is 30% for men and 11% for women.
- Race imbalance: about 86% White, 10% Black. Small groups (Amer-Indian-Eskimo, Other) have few rows, so results for them are unstable.
- Label imbalance: only 24% positive, so accuracy alone looks good even for a weak model. That is why F1 is also used.
- Missing values are not random: all 2,799 rows with `?` in workclass also have `?` in occupation (the other 10 are Never-worked).

## Privacy concerns

- No names or IDs, the data was anonymized by the Census Bureau.
- Quasi-identifiers are still present (age, sex, race, native-country, occupation). Rare combinations could help re-identify a person.
- race and sex are sensitive attributes. A model trained on them can learn and repeat discrimination,
  so a fairness check per group is needed before any real use.

## Label quality

- 5 groups of rows (10 rows) have identical features but different income labels.
- 50 random rows were checked against the data dictionary: no impossible values.
- On all rows: 4 rows where relationship and sex disagree (Husband/Female or Wife/Male),
  3 rows where a 17-year-old is married or divorced. These rows were reported, not removed.
