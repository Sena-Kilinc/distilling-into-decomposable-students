# Distilling into Decomposable Students

Code and results for the PhD candidate assignment *"Knowledge Distillation into Analytically Decomposable Students: Explainable AI via Polynomial Chaos Expansion and HDMR"* (TÜBİTAK 1001 project, PI Assoc. Prof. Dr. Gülşen Taşkın, ITU). Author: Sena Kılınç.

A frozen **LightGBM** teacher is distilled into two interpretable students. Both are evaluated on two real tabular datasets: **Online Shoppers Purchasing Intention** (17 features, binary) and **UCI-HAR** (561 features, 6 classes).

| | Student 1 | Student 2 |
|---|---|---|
| Model | Sparse PCE truncated as 2nd-order HDMR: data-driven orthonormal basis, lasso path + OLS refit | CART global surrogate: regression tree on the teacher's probabilities |
| Target | Teacher logits (binary log-odds or centred multiclass logits) | Teacher probability vector |
| Explanation | Sobol' S1/ST from the coefficients, plus a covariance (ANCOVA-type) index | Impurity (variance-reduction) importance |

The post-hoc explainer applied to the teacher is **TreeSHAP**.

## Everything is in one notebook

`kd_pce_hdmr_distillation.ipynb` runs the whole study top to bottom:
1. data loading and leakage-safe splits;
2. teacher training, then freezing;
3. both students;
4. fidelity and label metrics;
5. TreeSHAP and explanation agreement;
6. bootstrap stability;
7. figures, tables and a generated summary of key findings.

The notebook is stored **with its outputs**, so the results can be read on GitHub without running anything.

## Reproduce

1. **Python ≥ 3.11** (3.13 was used). Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data.** The raw data are not included (UCI, CC BY 4.0). Download the official files into `data/raw/`:
   - https://archive.ics.uci.edu/dataset/468 (DOI 10.24432/C5F88Q), giving `data/raw/online_shoppers_intention.csv`
   - https://archive.ics.uci.edu/dataset/240 (DOI 10.24432/C54S4K), giving `data/raw/UCI HAR Dataset/` (the zip also works)

   Checksums of the files used are in `data/raw_checksums.sha256`.
3. **Run.** Open the notebook and choose *Run All*, or run it headless:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace kd_pce_hdmr_distillation.ipynb
   ```
   It takes about 8 minutes on 2 CPU cores. Outputs are written to `outputs/results/`, `outputs/figures/` and `outputs/tables/`.

Seeds are fixed. Two complete runs gave identical results in our environment. LightGBM multithreading can cause tiny numerical differences on other hardware.

## Protocol in brief

- **Splits:**
  - OSPI: 125 exact duplicates removed before splitting, then a stratified 60/20/20 split.
  - HAR: the official subject-wise split, with 5 training subjects held out for validation.
- **Teacher:** hyper-parameters chosen on validation, then frozen.
- **Model selection:** one rule for both students, fixed before testing. Choose the least complex model within 0.005 of the best **validation** macro-F1 against the teacher's labels.
- **Evaluation:** every metric is reported on the common test set.
  - label metrics;
  - fidelity: agreement, macro-F1 vs teacher, TV / KL / Brier between probability vectors;
  - structural complexity;
  - explanation agreement: Spearman, Kendall τ-b, top-k Jaccard; HAR also at the level of 18 signal groups;
  - bootstrap stability: B = 5, teacher frozen.

## Key limitation

Coefficient-based Sobol' indices assume independent inputs. We measure how far this fails, as ρ = Σc² / Var, which equals 1 for an orthonormal basis. ρ is 0.85 on OSPI and 4.70 on HAR, so on HAR the analytic indices are not valid variance shares. Other limitations:
- one teacher family, one HAR split, 5 OSPI splits and 5 bootstraps;
- HAR pairwise terms are restricted to the 30 inputs selected by training-only screening;
- the CART student is fitted on real training rows only (no TREPAN-style query sampling).
