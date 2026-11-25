# FAA Wildlife-Strike Damage Classification (DS203)

![README: This README is AI generated](https://img.shields.io/badge/README-This%20README%20is%20AI%20generated-orange)

Predict the damage outcome of a reported wildlife strike from aircraft, flight, environmental and wildlife
information, using the FAA National Wildlife Strike Database (NWSD).

- **Main task (`y3`):** None / Minor / Substantial-or-Destroyed
- **Secondary task (`y_bin`):** damaged vs. not damaged

Assignment brief: [docs/DS203_FAA_wildlife_strike_classification.pdf](docs/DS203_FAA_wildlife_strike_classification.pdf)

> The damage classes are the database manager's coding of voluntarily submitted reports. They are not an
> official risk rating and say nothing about how often strikes happen.

---

## Project structure

```
faa-wildlife-strike-classification/
├── data/
│   ├── raw/                      # source files, never edited (git-ignored)
│   │   ├── Public.xlsx           # NWSD export
│   │   ├── AVONET.xlsx           # bird body mass + taxonomy
│   │   └── clements.csv          # eBird/Clements v2021 checklist (common -> scientific name)
│   └── processed/
│       └── dataset.csv           # cleaned, model-ready strike table (output of build-dataset.py)
├── utils/
│   ├── build-dataset.py          # the whole raw -> dataset.csv pipeline
│   └── modelling.py              # shared column lists, split, preprocessing, evaluation helpers
├── src/
│   ├── eda.ipynb                 # exploratory analysis
│   ├── dummy.ipynb               # majority baseline + class counts per split
│   ├── logistic_regression.ipynb # Model A and B, class-weight (gamma) tuning
│   ├── decision_tree.ipynb
│   ├── random_forest.ipynb
│   ├── imputation.ipynb          # mean vs median, with/without missing flags
│   ├── pca.ipynb                 # explained variance, loadings, models with PCA
│   ├── feature_groups.ipynb      # numerical-only, categorical-only, dropping single features
│   ├── interpretation.ipynb      # LR coefficients, permutation importance, misclassified strikes
│   └── binary.ipynb              # damaged vs not damaged (y_bin)
├── results/                      # CSVs written by the notebooks (scores, CV, importances)
├── plots/                        # figures written by the notebooks' "## plots" cells
├── report/
│   ├── report.tex                # long report
│   ├── report_short.tex          # 3–4 page report (Introduction / Methodology / Experiments)
│   ├── make_tables.py            # results/*.csv -> report/tables/*.tex
│   └── report_overleaf.zip       # upload to Overleaf to compile
├── presentation/
│   ├── wildlife_strike_presentation.pptx
│   └── script.md                 # four-speaker video script (also in the slide notes)
├── docs/                         # assignment brief
└── requirements.txt
```

## Setup and reproducing everything

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python utils/build-dataset.py      # run from the project root; ~30 s
```

`build-dataset.py` reads only `data/raw/` and overwrites `data/processed/dataset.csv`, so the dataset can always
be rebuilt from scratch.

Then run the notebooks in `src/` (each one reads `../data/processed/dataset.csv` and writes to `../results/` and
`../plots/`). Run `dummy`, `logistic_regression`, `decision_tree` and `random_forest` before `interpretation`.
Finally rebuild the report tables:

```bash
.venv/bin/python report/make_tables.py
```

---

## Data sources

| File | Source | Version | Accessed |
|---|---|---|---|
| `Public.xlsx` | FAA NWSD, https://wildlife.faa.gov/search ("Download Excel") | 361,499 records, 1990–2026 | 2026-11-23 |
| `AVONET.xlsx` | Tobias et al. (2022), *Ecology Letters*, Supplementary dataset 1, https://figshare.com/s/b990722d72a26b5bfead | sheet `AVONET2_eBird` | 2026-11-24 |
| `clements.csv` | eBird/Clements Checklist **v2021** (August 2021), https://www.birds.cornell.edu/clementschecklist/ | v2021 | 2026-11-24 |
| Field definitions | FAA field list, https://wildlife.faa.gov/assets/fieldlist.pdf | — | 2026-11-24 |

Clements **v2021** is used on purpose: AVONET's own metadata says its eBird taxonomy follows v2021. With the
newer v2026 checklist, renamed species (e.g. Cooper's hawk, American barn owl) fail to match AVONET.

---

## Data cleaning

Everything below happens in `utils/build-dataset.py`, in this order. Only **fixed, hand-written rules** are
applied here. Anything that is *learned from data* (imputation, rare-level grouping, encoding, scaling) is left
for the modelling pipeline, so it can be fitted on the training split only.

### 1. Load and restrict to 2010–2024
- Read the NWSD export (361,499 strikes) and keep incident years **2010–2024** → **217,993 strikes**.
- Keep only the columns we need: the brief's predictors, `SPECIES` (for matching), `DAMAGE_LEVEL` (target) and
  the `STR_*` / `ING_*` strike-location flags (Model B only).
- Everything else is dropped here, including all leakage fields (`DAM_*`, `EFFECT*`, `AOS`, `COST_*`,
  `NR_INJURIES`, `NR_FATALITIES`, `INDICATED_DAMAGE`, `COMMENTS`, `REMARKS`) and identifiers (`INDEX_NR`, `REG`,
  `FLT`, reporter names, dates).

### 2. Decode categorical codes
Codes are mapped to readable labels using FAA's field list. Unknown/blank values become `NaN` (not a
category), so they are handled by imputation later.

| Column | Rule |
|---|---|
| `AC_CLASS` | Strip padding (`'A  '` → `A`); A → airplane, B → helicopter, C/D/F/I/J/Y → other; Z (Unknown) → NaN |
| `TYPE_ENG` | A → piston, B → turbojet, C → turboprop, D → turbofan, E → none (glider), F → turboshaft, Y → other |
| `WARNED` | Yes → yes, No → no, **"Unknown" → NaN** |
| `PRECIPITATION` | See below |

**Precipitation.** FAA does not define what a blank means. In the data, "None" never appears on its own (only
inside combinations like "None, Rain"), so "no precipitation" is stored as a blank. Rule used:
- blank **and `SKY` reported** → `none` (weather section filled in, nothing ticked)
- blank **and `SKY` blank** → `NaN` (weather section skipped)
- two or more types (e.g. "Fog, Rain") → `mixed`

This cuts precipitation missingness from 96% to 34%.

### 3. Speed plausibility
- `SPEED` above **400 kt** indicated airspeed → `NaN` (above airliner maximum operating speeds). This removes one
  record (541 kt at 32,000 ft).
- `SPEED = 0` is **kept**: all 25 such strikes are on parked aircraft.
- The 400 kt cap is a fixed physical limit, not estimated from the data.

### 4. `BodyMass` — species matching
NWSD gives common names; AVONET uses scientific names. Two lookups link them:

```
SPECIES ──clean──> common name ──Clements v2021──> scientific name ──AVONET──> BodyMass, Order, Family
```

- **Name cleaning:** lowercase, strip, remove apostrophes, hyphens → spaces (so "Western Cattle-Egret" matches
  "western cattle egret").
- Only Clements rows with `category == "species"` are used, so each common name maps to exactly one species.
- AVONET `Mass` is in grams → divided by 1000 to give **kg**.
- **Manual fixes** (NWSD uses names introduced after 2021; each is a documented split where the 2021 species
  contained the new one):

  | NWSD name | Scientific name used |
  |---|---|
  | American barn owl | *Tyto alba* |
  | Western cattle egret | *Bubulcus ibis* |
  | Northern house wren | *Troglodytes aedon* |

- **Left unmatched on purpose:** generic entries ("Unknown bird – small", "Unknown bird"), group names
  ("Gulls", "Sparrows"), ambiguous names ("Western flycatcher", "Redpoll"), bats, other mammals and reptiles.
  AVONET only covers birds, and a group name has no single mass. `SIZE` remains as a coarse fallback.
- Both merges are left joins and the row count is unchanged, so no strike is dropped or duplicated.

Match rate: 56.0% of all 2010–2024 strikes, **41.8% of the final (labelled) dataset**. About a third of all
strikes are "Unknown bird", which caps what any lookup can achieve.

### 5. `ImpactEnergy`
- E = ½ · m · v², with m = `BodyMass` (kg) and v = 0.514444 × `SPEED` (knots → m/s), giving joules.
- `NaN` whenever mass or speed is missing; parked strikes (0 kt) get E = 0.
- `log_ImpactEnergy = log1p(E)` because E is heavily right-skewed.
- Available for only **13.4%** of the final dataset, because `SPEED` is reported in a minority of strikes.

### 6. `SpeciesGroup`
- Taxonomic **order** of the matched species (22 orders; chosen over 80 families to keep categories manageable).
- Unmatched strikes get an explicit `Unknown` level.
- Grouping rare orders into `Other` is **not** done here — the threshold must be fixed on the training split.

### 7. Targets
From `DAMAGE_LEVEL` (FAA/ICAO definitions):

| `DAMAGE_LEVEL` | Meaning | `y3` | `y_bin` | Strikes |
|---|---|---|---|---|
| N | none | 0 | 0 | 119,882 |
| M | minor | 1 | 1 | 2,799 |
| M? | damaged, extent undetermined | NaN | 1 | 5,528 |
| S | substantial | 2 | 1 | 1,428 |
| D | destroyed | 2 | 1 | 36 |
| blank | unknown | *row removed* | *row removed* | 88,320 |

- Blank labels have no usable label for either task, so **88,320 strikes are removed**.
- M? cannot be placed on the three-class scale, so it is `NaN` in `y3` but counts as damaged in `y_bin`.
- `DAMAGE_LEVEL` is then dropped so it cannot leak into the predictors.

### 8. Final tidy-up
`SPECIES`, `scientific_name`, `Order` and `Family` are dropped — they were only needed to build `BodyMass` and
`SpeciesGroup`.

---

## Final dataset: `data/processed/dataset.csv`

**129,673 strikes × 41 columns**, one row per strike, missing values preserved as blanks.

| Group | Columns |
|---|---|
| Numerical | `HEIGHT`, `SPEED`, `DISTANCE`, `INCIDENT_MONTH`, `BodyMass`, `ImpactEnergy`, `log_ImpactEnergy` |
| Ordinal | `AC_MASS` (1–5), `SIZE` (Small < Medium < Large), `NUM_SEEN`, `NUM_STRUCK` (1 < 2-10 < 11-100 < More than 100) |
| Nominal | `AC_CLASS`, `NUM_ENGS`, `TYPE_ENG`, `PHASE_OF_FLIGHT`, `TIME_OF_DAY`, `FAAREGION`, `SKY`, `PRECIPITATION`, `WARNED`, `SpeciesGroup` |
| Strike location (Model B only) | `STR_RAD`, `STR_WINDSHLD`, `STR_NOSE`, `STR_ENG1–4`, `ING_ENG1–4`, `STR_PROP`, `STR_WING_ROT`, `STR_FUSE`, `STR_LG`, `STR_TAIL`, `STR_LGHTS`, `STR_OTHER` (0/1) |
| Targets | `y3` (0/1/2, NaN for M?), `y_bin` (0/1) |

**Class balance**
- `y3` (124,145 usable rows): None 119,882 · Minor 2,799 · Substantial/Destroyed 1,464 — roughly 82 : 2 : 1
- `y_bin` (129,673 rows): not damaged 119,882 · damaged 9,791

**Missingness (% of rows)**

| ≥ 50% | 30–50% | 15–30% | < 1% |
|---|---|---|---|
| `ImpactEnergy` 86.6, `BodyMass` 58.2, `SPEED` 56.8, `NUM_SEEN` 55.0, `WARNED` 50.7 | `DISTANCE` 44.9, `SKY` 34.2, `PRECIPITATION` 34.1, `HEIGHT` 33.3 | `TIME_OF_DAY` 22.9, `FAAREGION` 21.4, `SIZE` 18.5, `PHASE_OF_FLIGHT` 18.3 | `AC_*`, `NUM_ENGS`, `TYPE_ENG`, `NUM_STRUCK`, `INCIDENT_MONTH`, `SpeciesGroup`, `STR_*`/`ING_*` |

Missingness is **not random**: e.g. body mass is known for 40.6% of N strikes but 55.7% of S strikes, and
labelled reports include speed more often than unlabelled ones. This is analysed in the EDA.

---

## Modelling

All shared code is in `utils/modelling.py`.

1. **Load** `dataset.csv` and drop rows without the target (`y3` for the main task, `y_bin` for the binary task).
2. **Split** 60 / 20 / 20 train / validation / test, **stratified** on the target, seed 67. The same split is used
   by every notebook. The test set is scored once, in the last cell of each model notebook.
3. **Preprocess inside a scikit-learn `Pipeline` + `ColumnTransformer`** (fitted on training data only, refitted
   in each CV fold):
   - numerical (`HEIGHT`, `SPEED`, `DISTANCE`, `BodyMass`, `log_ImpactEnergy`, `AC_MASS`, `NUM_ENGS`): median
     imputation + missingness indicators, then standard scaling
   - ordinal (`SIZE`, `NUM_SEEN`, `NUM_STRUCK`): ordered integer encoding, most-frequent imputation + indicators
   - nominal (incl. `INCIDENT_MONTH`, treated as a category): explicit `missing` level, levels seen < 50 times in
     training grouped as infrequent, one-hot with unseen levels mapped to the infrequent column
   - Model B passes the 18 `STR_*` / `ING_*` flags through unchanged
4. **Class imbalance:** class weights `(n_None / n_class) ** gamma`, with `gamma` tuned between 0 (no weighting)
   and 1 (`"balanced"`).
5. **Models and tuning** (5-fold stratified CV on macro F1):
   - majority-class baseline
   - multinomial logistic regression: `gamma`
   - decision tree: `max_depth`, `min_samples_leaf`, `gamma` (grid search)
   - random forest (100 trees): `max_depth`, `min_samples_leaf`, `max_features`, `gamma` (randomised search)
6. **Further experiments** (Model A): mean vs median imputation, PCA, feature-group ablations, LR coefficients,
   permutation importance, misclassified strikes, binary task.

## Results (test set)

| Model | Macro F1 (A) | Macro F1 (B) | Balanced acc. (A) | Recall S/D (A) |
|---|---|---|---|---|
| Majority baseline | 0.328 | — | 0.333 | 0.00 |
| Logistic regression | 0.495 | 0.541 | 0.528 | 0.41 |
| Decision tree | 0.460 | 0.537 | 0.446 | 0.27 |
| **Random forest** | **0.516** | **0.562** | **0.535** | **0.42** |

- Binary task (damaged vs not): random forest macro F1 0.722 (A) / 0.755 (B), PR-AUC 0.467 (A).
- Most important features (random forest, Model A): `SIZE`, `BodyMass`, `AC_MASS`, `HEIGHT`, `PHASE_OF_FLIGHT`.
- Class weighting (`gamma` ≈ 0.25–0.5) mattered most; PCA lowered scores and the imputation method made no
  measurable difference.
- Full tables are in `results/` and the reports in `report/`.

## Known limitations
- Reporting is voluntary; damaging strikes are more likely to be reported in detail.
- 40% of 2010–2024 strikes have no damage label and are excluded.
- Body mass is unavailable for unidentified animals, group names and non-birds; impact energy additionally needs
  a reported speed.
- From 29 March 2021, non-engine ingestions are recorded differently (`INGESTED_OTHER`, not in this dataset).
- Severe class imbalance (≈ 82 : 2 : 1).
- The split is random, not by time; a time-based robustness check was not done.
- The random forest's best `max_depth` (None) and `max_features` (0.3) are at the edge of the searched values.
