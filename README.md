# Data Notes

This repository does not include the full datasets used in the manuscript. It provides the minimum information needed to understand the expected model inputs and to reconstruct the workflow from the original source data.

## Source Data

The meteorological data used in this study were obtained from the China Meteorological Administration. These data were used to construct the climate predictors required for CNN model training, evaluation, and prediction.

Due to file-size constraints and data redistribution restrictions, the complete raw meteorological datasets are not included in this GitHub repository. Users should obtain the original data directly from the China Meteorological Administration or the corresponding authorized data service platform, following the applicable data access and usage terms.

The manuscript workflow also uses SST-derived predictors and basin or lake-area information. Before submission, please add the official source name, URL, DOI, or accession information for these datasets if they are derived from sources other than the China Meteorological Administration.

## Expected Processed Table Schema

The model-training script expects one processed CSV file per lead time:

- `basin1.csv`
- `basin2.csv`
- `basin3.csv`

These files are intentionally not tracked by Git. The expected column groups are:

| Columns | Role |
|---|---|
| `F01` to `F12` | monthly target variables |
| `year`, `month` | time index |
| `A1` to `A88` | atmospheric or basin-scale predictors |
| `O1` to `O26` | ocean or SST-derived predictors |

The first 12 columns are treated as target variables. Columns after the first 12 are treated as predictors by `src/train_cnn.py` and `src/predict_cnn.py`.

## Files Excluded From GitHub

The following files should remain local or be deposited in a dedicated data repository if required by the journal or reviewers:

- processed tables: `CNN/basin1.csv`, `CNN/basin2.csv`, `CNN/basin3.csv`
- trained weights: `CNN/results*/F*_model.pth`, `results/F*_model.pth`
- generated results: `CNN/results*/predictions_*.csv`, `results/cnn/**/*.csv`, generated figures
- raw meteorological, climate, or geospatial data: `*.nc`, `*.grib`, `*.grib2`, `*.tif`, `*.shp`, `*.zip`

## Optional Data Repository Plan

If reviewers or editors require access to the processed data, only the minimum necessary files for verification and reproducibility should be deposited in a dedicated data repository such as Zenodo, Figshare, or OSF:

- `basin1.csv`
- `basin2.csv`
- `basin3.csv`
- optional trained CNN weights used for exact reproduction
- a short README describing column meanings, data provenance, and processing steps

After deposit, update `DATA_AVAILABILITY.md` with the repository DOI or permanent access link.
