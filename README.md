# Data Notes

This repository does not include the full datasets used in the manuscript. It records the minimum information needed to understand the expected inputs and to reconstruct the workflow from source data.

## Public Source Data

The workflow uses seasonal forecast variables retrieved from the Copernicus Climate Data Store:

- dataset: `seasonal-original-single-levels`
- originating centre used in the provided script: `ecmwf`
- example system: `5`
- example variables: `2m_temperature`, `total_precipitation`, `evaporation`
- example retrieval script: `scripts/download_cds_seasonal.py`

The manuscript workflow also uses SST-derived predictors and basin or lake-area information. Before submission, add the official source name, URL, DOI, or accession information for those datasets here.

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

The following files should remain local or be deposited in a dedicated data repository if the journal asks for them:

- processed tables: `CNN/basin1.csv`, `CNN/basin2.csv`, `CNN/basin3.csv`
- trained weights: `CNN/results*/F*_model.pth`, `results/F*_model.pth`
- generated results: `CNN/results*/predictions_*.csv`, `results/cnn/**/*.csv`, generated figures
- raw climate or geospatial data: `*.nc`, `*.grib`, `*.grib2`, `*.tif`, `*.shp`, `*.zip`

## Optional Data Repository Plan

If reviewers require the processed data, deposit only the minimum necessary files in Zenodo, Figshare, or OSF:

- `basin1.csv`
- `basin2.csv`
- `basin3.csv`
- optional trained CNN weights used for exact reproduction
- a short README describing column meanings and provenance

After deposit, update `DATA_AVAILABILITY.md` with the DOI.
