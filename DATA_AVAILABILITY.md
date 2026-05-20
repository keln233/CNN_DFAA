# Software and Data Availability

## Ready-to-paste statement

The source code used for CNN model training, evaluation, and prediction is publicly available at `https://github.com/<owner>/<repository>`. The public seasonal forecast predictors were obtained from the Copernicus Climate Data Store seasonal forecast product (`seasonal-original-single-levels`), and an example retrieval script is provided in the repository. The complete raw meteorological and geospatial datasets are not stored in the GitHub repository because of file-size constraints and third-party data terms; they can be downloaded from the original data providers using the sources and workflow described in the repository. The processed basin-level feature tables and trained CNN weights are intermediate analysis products and are excluded from GitHub; if required by the journal or reviewers, they should be deposited in a data repository such as Zenodo, Figshare, or OSF and cited with a DOI before publication.

## Chinese check before submission

- Replace `https://github.com/<owner>/<repository>` with the final GitHub URL.
- Confirm the exact public source for the SST data used in `cal_SST.ipynb`; add its official URL or DOI.
- Confirm whether the journal requires the processed feature tables and trained weights to be deposited separately. If yes, upload those files to Zenodo/OSF rather than GitHub and add the DOI here.
- If any basin boundary or lake-area data are not public, state the access restriction and provider clearly.

## Repository and citation actions

- Add the GitHub URL after the repository is created.
- Consider archiving the GitHub release with Zenodo after acceptance if the journal requests a DOI for software.
- Deposit only the minimum processed data needed for review if the editor requests data files, preferably in a data repository rather than GitHub.
