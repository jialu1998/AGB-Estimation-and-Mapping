# Influence of seasonal canopy conditions on ICESat-2-based aboveground biomass estimation in deciduous forests

### 1. Source Data

This study utilized ICESat-2 ATL08 data collected within the boundaries of the New England study area, covering the period from November 2018 to September 2019. The ATL08 data were obtained from the [National Snow and Ice Data Center (NSIDC)](https://nsidc.org/data/ATL08/versions/6). The reference aboveground biomass (AGB) dataset is available from the [ORNL DAAC website](https://daac.ornl.gov/CMS/guides/AGB_CanopyHt_Cover_NewEngland.html), and the global land cover data product can be accessed via [CASEarth](https://data.casearth.cn/).

------

### 2. Source Code

The source code for AGB model construction and mapping is available in the GitHub repository: [AGB-Estimation-and-Mapping](https://github.com/jialu1998/AGB-Estimation-and-Mapping).

- **Random Forest-based AGB estimation**:
   [`source code/AGB estimation`](https://github.com/jialu1998/AGB-Estimation-and-Mapping/tree/main/source code/AGB estimation)
- **Multiple Linear Stepwise Regression (MLSR)-based AGB estimation**:
   Implemented using SPSS software (not included in this repository).
- **CNN-based AGB mapping**:
   [`source code/Forest AGB mapping/CNN_mapping`](https://github.com/jialu1998/AGB-Estimation-and-Mapping/tree/main/source code/Forest AGB mapping/CNN_mapping)
- **Random Forest-based AGB mapping**:
   [`source code/Forest AGB mapping/RF_mapping`](https://github.com/jialu1998/AGB-Estimation-and-Mapping/tree/main/source code/Forest AGB mapping/RF_mapping)

------

### 3. Result Data

Accuracy evaluation results are available at:
 [`result data/AGB estimation`](https://github.com/jialu1998/AGB-Estimation-and-Mapping/tree/main/result data/AGB estimation)