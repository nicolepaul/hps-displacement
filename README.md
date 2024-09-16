# Household displacement after disasters in the United States

## Overview

In 2022, more than 1.3% of the adult population (3.3M) in the United States was displaced by disasters [(AP News)](https://apnews.com/article/natural-disasters-indiana-florida-climate-and-environment-0bfdab41b233feba55e08382a0594258). This repository includes code to investigate the public use files (PUF) from the [**United States Household Pulse Survey (HPS)**](https://www.census.gov/programs-surveys/household-pulse-survey.html). Information regarding displacement following disasters was introduced from Phase 3.7 (Week 52). The availability of microdata allows an exploration of various factors that may be associated longer displacement durations.

To explore data trends, a simple dashboard is available at [hps.nicolepaul.io](https://hps.nicolepaul.io/)

## Contents

This repository contains Python code to perform exploratory analysis and fit machine learning models to the HPS data. This work is being submitted for publication.

This code mainly comprises four Jupyter Notebooks that were used to derive the published results:

- [**0. Exploratory analysis.ipynb**](/0.%20Exploratory%20analysis.ipynb): Basic data analysis to get descriptive statistics and explore trends between factors related to household displacement and return after disasters
- [**1. Classification tree.ipynb**](/1.%20Classification%20tree.ipynb): Fits a decision tree model for household displacement durations
- [**2. Random forest.ipynb**](/2.%20Random%20forest.ipynb): Fits a random forest model for household displacement durations
- [**3. Model variant - physical factors only.ipynb**](/3.%20Model%20variant%20-%20physical%20factors%20only.ipynb): Fits a decision tree model, but only considers physical factors typically included within disaster risk analyses
- [**4. Random forest explanations.ipynb**](/2.%20Random%20forest%20explanations.ipynb): Loads a presaved random forest model and uses SHAP values to explain model predictions

Additionally, presaved versions of the fitted models are available in the **presaved** folder.
- **model_tree.sav**: The TreeP&S model
- **model_forest.sav**: The ForestP&S model
The **grid_** files provide the results from hyperparameter tuning

We also include supplemental notebooks to support various model variants, such as considering different machine learning model types. Please see the **supplement** folder to see those.

## Running the notebooks

Several common python libraries are used in the notebooks, in addition to custom scripts. It is recommended to run these notebooks using a [virtual environment](https://docs.python.org/3/library/venv.html). Once you have a virtual environment activated, you can install all dependencies with `pip install -r requirements.txt`

The HPS PUF zipped CSVs will need to be downloaded separately: https://www.census.gov/programs-surveys/household-pulse-survey/data/datasets.html. Once downloaded, please specify the path to the folder containing all downloaded PUFs as `puf_folder` in each notebook.