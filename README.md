# Household displacement after disasters in the United States

## Overview

This repository includes code to investigate the public use files (PUF) from the [**United States Household Pulse Survey (HPS)**](https://www.census.gov/programs-surveys/household-pulse-survey.html) to understand displacement duration and return after recent disasters. Please refer to the associated open access paper for more details:

> Paul, N., Galasso, C., Baker, J., & Silva, V. (2025). A predictive model for household displacement duration after disasters. *Risk Analysis*, 1–29. https://doi.org/10.1111/risa.17710

To explore data trends, a simple dashboard is available at [hps.nicolepaul.io](https://hps.nicolepaul.io/). For the dashboard code, please refer to a separate [repository](https://github.com/nicolepaul/dash-ushh-displacement).

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

The HPS PUF zipped CSVs will need to be downloaded separately: https://www.census.gov/programs-surveys/household-pulse-survey/data/datasets.html. The disaster displacement questions were added during Phase 3.7 (Week 52) and stopped after Phase 4.2 Cycle 9. Once downloaded, please specify the path to the folder containing all downloaded PUFs as `puf_folder` in each notebook. Note that the code may error if you include PUFs that do not contain the disaster displacement questions (i.e., before Phase 3.7 Week 52 or after Phase 4.2 Cycle 9).
