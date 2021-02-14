# Introduction

This is my Capstone project for the **Udacity Machine Learning Engineer Nanodegree.**

The goal is to build a machine learning model inspired by the following question posed under the “Inspiration” header of [Kaggle’s 1.88 Million US Wildfires dataset](https://www.kaggle.com/rtatman/188-million-us-wildfires);

> “Given the size, location and date, can you predict the cause of a wildfire?”

The input features for my model include the wildfire’s date, time of discovery, (geopolitical) state, and estimated size (in acres). Its output will be a cause amongst all possible wildfire causes in the dataset.

More details can be found in the [`report.pdf`](blob/main/report.pdf) file.

# Implementation
The results of this project can be reproduced by loading the repo into an [Amazon SageMaker](https://aws.amazon.com/sagemaker/) Notebook Instance. The notebooks in the `src/` directory can be executed in the order specified by their filename prefixes.

You may need to specify the following Notebook kernels:

* `conda_amazonei_mxnet_p36` for the the `3_Train_Linear_Model.ipynb` notebook.
* `conda_mxnet_python36` for the `7_Interpretability.ipynb` notebook.
* `conda_python36` for all other notebooks.

# In this repository

## Documents

* [`proposal.pdf`](blob/main/proposal.pdf) - Project proposal.
* [`report.pdf`](blob/main/report.pdf) - Overview of the project and results.

## Data

The data used in this project was obtained from [Kaggle’s 1.88 Million US Wildfires dataset](https://www.kaggle.com/rtatman/188-million-us-wildfires) and is stored in the [`src/wildfire_data`](tree/main/src/wildfire_data) directory. This data can be loaded using the function `load_raw_wildfire_data` from [`src/utils.py`](blob/main/src/utils.py).

## Notebooks

Notebooks in this repository are numbered and are meant to be executed in the order listed below. The `2_Feature_Engineering.ipynb` notebook generates new `pkl` and `csv` files into the `src/wildfire_data` directory. These are essential to the other notebooks.

* [`1_Data_Exploration.ipynb`](blob/main/src/1_Data_Exploration.ipynb) - For exploratory purposes.
* [`2_Feature_Engineering.ipynb`](blob/main/src/2_Feature_Engineering.ipynb) - Preprocess and store train, validation and test data.
* [`3_Train_Linear_Model.ipynb`](blob/main/src/3_Train_Linear_Model.ipynb) - Train a Linear Learner model.
* [`4_Train_MLP_Model.ipynb`](blob/main/src/4_Train_MLP_Model.ipynb) - Train an MLP model.
* [`5_Train_XGB_Model.ipynb`](blob/main/src/5_Train_XGB_Model.ipynb) - Train a XGBoost model.
* [`6_Train_Refined_MLP_Model.ipynb`](blob/main/src/6_Train_Refined_MLP_Model.ipynb) - Train a refined MLP model with a reduced output classes.
* [`7_Interpretability.ipynb`](blob/main/src/7_Interpretability.ipynb) - Explore feature importance under MLP model.

## Other Python Code

* [`utils.py`](blob/main/src/utils.py) - Basic helper functions including data loading.
* [`get_model.py`](blob/main/src/get_model.py) - Helper function to download SageMaker trained models.
* [`mlp_source/train_mlp.py`](blob/main/src/mlp_source/train_mlp.py) - Entry point for MLP model training.

## Models

* [`models/`](tree/main/src/models) - All models trained by notebooks 3, 4, 5, and 6 above, as archived by SageMaker.

## Generated Data

When the `2_Feature_Engineering.ipynb` notebook is run, the following files are generated inside the `src/wildfire_data` directory:

* `train.csv`, `validation.csv`, `test.csv` - CSV data for training, validation, and testing.
* `train_ref.csv`, `validation_ref.csv`, `test_ref.csv` - CSV data for refined models with reduced output classes.
* `causes_names.pkl` - List of all original wildfire cause names.
* `causes_names_refinement.pkl` - List of reduced wildfire cause names.
* `feature_names.pkl` - List of names of the features used for training and testing the models.

# References

[Kaggle’s 1.88 Million US Wildfires dataset](https://www.kaggle.com/rtatman/188-million-us-wildfires)

**The above dataset was originally obtained from:**

Short, Karen C. 2017. Spatial wildfire occurrence data for the United States, 1992-2015 [FPAFOD20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
