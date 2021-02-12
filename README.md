# Introduction

The goal is to build a machine learning model inspired by the following question posed under the “Inspiration” header of Kaggle’s 1.88 Million US Wildfires dataset;

“Given the size, location and date, can you predict the cause of a wildfire?”

The input features for the model will include the wildfire’s date, time of discovery, [geopolitical] state, and estimates size (in acres). Its output will be a cause amongst all possible causes in the dataset. More details can be found in the report.pdf file

# On this repository
* Data_exploration.ipynb - notebook for exploratory purposes
* Feature_engineering.ipynb - notebook to preprocess and extract train, validation and test data
* Train_xbg_model.ipynb - notebook to train a XGBoost model
* Train_linear_model.ipynb - notebook to train a Linear Learner model
* Train_mlp_model.ipynb - notebook to train an MLP model using scklearn.
* Refinement_mlp_model.ipynb - notebook for MLP model refinement using scklearn.
* Model_analysis.ipynb - notebook to train an MLP model using scklearn.
* source_sklearn/train_mlp.py - python file that is needed as an entry point for Train_mlp_model.ipynb notebook
* utils.py - python file with preprocessing function and model loading functionality.
* models - includes all the model used for training.
* wildfire_data - train, validation and test file in .csv formats.
* report.pdf: a full overview of the project and results

# Implementation
All the code is run in Amazon Sagemaker. The environments that were used are:
* conda_python36
* conda_mxnet_python36

# Citation

Kaggle dataset: https://www.kaggle.com/rtatman/188-million-us-wildfires

Short, Karen C. 2017. Spatial wildfire occurrence data for the United States, 1992-2015 [FPAFOD20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
