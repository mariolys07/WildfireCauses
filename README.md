# Introduction

The goal is to build a machine learning model inspired by the following question posed under the “Inspiration” header of Kaggle’s 1.88 Million US Wildfires dataset;

“Given the size, location and date, can you predict the cause of a wildfire?”

The input features for the model will include the wildfire’s date, time of discovery, [geopolitical] state, and estimates size (in acres). Its output will be a cause amongst all possible causes in the dataset. More details can be found in the report.pdf file

# On this repository
* 1_Data_Exploration.ipynb - notebook for exploratory purposes
* 2_Feature_Engineering.ipynb - notebook to preprocess and extract train, validation and test data
* 3_Train_Linear_Model.ipynb - notebook to train a Linear Learner model
* 4_Train_MLP_Model.ipynb - notebook to train an MLP model using scklearn.
* 5_Train_XGB_Model.ipynb - notebook to train a XGBoost model
* 6_Train_Refined_MLP_Model.ipynb - notebook for MLP model refinement using scklearn.
* 7_Interpretability.ipynb - notebook to train an MLP model using scklearn.
* utils.py - python file with preprocessing function and model loading functionality.
* get_model.py - python file to download trained models.
* models - includes all the models obtained after training and refinement.
* mlp_source/train_mlp.py - python file that is needed as an entry point to train MLP models.
* wildfire_data/train, validation, test - files in .csv formats used for trainning.
* wildfire_data/train_ref, validation_ref, test_ref - files in .csv formats used for refinement.
* wildfire_data/causes_names.pkl - correspondence between cause descriptions and integer values.
* wildfire_data/causes_names_refinement.pkl - correspondence between refined cause descriptions and integer values.
* wildfire_data/feature_names.pkl - correspondence between all features and its preprocessed values.
* report.pdf: a full overview of the project and results

# Implementation
All the code is run in Amazon Sagemaker. The environments that were used are:
* conda_amazonei_mxnet_p36 - for the the `3_Train_Linear_Model.ipynb` notebook.
* conda_mxnet_python36 - for the `7_Interpretability.ipynb` notebook.
* conda_python36 - for the rest of the notebooks.

# Citation

Kaggle dataset: https://www.kaggle.com/rtatman/188-million-us-wildfires

Short, Karen C. 2017. Spatial wildfire occurrence data for the United States, 1992-2015 [FPAFOD20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
