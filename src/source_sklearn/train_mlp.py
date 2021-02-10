from __future__ import print_function

import argparse
import os
import pandas as pd
from collections import Counter
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

def model_fn(model_dir):
    """
    Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    
    args = parser.parse_args()

    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train_ref.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    """
    Refinement attempts:
    MODEL 1:
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(50,), verbose=True, max_iter=100, learning_rate_init=0.005)
    Loss: 1.20
    The accuracy for val set is: 0.4157066765762418
    The accuracy for test set is: 0.40956052320009023
    TEST:
    Causes Description	F1 scores
    0	Lightning	0.621717
    1	Debris Burning	0.523913
    2	Arson	0.266126
    3	Miscellaneous	0.389600
    4	Missing/Undefined	0.096244
    5	Other	0.272116
    VAL:
    Causes Description	F1 scores
    0	Lightning	0.621647
    1	Debris Burning	0.526512
    2	Arson	0.294259
    3	Miscellaneous	0.327812
    4	Missing/Undefined	0.127157
    5	Other	0.296449
    
    
    Model 2: 
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(50,), verbose=True, max_iter=100, learning_rate_init=0.05)
    Iteration 13, loss = 1.29906690
    Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
    2021-02-10 05:57:59,538 sagemaker-containers INFO     Reporting training SUCCESS

    Model 3:
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(25,), verbose=True, max_iter=100, learning_rate_init=0.005)
    Iteration 100, loss = 1.20385515
    /miniconda3/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:585: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (100) reached and the optimization hasn't converged yet
    The accuracy for val set is: 0.41752100319875257
    The accuracy for test set is: 0.39891892653774597
    TESTS:
    Causes Description	F1 scores
    0	Lightning	0.619219
    1	Debris Burning	0.513346
    2	Arson	0.287825
    3	Miscellaneous	0.369872
    4	Missing/Undefined	0.089993
    5	Other	0.276137
    VAL:
    Causes Description	F1 scores
    0	Lightning	0.618398
    1	Debris Burning	0.521118
    2	Arson	0.308528
    3	Miscellaneous	0.336867
    4	Missing/Undefined	0.155206
    5	Other	0.308634
    
    Model 4:
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(25,), verbose=True, max_iter=100, learning_rate_init=0.05)
    Iteration 13, loss = 1.30057939
    2021-02-10 06:52:08 Uploading - Uploading generated training modelIteration 14, loss = 1.30060966
    Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
    2021-02-10 06:52:06,114 sagemaker-containers INFO     Reporting training SUCCESS
    The accuracy for val set is: 0.4085368075137896
    The accuracy for test set is: 0.3815188588825619
    Test:
    Causes Description	F1 scores
    0	Lightning	0.579014
    1	Debris Burning	0.483936
    2	Arson	0.272922
    3	Miscellaneous	0.300102
    4	Missing/Undefined	0.089003
    5	Other	0.262530
    Val:
    Causes Description	F1 scores
    0	Lightning	0.582622
    1	Debris Burning	0.519680
    2	Arson	0.364454
    3	Miscellaneous	0.257893
    4	Missing/Undefined	0.114709
    5	Other	0.287345
    
    Model 5:
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(25,), verbose=True, max_iter=100, learning_rate_init=0.009)
    Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
    2021-02-10 20:35:50,294 sagemaker-containers INFO     Reporting training SUCCESS
    
    Model 6:
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(25,), verbose=True, max_iter=100, learning_rate_init=0.0001)
    Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
    2021-02-10 21:15:55,931 sagemaker-containers INFO     Reporting training SUCCESS
    
    Model 7:
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(25,), verbose=True, max_iter=200, learning_rate_init=0.005)
    Iteration 200, loss = 1.19330057
    /miniconda3/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:585: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
    The accuracy for val set is: 0.41180842459614836
    The accuracy for test set is: 0.3902012741726335
    Test
    Causes Description	F1 scores
    0	Lightning	0.612258
    1	Debris Burning	0.495596
    2	Arson	0.298165
    3	Miscellaneous	0.376625
    4	Missing/Undefined	0.098685
    5	Other	0.257304
    Causes Description	F1 scores
    0	Lightning	0.615521
    1	Debris Burning	0.506771
    2	Arson	0.321058
    3	Miscellaneous	0.353165
    4	Missing/Undefined	0.144904
    5	Other	0.290898
    
    
    """
    
    
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(25,), verbose=True, max_iter=200, learning_rate_init=0.005)
    model.fit(train_x, train_y)
    
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))