from __future__ import print_function

import argparse
import os
import pandas as pd
from collections import Counter
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

# def create_balanced_sample_weights(y_train, largest_class_weight_coef):
#     classes = y_train.unique()
#     classes.sort()
#     class_samples = np.bincount(y_train)
#     total_samples = class_samples.sum()
#     n_classes = len(class_samples)
#     weights = total_samples / (n_classes * class_samples * 1.0)
#     class_weight_dict = {key: value for (key, value) in zip(classes, weights)}
#     class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * largest_class_weight_coef
#     sample_weights = [class_weight_dict[y] for y in y_train]

#     return sample_weights

def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
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

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
#     n = train_y.shape[0]
#     occurrence_count = Counter(list(train_y)) 
#     occurrence_rate_majority_class = (occurrence_count.most_common(1)[0][1])/n
#     sample_weights_train_y = create_balanced_sample_weights(train_y, occurrence_rate_majority_class)
    
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(50s,), verbose=True, max_iter=100, learning_rate_init=0.005)
    model.fit(train_x, train_y)
    
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))