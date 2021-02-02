import math
import pandas as pd 
import numpy as np
import math

def standardize_data(data):
    df_copy = data.copy()

    df_copy['FIRE_YEAR'] = df_copy['FIRE_YEAR']/2000 # year column 
    df_copy['DISCOVERY_DOY_SIN'] = df_copy['DISCOVERY_DOY'].apply(lambda x: math.sin(2 * math.pi * x /366)) # Discovery DOY 
    df_copy['DISCOVERY_DOY_COS'] = df_copy['DISCOVERY_DOY'].apply(lambda x: math.cos(2 * math.pi * x/366)) # Discovery DOY
    df_copy['FIRE_SIZE'] = df_copy['FIRE_SIZE'].apply(lambda x: np.log(x))# fire size
    del df_copy['DISCOVERY_DOY']
    
    # dummies variables
    df_dummies = df_copy
    object_type_columns = [column_name for column_name in df_dummies.columns 
                       if pd.api.types.is_object_dtype(df_dummies[column_name].dtype)]
    for column_name in object_type_columns:
        dummies = pd.get_dummies(df_dummies[column_name])
        del dummies[dummies.columns[-1]]
        df_dummies = pd.concat([df_dummies, dummies], axis=1)
        del df_dummies[column_name]
    return df_dummies



def create_balanced_sample_weights(y_train, largest_class_weight_coef):
    classes = y_train.unique()
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key: value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]

    return sample_weights