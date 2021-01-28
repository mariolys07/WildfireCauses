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