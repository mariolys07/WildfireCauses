import io
import os
import math
import pickle
import numpy as np
import pandas as pd
import boto3

def load_raw_wildfire_data(paths = None):
    
    if isinstance(paths, str):
        paths = [paths]
        
    paths = paths or ["wildfire_data/Fires0.pkl", "wildfire_data/Fires1.pkl"]
    
    if len(paths) > 0 and os.path.isfile(paths[0]):
        # This code loads the data from a local file.
        # Multiple files can be used to avoid large files.
        
        fires = b''
        
        for path in paths:
            with open(path, "rb") as file:
                fires += file.read()
                
        df = pickle.loads(fires)
        
        print("Loaded from local files.")

    else:
        # This code loads it from from an S3 bucket

        s3_client = boto3.client('s3')
        bucket_name = 'wildfires'

        # get a list of objects in the bucket
        obj_list=s3_client.list_objects(Bucket=bucket_name)

        # print object(s)in S3 bucket
        files=[]
        for contents in obj_list['Contents']:
            files.append(contents['Key'])

        file_name = files[0]
        data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)

        data_body = data_object["Body"].read()
        data_stream = io.BytesIO(data_body)
        df = pd.read_pickle(data_stream)
        
        print("Loaded from S3 bucket.")
        
    return df

def get_cause_names():
    with open("wildfire_data/cause_names.pkl", "rb") as file: names = pickle.load(file)
    return names

def get_cause_names_refinement():
    with open("wildfire_data/cause_names_refinement.pkl", "rb") as file: names = pickle.load(file)
    return names

def print_f1_scores(f1_score_result, cause_for_code=None):
    cause_for_code = cause_for_code or get_cause_names()
    """
    Input:
        f1_score_result: an array of probability values
        cause_for_code: a dictionary that maps integer classes to corresponding cause descriptions
    Output:
        df_f1_score: a dataframe with cause descriptions and their f1 scores
    """
    n = len(f1_score_result)
    all_tuples = [(cause_for_code[i], f1_score_result[i]) for i in range(n)]
    df_f1_score = pd.DataFrame(all_tuples, columns=['Causes Description', 'F1 scores'])
    return df_f1_score