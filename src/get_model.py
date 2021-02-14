import os
import pickle
import tarfile
import joblib
import s3fs
import mxnet as mx
import xgboost 


def get_model_weights(model_data_path, dest, method): # method in {'joblib', 'mxnet', pickle}
    """
    Input:
        model_data_path: Model path in sagemaker
        dest: destination path where the model will be saved
        method: whether the model is in {'joblib', 'mxnet', pickle} format
    Output:
        loaded_model: loaded model
    """
    
    if not os.path.isfile(dest):
        fs = s3fs.S3FileSystem()

        s_file = fs.open(model_data_path)
        d_file = open(dest, 'wb')

        d_file.write(s_file.read())

        s_file.close()
        d_file.close()
    

    tar = tarfile.open(dest, "r:gz")
    if method == 'joblib':
        loaded_model = joblib.load(tar.extractfile(member=tar.getmember(name="model.joblib")))
    elif method == 'mxnet':
        tar.extractall()
        os.system('unzip model_algo-1')
        loaded_model = mx.module.Module.load('mx-mod', 0)
        os.remove("model_algo-1")
        os.remove("mx-mod-0000.params")
        os.remove("mx-mod-symbol.json")
        os.remove("manifest.json")
        os.remove("additional-params.json")
    else: 
        loaded_model = pickle.load(tar.extractfile(member=tar.getmember(name="xgboost-model")))
    tar.close()
    return loaded_model