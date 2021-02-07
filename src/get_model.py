import s3fs
import tarfile
import joblib

def get_model_weights(model_data_path, dest):
    fs = s3fs.S3FileSystem()

    s_file = fs.open(model_data_path)
    d_file = open(dest, 'wb')

    d_file.write(s_file.read())

    s_file.close()
    d_file.close()

    tar = tarfile.open(dest, "r:gz")
    loaded_model = joblib.load(tar.extractfile(member=tar.getmember(name="model.joblib")))
    return loaded_model