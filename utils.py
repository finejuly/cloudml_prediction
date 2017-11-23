import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout

import os

from google.cloud import storage

from googleapiclient import discovery

def mnist_preprocessing(x_train, y_train, x_test, y_test):
    
    num_classes = 10
    
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

def gen_model():
    
    num_classes = 10
    
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model


def upload_blob_folder(bucket_name, source_folder_name, destination_path_name):
    """Uploads all the files in a folder to the bucket."""
    """ Il-Young Jeong finejuly@gmail.com """
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    for (root, dirs, files) in os.walk(source_folder_name):
        for (fi) in files:
            source_file_name = os.path.join(root,fi)
            destination_blob_name = os.path.join(destination_path_name,source_file_name[2:])
            blob = bucket.blob(destination_blob_name)   
            blob.upload_from_filename(source_file_name)

            print('File {} uploaded to {}.'.format(
                source_file_name, destination_blob_name))
    print('Done')
    
    
def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        #body={'instances': instances}
        body = instances
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']