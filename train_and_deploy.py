'''
DERIVED FROM:https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sklearn/README.rst

Preparing the Scikit-learn training script
Your Scikit-learn training script must be a Python 2.7 or 3.5 compatible source file.

The training script is very similar to a training script you might run outside of SageMaker, 
but you can access useful properties about the training environment through various environment variables, 
such as

- SM_MODEL_DIR: 
        A string representing the path to the directory to write model artifacts to. 
        These artifacts are uploaded to S3 for model hosting.
- SM_OUTPUT_DATA_DIR: 
        A string representing the filesystem path to write output artifacts to. 
        Output artifacts may include checkpoints, graphs, and other files to save, 
        not including model artifacts. These artifacts are compressed and uploaded 
        to S3 to the same S3 prefix as the model artifacts.

        Supposing two input channels, 'train' and 'test', 
        were used in the call to the Scikit-learn estimator's fit() method, 
        the following will be set, following the format "SM_CHANNEL_[channel_name]":

- SM_CHANNEL_TRAIN: 
        A string representing the path to the directory containing data in the 'train' channel

- SM_CHANNEL_TEST: 
        Same as above, but for the 'test' channel.
        A typical training script loads data from the input channels, 
        configures training with hyperparameters, trains a model, 
        and saves a model to model_dir so that it can be hosted later. 
        Hyperparameters are passed to your script as arguments and can 
        be retrieved with an argparse.ArgumentParser instance. 
        For example, a training script might start with the following:


Because the SageMaker imports your training script, 
you should put your training code in a main guard (if __name__=='__main__':) 
if you are using the same script to host your model, 
so that SageMaker does not inadvertently run your training code at the wrong point in execution.

For more on training environment variables, please visit https://github.com/aws/sagemaker-containers.
'''


import argparse
import os

import pandas as pd

import sklearn

from sklearn import linear_model

import numpy as np

import six
from six import StringIO, BytesIO

# from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_NPY
# Interesting fact: 
#   on SageMaker model training instance, py-sagemaker is not installed
# import sagemaker 

# matplotlib is not available 
# from matplotlib import pyplot as plt

from sklearn.externals import joblib
import json 

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


fn = 'sample_data.json'

MA_list = [50, 100, 200, 300, 400, 800, 1600]
intput_col = ["MA-{}".format(ma_lag) for ma_lag in MA_list[:-1]]
benchmark_col = 'MA-1600'


'''
The RealTimePredictor used by Scikit-learn in the SageMaker 
Python SDK serializes NumPy arrays to the NPY format by default, 
with Content-Type application/x-npy. The SageMaker Scikit-learn model server 
can deserialize NPY-formatted data (along with JSON and CSV data).
'''
def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    # print("request_body=",str(request_body))
    # print("np.load(StringIO(request_body))=",np.load(StringIO(request_body)))

    if request_content_type == "application/python-pickle":
        array = np.load(BytesIO((request_body)))
        # print("array=",array)
        return array
    elif request_content_type == 'application/json':
        jsondata = json.load(StringIO(request_body))
        normalized_data, benchmark_data = process_input_data(jsondata)
        # print("normalized_data=",normalized_data)
        return normalized_data, benchmark_data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise ValueError("{} not supported by script!".format(request_content_type))

def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        return worker.Response(json.dumps(prediction), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise ValueError("{} accept type is not supported by this script.".format(accept))

def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    normalized_data, benchmark_data = input_data
    
    prediction = model.predict(normalized_data)
    
    output = np.array(prediction) * np.array(benchmark_data)
    
    return {'prediction-base-time': str(normalized_data.index[-1]), 
            'predicted-value': output[-1]}

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def process_input_data(cmcjsondata, for_training = False):
    raw_data = pd.DataFrame(cmcjsondata)
    dat = pd.DataFrame(list(raw_data['price_usd']), columns=['timestamp', 'usd'])
    dat['dt_utc'] = pd.to_datetime(dat['timestamp']*1e6)
    dat = dat.set_index('dt_utc')
    resampled_data = dat['usd'].resample('30S').mean().interpolate('linear')

    feature = {}

    if for_training:
        Y = resampled_data.rolling(2880).median().shift(-2879) # next 24 hour
        feature['Y'] = Y

    for ma_lag in MA_list:
        feature["MA-{}".format(ma_lag)] = resampled_data.rolling(ma_lag).mean()

    data = pd.DataFrame(feature).dropna()
    benchmark_data = data[benchmark_col].copy()
    normalized_data = data.div(data[benchmark_col], axis=0)
    
    _col = intput_col + (['Y'] if for_training else [])

    return normalized_data[_col], benchmark_data
    
def run_training(args):

    with open(os.path.join(args.train, fn)) as fp:
        jsondata = json.load(fp)
    normalized_data, _ = process_input_data(jsondata, for_training = True)

    model = linear_model.Ridge()
    model.fit(normalized_data[intput_col], normalized_data['Y'])

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()

    run_training(args)

    # ... load from args.train and args.test, train a model, write model to args.model_dir.

