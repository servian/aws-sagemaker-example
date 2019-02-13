import json
import boto3
import time
from datetime import datetime
from botocore.vendored import requests

src_path = 's3://<path-to-source-code-train_and_deploy.py>.tar.gz'
role_arn = 'arn:aws:iam::<your-sagemaker-role>'
endpoint_name='<your-endpoint-name>'


def retrain_the_model():
    now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    training_job_name = f'<your-job-prefix>-{now_str}'
    sm = boto3.client('sagemaker')
    resp = sm.create_training_job(
            TrainingJobName = training_job_name, 
            AlgorithmSpecification={
                'TrainingInputMode': 'File',
                'TrainingImage': '783357654285.dkr.ecr.<your-region>.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3',
            }, 
            RoleArn=role_arn,
            InputDataConfig=[
                                {
                                    'ChannelName': 'train',
                                    'DataSource': {
                                        'S3DataSource': {
                                            'S3DataType': 'S3Prefix',
                                            'S3Uri': 's3://<path-to-training-data-dir>/',
                                            'S3DataDistributionType': 'FullyReplicated',
                                        }
                                    },
                                },
                            ], 
            OutputDataConfig={
                                'S3OutputPath': 's3://<path-to-output_dir>/'
                            },
            ResourceConfig={
                            'InstanceType': 'ml.m4.xlarge',
                            'InstanceCount': 1,
                            'VolumeSizeInGB': 30,
                        }, 
            StoppingCondition={
                                'MaxRuntimeInSeconds': 600
                            },
            HyperParameters={
                'sagemaker_submit_directory': 's3://<path-to-sagemaker_submit_directory>/',
                'sagemaker_program' : "train_and_deploy.py",
                'sagemaker_region': "<your-region>",
                'sagemaker_job_name': training_job_name,
                'sagemaker_submit_directory': src_path
            },
            Tags=[]
    )
    


def deploy_model(training_job_name):
    sm = boto3.client('sagemaker')
    model = sm.create_model(
        ModelName=training_job_name,
        PrimaryContainer={
            'ContainerHostname': 'model-Container',
            'Image': '783357654285.dkr.ecr.<your-region>.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3',
            'ModelDataUrl': f's3://<path-to-training-job-output_dir>/{training_job_name}/output/model.tar.gz',
            'Environment': {
                'SAGEMAKER_PROGRAM': 'train_and_deploy.py',
                'SAGEMAKER_REGION':'<your-region>',
                'SAGEMAKER_SUBMIT_DIRECTORY': src_path
    
            },
        },
        ExecutionRoleArn=role_arn,
    )
    endpoint_config = sm.create_endpoint_config(
        EndpointConfigName=training_job_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': training_job_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
            },
        ],
    )
    
    sm.update_endpoint(EndpointName=endpoint_name,
                   EndpointConfigName=training_job_name)

def make_prediction():
    data = download_latest_price()
    runtime = boto3.Session().client(service_name='runtime.sagemaker',region_name='<your-region>')

    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                        ContentType='application/json',
                                        Body=json.dumps(data))

    result = json.loads(response['Body'].read().decode())
    
    return result

def download_latest_price():
    # data for the next 2 days
    now = time.time()
    r = requests.get(f"https://graphs2.coinmarketcap.com/currencies/bitcoin/{int(1000*(now-172800))}/{int(1000*now)}/")

    data = r.json()
    
    s3 = boto3.resource('s3')
    s3object = s3.Object('aws-ip-sagemaker-project', f'training-data/data-{datetime.utcnow().strftime("%Y-%m-%d")}.json')
    
    s3object.put(
        Body=(bytes(json.dumps(data).encode('UTF-8')))
    )
    return data

def handle_s3_event(s3):
    print("s3=",s3)
    bucket = s3['bucket']['name']
    fn = s3['object']['key']
    jobid = fn.split("/")[-3]
    print("jobid=",jobid)
    return deploy_model(jobid)

def lambda_handler(event, context):
    response = event
    
    # Lambda event
    if 'Records' in event:
        for records in event['Records']:
            if 's3' in records:
                handle_s3_event(records['s3'])
    else:
        pass
    # print(retrain_the_model())
    
    if 'task' in event:
        if event['task'] == 'retrain':
            # start retrain the model
            download_latest_price()
            retrain_the_model()
            response = "OK"
        if event['task'] == 'prediction':
            response = make_prediction()

    return {
        'statusCode': 200,
        'body': response,
    }
