from sagemaker.sklearn import SKLearn
# Initialise SDK
sklearn_estimator = SKLearn(
        entry_point='train_and_deploy.py',
        role = 'arn:aws:iam::<your-sagemaker-role>',
        # train_instance_type='ml.m4.xlarge',
        train_instance_type='local',
        output_path='s3://<path-to-output-dir>/',
        hyperparameters = {
            'sagemaker_submit_directory': 's3://<path-to-sagemaker_submit_directory>'
        },
        code_location='s3://<path-to-code_location>',
        framework_version='0.20.0'
)
# Run model training job
sklearn_estimator.fit({'train': 's3://<path-to-training-data-dir>'})
# Deploy trained model to an endpoint
predictor = sklearn_estimator.deploy(
        # instance_type='ml.t2.medium',
        instance_type='local',
        initial_instance_count=1,
        endpoint_name='<your-end-point-name>',
)