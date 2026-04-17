import json
import boto3
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    # ──────────────────────────────────────────────────────────
    #  1) configuration & inputs
    # ──────────────────────────────────────────────────────────
    sm = boto3.client("sagemaker")
    model_artifact_s3_uri = event.get('challenger_model_uri') 

    if not model_artifact_s3_uri:
        job_id = event.get('id')
        if job_id:
            model_artifact_s3_uri = f"s3://amzn-models-bucket/staging/{job_id}-training/output/model.tar.gz"
    
    if not model_artifact_s3_uri:
        raise ValueError("event must contain key with the S3 URI of the model artifact or an 'id'")

    execution_role_arn    = "arn:aws:iam::025410243600:role/service-role/SageMaker-SageMakerIAMRole"
    model_name            = "fraudv10-production-model"
    ep_config_name        = "fraudv10-production-endpoint-config"
    endpoint_name         = "fraudv10-production-endpoint"

    # ──────────────────────────────────────────────────────────
    #  2) delete old model
    # ──────────────────────────────────────────────────────────
    try:
        sm.delete_model(ModelName=model_name)
        print(f"[INFO] deleted previous model: {model_name}")
    except ClientError as e:
        print(f"[INFO] no existing model to delete or deletion failed")

    # ──────────────────────────────────────────────────────────
    #  3) create new model
    # ──────────────────────────────────────────────────────────
    try:
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=execution_role_arn,
            PrimaryContainer={
                "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
                "ModelDataUrl": model_artifact_s3_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM":          "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model"
                },
            },
        )
        print(f"[INFO] created model: {model_name}")
    except ClientError as e:
        print(f"[ERROR] create_model failed: {e.response['Error']['Message']}")
        raise

    # ──────────────────────────────────────────────────────────
    #  4) delete old endpoint-config
    # ──────────────────────────────────────────────────────────
    try:
        sm.delete_endpoint_config(EndpointConfigName=ep_config_name)
        print(f"[INFO] deleted previous endpoint-config: {ep_config_name}")
    except ClientError as e:
        print(f"[INFO] no endpoint-config to delete")

    # ──────────────────────────────────────────────────────────
    #  5) create new endpoint-config
    # ──────────────────────────────────────────────────────────
    try:
        sm.create_endpoint_config(
            EndpointConfigName=ep_config_name,
            ProductionVariants=[{
                "VariantName":          "AllTraffic",
                "ModelName":            model_name,
                "InitialInstanceCount": 1,
                "InstanceType":         "ml.m5.large",
                "InitialVariantWeight": 1,
            }],
        )
        print(f"[INFO] created endpoint-config: {ep_config_name}")
    except ClientError as e:
        print(f"[ERROR] create_endpoint_config failed: {e.response['Error']['Message']}")
        raise

    # ──────────────────────────────────────────────────────────
    #  6) update if exists -- else create endpoint
    # ──────────────────────────────────────────────────────────
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"[INFO] updating endpoint: {endpoint_name}")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=ep_config_name,
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ValidationException" or code == "404": 
            print(f"[INFO] endpoint not found; creating: {endpoint_name}")
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=ep_config_name,
            )
        else:
            print(f"[ERROR] describe/update endpoint failed: {e.response['Error']['Message']}")
            raise

    return {
        "statusCode": 200,
        "body": json.dumps("Deployment request accepted.")
    }