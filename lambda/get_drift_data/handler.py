import json
import boto3
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        bucket = "amzn-models-bucket"
        key = "staging/drift_result.json"
        logger.info(f"Fetching drift result from s3://{bucket}/{key}")

        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')

        data = json.loads(content)
        
        logger.info("Parsed JSON content: %s", data)
        return data

    except Exception as e:
        logger.error("Error processing the S3 JSON: %s", str(e))
        raise e