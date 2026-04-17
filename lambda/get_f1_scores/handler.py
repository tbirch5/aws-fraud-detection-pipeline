import json
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket = "amzn-models-bucket"
    key = "staging/evaluation.json"
    
    response = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read().decode('utf-8'))

    challenger_f1 = float(data.get('challenger_f1', 0))
    champion_f1 = float(data.get('champion_f1', 0))

    # First-run logic: if there is no champion yet, promote challenger
    if champion_f1 == 0:
        is_better = True
    else:
        is_better = challenger_f1 > champion_f1

    return {
        "is_better": is_better,
        "challenger_f1": challenger_f1,
        "champion_f1": champion_f1
    }