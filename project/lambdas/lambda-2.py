"""
inferSerializedImageData: a lambda function for invoking the model endpoint and returning inferences
"""
import json
import base64
import boto3

# Name of the deployed model
ENDPOINT = "image-classification-2024-11-05-15-15-44-407"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/x-image', Body=image)
    inferences = response['Body'].read().decode('utf-8')
    event['body']["inferences"] = [float(x) for x in inferences[1:-1].split(',')]
    
    # We return the data back to the Step Function    
    return {
        'statusCode': 200,
        'body': {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['body']['inferences']
        }
    }