import json

THRESHOLD = .63

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['body']['inferences'] ## TODO: fill in
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = meets_threshold = any([x > THRESHOLD for x in inferences]) ## TODO: fill in
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event['body'])
    }
