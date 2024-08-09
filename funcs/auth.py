import boto3
from funcs.helper_functions import get_or_create_env_var

client_id = get_or_create_env_var('AWS_CLIENT_ID', 'aws_client_placeholder') # This client id is borrowed from async gradio app client
print(f'The value of AWS_CLIENT_ID is {client_id}')

user_pool_id = get_or_create_env_var('AWS_USER_POOL_ID', 'aws_user_pool_placeholder')
print(f'The value of AWS_USER_POOL_ID is {user_pool_id}')

def authenticate_user(username, password, user_pool_id=user_pool_id, client_id=client_id):
    """Authenticates a user against an AWS Cognito user pool.

    Args:
        user_pool_id (str): The ID of the Cognito user pool.
        client_id (str): The ID of the Cognito user pool client.
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """

    client = boto3.client('cognito-idp')  # Cognito Identity Provider client

    try:
        response = client.initiate_auth(
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
            },
            ClientId=client_id
        )

        # If successful, you'll receive an AuthenticationResult in the response
        if response.get('AuthenticationResult'):
            return True
        else:
            return False

    except client.exceptions.NotAuthorizedException:
        return False
    except client.exceptions.UserNotFoundException:
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

def download_file_from_s3(bucket_name, key, local_file_path):

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_file_path)
    print(f"File downloaded from S3: s3://{bucket_name}/{key} to {local_file_path}")