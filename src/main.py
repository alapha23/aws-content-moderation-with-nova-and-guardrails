import os
from dotenv import load_dotenv
import boto3
import json
from botocore.exceptions import ClientError
from typing import Literal
import traceback
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Guardrail limits image file size up to 4 MB
MAX_IMAGE_SIZE = 4 * 1024 * 1024

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.getenv('AWS_SESSION_TOKEN', None)
region = os.getenv('REGION', 'us-east-1')
guardrail_identifier = os.getenv('GUARDRAIL_IDENTIFIER')
guardrail_version = os.getenv('GUARDRAIL_VERSION')
MODEL_ID = os.getenv('MODEL_ID')

# Create a Boto3 session and bedrock runtime
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=region
)

bedrock_runtime = session.client('bedrock-runtime')

def guard_text(guard_content: str):
    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_identifier,
        guardrailVersion=guardrail_version,
        source='INPUT',
        content=[
            {
                'text': {
                    'text': guard_content,
                    'qualifiers': [
                        'guard_content',
                    ]
                }
            },
        ]
    )
    return response

def guard_image(guard_blob, img_format: Literal['png', 'jpeg']):
    # Check image size before sending the request
    image_size = len(guard_blob)
    if image_size > MAX_IMAGE_SIZE:
        raise ValueError(f'Image size ({image_size} bytes) exceeds the maximum allowed size ({MAX_IMAGE_SIZE} bytes)')

    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_identifier,
        guardrailVersion=guardrail_version,
        source='INPUT',
        content=[
            {
                'image': {
                    'format': img_format,
                    'source': {
                        'bytes': guard_blob,
                    }
                }
            },
        ]
    )
    return response

SYSTEM_PROMPT = """;; You are a Guardrails judge, making judgements based on the given functions
;; First, define our main guardrailing check function.
;; It strictly returns only "GUARDRAIL_INTERVENED" or "NONE".
(defun guardrail-check (content)
  "Evaluate CONTENT against guardrailing rules, returning only:
   - \"GUARDRAIL_INTERVENED\" if any rule is violated,
   - \"NONE\" otherwise.
   No other values are returned."
  (cond
    ;; Hate speech or discrimination
    ((hate-speech-or-discrimination-p content)
     "GUARDRAIL_INTERVENED")

    ;; Explicit sexual content
    ((explicit-sexual-content-p content)
     "GUARDRAIL_INTERVENED")

    ;; Illegal activities
    ((illegal-activities-p content)
     "GUARDRAIL_INTERVENED")

    ;; Violence or gore
    ((violence-or-gore-p content)
     "GUARDRAIL_INTERVENED")

    ;; Misinformation or conspiracy theories
    ((misinformation-or-conspiracy-p content)
     "GUARDRAIL_INTERVENED")

    ;; Harassment or bullying
    ((harassment-or-bullying-p content)
     "GUARDRAIL_INTERVENED")

    ;; Sensitive personal information
    ((sensitive-personal-info-p content)
     "GUARDRAIL_INTERVENED")

    ;; Spam or commercial content
    ((spam-or-commercial-content-p content)
     "GUARDRAIL_INTERVENED")

    ;; Impersonation
    ((impersonation-p content)
     "GUARDRAIL_INTERVENED")

    ;; Intellectual property infringement
    ((intellectual-property-infringement-p content)
     "GUARDRAIL_INTERVENED")

    ;; If none of the rules are violated:
    (t "NONE")))

;; Next, define a function that repeats guardrail-check until
;; we get either "GUARDRAIL_INTERVENED" or "NONE".
(defun guardrail-check-with-loop (content)
  "Repeatedly call `guardrail-check` on CONTENT until the result
   is either \"GUARDRAIL_INTERVENED\" or \"NONE\". Returns the result."
  (loop
     with result = (guardrail-check content)
     while (not (or (string= result "GUARDRAIL_INTERVENED")
                    (string= result "NONE")))
     do (setf result (guardrail-check content))
     finally (return result)))"""

def guard_nova_text(guard_content: str):
    # Your existing system prompt & messages
    system = [{ 
        'text': SYSTEM_PROMPT
    }]

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'text': (
                        'In your answer, include only GUARDRAIL_INTERVENED or NONE in plain text. '
                        'Do not include any other character.\n'
                        'content= "' + guard_content + '"\n'
                        ' (guardrail-check-with-loop content)'
                    )
                }
            ],
        }
    ]

    inf_params = {'maxTokens': 300, 'topP': 1.0, 'temperature': 0.0}
    additionalModelRequestFields = {
        'inferenceConfig': {
             'topK': 1
        }
    }
    
    while True:
        try:
            model_response = bedrock_runtime.converse(
                modelId=MODEL_ID, 
                messages=messages, 
                system=system, 
                inferenceConfig=inf_params,
                additionalModelRequestFields=additionalModelRequestFields
            )
            
            logging.debug('\n[Full Response]')
            logging.debug(json.dumps(model_response, indent=2))
            
            # Extract just the text from the model's response
            result_text = model_response['output']['message']['content'][0]['text']
            logging.debug('\n[Response Content Text]')
            logging.debug(result_text)

            # Check if the text is strictly one of the two valid outputs
            if result_text in ('GUARDRAIL_INTERVENED', 'NONE'):
                return result_text
            else:
                # If it's anything else, re-try
                logging.error('Unexpected response. Retrying...\n')
        
        except Exception as e:
            # Log the error, log the stack trace, then raise the error
            logging.error('An error occurred while requesting the model:\n', str(e))
            traceback.print_exc()
            raise e

def guard_nova_image(guard_blob, img_format: Literal['png', 'jpeg']):
    # Your existing system prompt & messages
    system = [{ 
        'text': SYSTEM_PROMPT
    }]

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'text': (
                        'In your answer, include only GUARDRAIL_INTERVENED or NONE in plain text. '
                        'Do not include any other character.\n'
                        'content= image\n'
                        '(guardrail-check-with-loop content)'
                    )
                },
                {
                        'image': {
                            'format': 'jpeg',
                            'source': {
                                'bytes': guard_blob
                            }
                        }
                }
            ],
        }
    ]

    inf_params = {'maxTokens': 300, 'topP': 1.0, 'temperature': 0.0}
    additionalModelRequestFields = {
        'inferenceConfig': {
             'topK': 1
        }
    }
    
    while True:
        try:
            model_response = bedrock_runtime.converse(
                modelId=MODEL_ID, 
                messages=messages, 
                system=system, 
                inferenceConfig=inf_params,
                additionalModelRequestFields=additionalModelRequestFields
            )
            
            logging.debug('\n[Full Response]')
            logging.debug(json.dumps(model_response, indent=2))
            
            # Extract just the text from the model's response
            result_text = model_response['output']['message']['content'][0]['text']
            logging.debug('\n[Response Content Text]')
            logging.debug(result_text)

            # Check if the text is strictly one of the two valid outputs
            if result_text in ('GUARDRAIL_INTERVENED', 'NONE'):
                return result_text
            else:
                # If it's anything else, re-try
                logging.error('Unexpected response. Retrying...\n')
        
        except Exception as e:
            # Log the error, log the stack trace, then raise the error
            logging.error('An error occurred while requesting the model:\n', str(e))
            traceback.print_exc()
            raise e

def handle_image(decision, img_path):
    """Reads image, decides extension, and updates decision dict accordingly."""
    if not os.path.exists(img_path):
        logging.warning('File does not exist: %s; proceed without image', img_path)
        return

    with open(img_path, 'rb') as f:
        img_blob = f.read()

    # Determine extension: 'jpeg' for JPG/JPEG, 'png' for PNG,
    # otherwise None (unsupported)
    ext = (
        'jpeg' if ('jpg' in img_path.lower() or 'jpeg' in img_path.lower())
        else 'png' if 'png' in img_path.lower()
        else None
    )

    if ext is None:
        logging.warning('warning: image type not supported; proceed without image')
        return

    # Call the guard functions (each returns 'GUARDRAIL_INTERVENED' or 'NONE')
    img_result = guard_image(img_blob, ext)
    if img_result == 'GUARDRAIL_INTERVENED':
        decision['guardrails'] = 'GUARDRAIL_INTERVENED'

    nova_img_result = guard_nova_image(img_blob, ext)
    if nova_img_result == 'GUARDRAIL_INTERVENED':
        decision['nova'] = 'GUARDRAIL_INTERVENED'

def guard(guard_content=None, img_path=None):
    """Decides if any guard (text/image) triggers an intervention, storing outcome in decision."""
    decision = {'guardrails': 'NONE', 'nova': 'NONE'}

    # 1. Check text (if provided)
    if guard_content is not None:
        text_result = guard_text(guard_content)
        if text_result == 'GUARDRAIL_INTERVENED':
            decision['guardrails'] = 'GUARDRAIL_INTERVENED'
    
        nova_text_result = guard_nova_text(guard_content)
        if nova_text_result == 'GUARDRAIL_INTERVENED':
            decision['nova'] = 'GUARDRAIL_INTERVENED'

    # 2. Check image (if provided)
    if img_path is not None:
        handle_image(decision, img_path)

    return decision

def main():
    """
    Add your test logic, such as
    decision = guard('hihi')
    logging.info(decision)
    decision = guard('FuckFuck')
    logging.info(decision)
    decision = guard('./1.jpg')
    logging.info(decision)
    decision = guard('./cancer.jpeg')
    logging.info(decision)
    decision = guard('fuck fuck fuck','./cancer.jpeg')
    logging.info(decision)
    """
    pass

if __name__ == '__main__':
    main()
