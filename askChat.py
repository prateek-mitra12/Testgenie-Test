import json
import os
import boto3
import traceback


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

bedrock_runtime = boto3.client(
    'bedrock-runtime',
    region_name=aws_default_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


def ask_chat(user_query, model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"):
    prompt = (
        f"User Query : {user_query} "
    )

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="*/*",
        body= json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200000,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        }
                    ]
                }
            ]
        })
    )

    if response.get("body"):
        response_body = response.get("body")
        response_data = json.loads(response_body.read())
        completion = response_data.get("content")[0].get("text")

        if completion:
            return completion
        else:
            return None
    else:
        return None
    

def get_llm_response(user_query, model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"):
    prompt = (
        f"User Query : {user_query} "
    )

    try:
        # Invoke the Anthropic Claude Sonnet model
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id,
            contentType="application/json",
            accept="*/*",
            body= json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200000,
                "top_k": 250,
                "stop_sequences": [],
                "temperature": 1,
                "top_p": 0.999,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt}"
                            }
                        ]
                    }
                ]
            })
        )

        # Get the event stream and process it
        event_stream = response.get('body', {})
        for event in event_stream:
            chunk = event.get('chunk')
            if chunk:
                message = json.loads(chunk.get("bytes").decode())
                if message['type'] == "content_block_delta":
                    yield message['delta']['text'] or ""
                elif message['type'] == "message_stop":
                    return "\n"
    except Exception as e:
        print(traceback.format_exc())
        return f"Error: {str(e)}"

