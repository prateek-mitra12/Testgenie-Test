import os
#from routellm.controller import Controller
import boto3


from dotenv import load_dotenv
load_dotenv()


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
os.environ["AWS_REGION_NAME"] = aws_default_region


def getRoutingModel(user_query):
    bedrock_runtime = boto3.client(
        'bedrock-runtime',
        region_name=aws_default_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    model_list = ["anthropic.claude-3-haiku-20240307-v1:0",
                "amazon.titan-text-premier-v1:0",
                "anthropic.claude-instant-v1",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                # "meta.llama2-70b-chat-v1",
                "anthropic.claude-3-sonnet-20240229-v1:0"]
    
    most_optimized_model = model_list[0]

    for i in range(len(model_list)-1):
        model_option_1 = most_optimized_model
        model_option_2 = model_list[i+1]

        client = Controller(
            routers=["mf"],
            strong_model=model_option_1,
            weak_model=model_option_2,
        )

        response = client.chat.completions.create(
            # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
            model="router-mf-0.11593",
            messages=[
                {
                    "role": "user", 
                    "content": f"{user_query}"
                }
            ]
        )

        print(f"Winning Model = {response.model}")

        most_optimized_model = response.model
    
    return most_optimized_model


model_id_name_mapping = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0" : "Claude 3.5 Sonnet",
    "anthropic.claude-3-sonnet-20240229-v1:0" : "Claude 3 Sonnet",
    "amazon.titan-text-premier-v1:0" : "Amazon TitanText Premier",
    "anthropic.claude-3-haiku-20240307-v1:0" : "Claude 3 Haiku",
    "meta.llama2-70b-chat-v1" : "LLaMA 2",
    "anthropic.claude-instant-v1" : "Claude Instant"
}



# # Example Response

# getRoutingModel("What is Computer")

# ModelResponse(id='chatcmpl-8f1490a8-f32f-47ef-9b76-fca53363137b', created=1739981021, 
#               model='anthropic.claude-instant-v1', object='chat.completion', system_fingerprint=None, 
#               choices=[Choices(finish_reason='stop', index=0, 
#                                message=Message(content='A computer is an electronic device that can be programmed to receive, process, store, and retrieve data and information. Here are some key things to know about computers:\n\n- Hardware: The physical parts of a computer system including internal components like the processor (CPU), memory (RAM), storage devices (hard drives, SSDs), graphics cards, network cards, etc. as well as peripheral devices like keyboards, mice, monitors, printers, etc.\n\n- Software: Programs and applications that tell the hardware how to function and carry out tasks. Examples include operating systems like Windows or macOS, productivity apps, web browsers, games, etc. \n\n- Processor: The central processing unit (CPU) that carries out basic arithmetic, logical, and input/output operations of the computer by interpreting and executing instructions from programs/software. \n\n- Memory: Components like RAM (random access memory) that temporarily stores data and programs that are being actively worked on by the CPU. Data stored in memory is lost when the computer is turned off.\n\n- Storage: Permanent storage devices like hard disk drives or solid state drives that can retain data and files even when the computer is turned off. \n\n- Input/output: Devices like keyboards, mice, touchscreens, microphones, cameras that allow users to input data and instructions. Displays, speakers and printers allow computers to output information.\n\n- Operating system: System software that manages all the basic tasks like files, memory, processors, inputs/outputs and acts as an interface between the user and the computer hardware.\n\nSo in summary, a computer is an electronic device that processes data using hardware and software to receive, process, store and output useful information.', role='assistant', tool_calls=None, function_call=None))], 
#                                usage=Usage(completion_tokens=354, prompt_tokens=12, total_tokens=366, completion_tokens_details=None, prompt_tokens_details=None))


# # Testing
# Write KMP algorithm code and Sieve of eratosthenes code. - routed to Claude 3.5 Sonnet
# What is Computer ? - routed to Claude Instant





########################################################################################################################

# # RouteLLM using Bedrock Prompt Router

# import json
# import boto3
# import os

# from dotenv import load_dotenv
# load_dotenv()


# aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
# aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# aws_default_region = os.getenv("AWS_DEFAULT_REGION")

# bedrock_runtime = boto3.client(
#     'bedrock-runtime',
#     region_name=aws_default_region,
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key
# )


# # Set your prompt router ARN
# MODEL_ID = "arn:aws:bedrock:us-east-1:058264125602:default-prompt-router/anthropic.claude:1"

# # User message to be processed
# user_message = "Tell me about Amazon Bedrock in less than 100 words."

# # Prepare messages for the API call
# messages = [
#     {"role": "user", 
#      "content": [{"text": user_message}]
#      }
# ]

# # Invoke the model using the prompt router
# streaming_response = bedrock_runtime.converse_stream(
#     modelId=MODEL_ID,
#     messages=messages,
# )

# # Process and print the response
# for chunk in streaming_response["stream"]:
#     if "contentBlockDelta" in chunk:
#         text = chunk["contentBlockDelta"]["delta"]["text"]
#         print(text, end="")
#     if "messageStop" in chunk:
#         print()
#     if "metadata" in chunk:
#         if "trace" in chunk["metadata"]:
#             print(json.dumps(chunk['metadata']['trace'], indent=2))


########################################################################################################################


