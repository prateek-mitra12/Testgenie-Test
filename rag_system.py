import os
import warnings
import boto3
import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader
import io
import json
import re
from sklearn.preprocessing import normalize
#import docx


nltk.download('punkt')
nltk.download('wordnet')

from dotenv import load_dotenv
load_dotenv()

# Suppress warnings related to protected namespaces
warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")
warnings.filterwarnings("ignore", message="Field .* in BedrockBase has conflict with protected namespace .*")
warnings.filterwarnings("ignore", message="Unsupported Windows version.*")

# Custom Pydantic class to handle the warning
class BedrockBase(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())

os.environ['USER_AGENT'] = 'YourAppName/1.0'

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name=os.getenv("AWS_DEFAULT_REGION"))

def setup_rag_system(uploaded_doc_text, uploaded_files):
    if uploaded_doc_text == None:
        return ""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.create_documents([uploaded_doc_text])

    splits_embeddings = []

    for split in splits:
        splits_embeddings.append(get_bedrock_embeddings(split))

    bucket_name = "test-genie"
    prefix = "rag_documents"

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' not in response:
        print("No files found in the specified bucket or prefix !!")
        return
    
    rag_prompt = ""

    for obj in response['Contents']:
        file_key = obj['Key']

        if file_key in uploaded_files:
            print(f"File {file_key} already uploaded !!")
            continue

        print(f"\n\nReading content of: {file_key}")

        file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = file_obj['Body'].read()

        # Handling different file types
        if file_key.endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(file_content))
                for page_number, page in enumerate(reader.pages):
                    print(f"\n--- Page {page_number + 1} ---")
                    page_text = page.extract_text()
                    print(page_text)

                    chunks = text_splitter.create_documents([page_text])

                    # Include chunk in rag prompt if already not present in the uploaded doc text
                    for chunk in chunks:
                        most_similar_split = ""
                        most_similar_split_similarity = -1

                        chunk_embedding = get_bedrock_embeddings(chunk)

                        for i, split in enumerate(splits):
                            similarity = compute_similarity(split, chunk, splits_embeddings[i], chunk_embedding)
                            if similarity > most_similar_split_similarity:
                                most_similar_split_similarity = similarity
                                most_similar_split = split
                        
                        if most_similar_split_similarity < 0.6:
                            rag_prompt += chunk
            except Exception as e:
                print(f"Error reading PDF {file_key}: {e}")
        # elif file_key.endswith(".doc") or file_key.endswith(".docx"):
        #     try:
        #         doc = docx.Document(io.BytesIO(file_content))
        #         text = "\n".join([para.text for para in doc.paragraphs])

        #         print(text)

        #         chunks = text_splitter.create_documents([text])

                # # Include chunk in rag prompt if already not present in the uploaded doc text
                # for chunk in chunks:
                #     most_similar_split = ""
                #     most_similar_split_similarity = -1

                #     chunk_embedding = get_bedrock_embeddings(chunk)

                #     for i, split in enumerate(splits):
                #         similarity = compute_similarity(split, chunk, splits_embeddings[i], chunk_embedding)
                #         if similarity > most_similar_split_similarity:
                #             most_similar_split_similarity = similarity
            #                 most_similar_split = split
                    
            #         if most_similar_split_similarity < 0.6:
            #             rag_prompt += chunk
            # except Exception as e:
            #     print(f"Error reading DOC: {e}")
        elif file_key.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(file_content))
                text = "\n".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))

                print(text)

                chunks = text_splitter.create_documents([text])

                # Include chunk in rag prompt if already not present in the uploaded doc text
                for chunk in chunks:
                    most_similar_split = ""
                    most_similar_split_similarity = -1

                    chunk_embedding = get_bedrock_embeddings(chunk)

                    for i, split in enumerate(splits):
                        similarity = compute_similarity(split, chunk, splits_embeddings[i], chunk_embedding)
                        if similarity > most_similar_split_similarity:
                            most_similar_split_similarity = similarity
                            most_similar_split = split
                    
                    if most_similar_split_similarity < 0.6:
                        rag_prompt += chunk
            except Exception as e:
                print(f"Error reading CSV: {e}")
        elif file_key.endswith(".xlsx"):
            try:
                df = pd.read_excel(io.BytesIO(file_content))
                text = "\n".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))

                print(text)

                chunks = text_splitter.create_documents([text])

                # Include chunk in rag prompt if already not present in the uploaded doc text
                for chunk in chunks:
                    most_similar_split = ""
                    most_similar_split_similarity = -1

                    chunk_embedding = get_bedrock_embeddings(chunk)

                    for i, split in enumerate(splits):
                        similarity = compute_similarity(split, chunk, splits_embeddings[i], chunk_embedding)
                        if similarity > most_similar_split_similarity:
                            most_similar_split_similarity = similarity
                            most_similar_split = split
                    
                    if most_similar_split_similarity < 0.6:
                        rag_prompt += chunk
            except Exception as e:
                print(f"Error reading XLSX: {e}")

    return rag_prompt 


# session = boto3.Session(profile_name = "genai-dev1-admin")
# bedrock_runtime = session.client("bedrock-runtime", region_name = "ap-southeast-2")

bedrock_runtime = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)
    

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = cleaned_text.lower()
    tokens = word_tokenize(cleaned_text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text


def get_bedrock_embeddings(text):
    try:
        response = bedrock_runtime.invoke_model(
            # modelId="amazon.titan-embed-text-v2:0",
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )

        response_body = json.loads(response['body'].read())

        if 'embedding' in response_body:
            return response_body['embedding']
        else:
            raise ValueError(f"Unexpected response from Bedrock: {response_body}")
    except Exception as e:
        print(f"Error fetching embedding for text: {e}")
        return None
    

def jaccard_similarity(sentence1, sentence2):
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return float(len(intersection)) / len(union) if union else 0.0


def compute_similarity(sentence1, sentence2, embedding1, embedding2):
    if sentence1:
        sentence1 = preprocess_text(sentence1)
    if sentence2:
        sentence2 = preprocess_text(sentence2)

    if sentence1 == "" and sentence2 == "":
        return 1.0

    if sentence1 and sentence2:
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)

        # Normalize embeddings before computing cosine similarity
        embedding1 = normalize(embedding1, axis=1)
        embedding2 = normalize(embedding2, axis=1)

        embedding_similarity = cosine_similarity(embedding1, embedding2)[0][0]
        lexical_similarity = jaccard_similarity(sentence1, sentence2)

        combined_similarity = 0.85*embedding_similarity + 0.15*lexical_similarity

        return combined_similarity

    return 0.0


def ask_chat(text):
    prompt = (
        
        f"User Query : {text} "
        
    )

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
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









# import os
# import warnings
# import boto3
# import pandas as pd
# from pydantic import BaseModel, ConfigDict
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnableLambda
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()

# # Suppress warnings related to protected namespaces
# warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version")
# warnings.filterwarnings("ignore", message="Field .* in BedrockBase has conflict with protected namespace .*")
# warnings.filterwarnings("ignore", message="Unsupported Windows version.*")

# # Custom Pydantic class to handle the warning
# class BedrockBase(BaseModel):
#     model_config  = ConfigDict(protected_namespaces=())

# os.environ['USER_AGENT'] = 'YourAppName/1.0'

# embeddings = BedrockEmbeddings(
#     model_id="amazon.titan-embed-text-v1",
#     region_name=os.getenv("AWS_DEFAULT_REGION")
# )

# prev_file_name = ""

# def setup_rag_system(documents, file_name):
#     global prev_file_name

#     if documents == None:
#         return ""

#     # Split
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.create_documents([documents])


#     # Load the Chroma database from disk
#     chroma_db = Chroma(persist_directory="Chroma", 
#                     embedding_function=embeddings,
#                     collection_name="doc_collection")

#     # Get the collection from the Chroma database
#     collection = chroma_db.get()

#     # st.write("prev file: ", prev_file_name)
#     # st.write("current file: ", file_name)

#     # If the collection is empty, create a new one
#     if len(collection['ids']) == 0 or file_name is not prev_file_name:
#         prev_file_name = file_name

#         # Create a new Chroma database from the documents
#         chroma_db = Chroma.from_documents(
#             documents=splits, 
#             embedding=embeddings,
#             collection_name="doc_collection",
#             persist_directory="Chroma"
#         )

#         # Save the Chroma database to disk
#         # chroma_db.persist()

#     # Return the retriever
#     # retriever = [chroma_db.as_retriever(search_type="similarity_score_threshold",
#     #                                     search_kwargs={'score_threshold': 0, 'k': 5}
#     #                                     ), 
#     #     chroma_db]
#     # return retriever

#     return chroma_db



