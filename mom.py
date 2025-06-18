import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import boto3


from dotenv import load_dotenv
load_dotenv()


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

bedrock_runtime = boto3.client(
    'bedrock-runtime',
    region_name=aws_default_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


# # Wait 60 seconds before connecting using these details, or login to https://console.neo4j.io to validate the Aura Instance is available
# NEO4J_URI=neo4j+s://edebba48.databases.neo4j.io
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=gjC2zNxRLqgZE3hdB4-UhNszVeorXezvL438kBWbOAk
# AURA_INSTANCEID=edebba48
# AURA_INSTANCENAME=Instance01

NEO4J_URI = "neo4j+ssc://edebba48.databases.neo4j.io"
NEO4J_AUTH = ("neo4j", "gjC2zNxRLqgZE3hdB4-UhNszVeorXezvL438kBWbOAk")
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "gjC2zNxRLqgZE3hdB4-UhNszVeorXezvL438kBWbOAk"

# # Create the driver instance
# driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# try:
#     driver.verify_connectivity()  # Check connection
#     print("Connected successfully!")
# except Exception as e:
#     print(f"Connection failed: {e}")
# finally:
#     driver.close()


class Neo4jDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_response(self, user_query, response_text, model_name):
        """Stores response as nodes in Neo4j with relationships."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (q:Query {text: $user_query})
                MERGE (r:Fact {text: $response_text, model: $model_name})
                MERGE (r)-[:BELONGS_TO]->(q)
                """,
                user_query=user_query,
                response_text=response_text,
                model_name=model_name,
            )

    def retrieve_relevant_facts(self, user_query):
        """Retrieves relevant stored response sections."""
        with self.driver.session() as session:
            results = session.run(
                """
                MATCH (q:Query)-[:BELONGS_TO]-(r:Fact)
                WHERE q.text CONTAINS $user_query
                RETURN r.text
                """,
                user_query=user_query,
            )
            facts = [record["r.text"] for record in results]
        return facts


def merge_responses(facts):
    """
    Merges multiple responses into a single refined response.
    """
    merged_response = []
    seen_sentences = set()

    for fact in facts:
        sentences = fact.split(". ")  # Split into sentences
        for sentence in sentences:
            if sentence and sentence not in seen_sentences:  # Avoid duplicates
                seen_sentences.add(sentence)
                merged_response.append(sentence)

    return ". ".join(merged_response)  # Reconstruct the final response


# Initialize Neo4j
neo4j_db = Neo4jDB(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)


def modelOfModels(user_query):
    """
    Generates responses using multiple LLM models and merges relevant sections into a single response.
    """
    response1 = llm_response(user_query, "anthropic.claude-3-5-sonnet-20240620-v1:0")
    response2 = llm_response(user_query, "anthropic.claude-3-sonnet-20240229-v1:0")

    if not response1 and not response2:
        return "No response generated."
    
    print(f"Response 1 = {response1} \n\n")
    print(f"Response 2 = {response2} \n\n")

    # Store responses in Neo4j
    if response1:
        neo4j_db.store_response(user_query, response1, "Claude-3.5")
    if response2:
        neo4j_db.store_response(user_query, response2, "Claude-3")

    # Retrieve relevant facts from stored responses
    relevant_facts = neo4j_db.retrieve_relevant_facts(user_query)

    # print("Relevant Facts --> \n")
    # print(relevant_facts)

    # Merge relevant facts into a refined response
    final_response = merge_responses(relevant_facts)

    # Store final response
    neo4j_db.store_response(user_query, final_response, "Merged-Response")

    return final_response


def llm_response(user_query, model_ID):
    prompt = (
        f"User Query : {user_query} "
    )

    response = bedrock_runtime.invoke_model(
        modelId=f"{model_ID}",
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




# Example usage
if __name__ == "__main__":
    user_query = "Mercedes - Company's story ?"
    final_answer = modelOfModels(user_query)
    print("\n\n Final Answer --> \n")
    print(final_answer)

    # Close Neo4j connection
    neo4j_db.close()







