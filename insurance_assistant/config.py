import os
from dotenv import load_dotenv

load_dotenv()

## ENV

groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

## Data Path

data_path = os.path.join("data","apex_health_insurance.txt")

## Vector Store

vector_store_path = os.path.join("data","faiss_index")

## Models
# LLM and Embedding Models 

llm_model_name="openai/gpt-oss-120b"

embedding_model_name="jina-embeddings-v2-base-en"

## Chunks / Text Splitting Configs

chunk_size=1000
chunk_overlap=150

## Retreival Result

top_k_results = 5

## System Prompt

# system_prompt="""
#         You are an assistant for answering questions about insurance policies.
#         Answer the question using only the provided context.
#         Politely response to questions about who you or and what you do
#         If the answer is not available in the context, say that you don't the answer for this particular question based on the provided information.
#         Keep your response clear, accurate, and concise without making assumptions.
#         """

system_prompt = """
    You are an assistant for answering questions about insurance policies.
    Respond politely to greetings and questions about who you are.
    For policy questions, answer using only the provided context; if the information is not in the context, say that you do not know.
    Keep your response clear, accurate, and concise without making assumptions.
    """

def check_api_keys() -> None:
    if not groq_key:
        raise ValueError("Missing GROQ API KEY. Please add it in config file")
    if not jina_key:
        raise ValueError("Missing JINA API KEY. Please add it in config file")

