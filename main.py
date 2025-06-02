import os
import pandas as pd
import logging
# from dotenv import load_dotenv
from llama_index.llms.lmstudio import LMStudio
from llama_index.core.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import multiprocessing

# # Load environment variables
# load_dotenv()
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e4nXOjJwPF5WMx_l3RTaqIR1uc2E9-OStA8zlNenApY"
QDRANT_URL = "https://4e5a634d-962c-44f3-8070-0ecc660fa546.eu-west-2-0.aws.cloud.qdrant.io:6333"

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log"), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
 
# Constants
COLLECTION_NAME = "llama-llm-multi-context"

# Define available models
llms = [
    {
        "name": "openchat-3.5-0106",
        "llm": LMStudio(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_name="openchat-3.5-0106",
            temperature=0,
            request_timeout=1000,
            max_tokens=2000,
            chat_mode=False
        )
    },
    {
        "name": "deepseek-r1-distill-llama-8b",
        "llm": LMStudio(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_name="deepseek-r1-distill-llama-8b",
            temperature=0,
            request_timeout=1000,
            max_tokens=2000,
            chat_mode=False
        )
    },
    {
        "name": "meta-llama-3.1-8b-instruct",
        "llm": LMStudio(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_name="meta-llama-3.1-8b-instruct",
            temperature=0,
            request_timeout=1000,
            max_tokens=2000,
            chat_mode=False
        )
    },
    {
        "name": "mistral-7b-instruct-v0.3",
        "llm": LMStudio(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_name="mistral-7b-instruct-v0.3",
            temperature=0,
            request_timeout=1000,
            max_tokens=2000,
            chat_mode=False
        )
    }
]

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="all-mpnet-base-v2")

# Load and validate CSV
df = pd.read_csv("data/data_cleaned.csv")
if not {"Model Name","Capabilities", "Description"}.issubset(df.columns):
    raise ValueError("CSV must contain 'Capabilities' and 'Description' columns.")

# Convert rows into documents
documents = [
    Document(
        text=f"Capabilities: {str(row['Capabilities'])}\nDescription: {str(row['Description'])}",
        metadata=row.to_dict()
    )
    for _, row in df.iterrows()
]

# Setup Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Setup vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME, enable_hybrid=True)
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, embed_model=embed_model)

# Prompts
context_extraction_prompt = PromptTemplate("""
You are a precise assistant built to extract **all distinct functional contexts** from a multi-intent user query.

Your responsibilities:
- Decompose the input into distinct functional tasks based on the actions the user is requesting.
- For each task, assign a **precise context** such as summarization, rewriting, transcription, translation, sentiment analysis, image analysis, idea extraction, feature suggestion, etc.
- For each function, return the **exact sentence or phrase** from the user’s input as the "Matched Text".

Extraction Rules:
1. Break compound sentences into separate context entries if they imply multiple functions.
2. Infer the most **specific context label** possible — avoid generic ones like “analysis” or “review”.
3. Include both **explicit** and **implicit** requests, even if a function is implied (e.g., "tell me what you notice in it" implies video analysis).
4. If the user has not provided any actionable content (i.e., the query is empty or non-functional), output:  
   **Please provide the content.**
5. Do not include setup or greeting lines unless they involve a request for action.
6. Do not return headings, numbers, or extra text — only clean `Context:` and `Matched Text:` pairs, or the fallback message.

Format your output exactly like this:

Context: <one or two words indicating the core task type>
Matched Text: <exact sentence or phrase from the user’s input>

Process the following User Query:
{query}
""")

recommendation_query_prompt = PromptTemplate("""
You are an expert model selector.

Given:
- User Context: {query}
- Retrieved Model Information: {context_str}

Your task:
1. Based on the user context and the model info, recommend if the model fits well.
2. Provide a brief explanation.

Respond exactly in the following format:

Reason: <clear reason why this model is good for the given user context>
""")

def choose_best_llm(context_text):
    lowered = context_text.lower()
    if any(word in lowered for word in ["chat", "conversation", "dialogue", "assistant","summarization","text generation","generate"]):
        return llms[0]  # openchat
    elif any(word in lowered for word in ["image", "vision", "recognition"]):
        return llms[1]  # deepseek
    elif any(word in lowered for word in ["sentiment", "emotion", "analysis"]):
        return llms[2]  # llama 3.1
    else:
        return llms[3]  # mistral

def get_model_recommendations(user_query):
    try:
        context_llm = llms[0]['llm']
        context_prompt_filled = context_extraction_prompt.format(query=user_query)
        extraction = context_llm.complete(context_prompt_filled).text.strip()

        lines = [line.strip() for line in extraction.splitlines() if line.strip()]
        seen_matches = set()
        results = []
        context, matched = None, None

        for line in lines:
            if line.startswith("Context:"):
                context = line.replace("Context:", "").strip()
            elif line.startswith("Matched Text:"):
                matched = line.replace("Matched Text:", "").strip()
                if context and matched and matched not in seen_matches:
                    results.append({"context": context, "matched_text": matched})
                    seen_matches.add(matched)

        if not results:
            return [{
                "context": "Unclear",
                "matched_text": "N/A",
                "top_model_details": [{
                    "model_info": "N/A",
                    "similarity_score": "N/A",
                    "reason": "Could not extract clear contexts."
                }]
            }]

        logger.info("Processing each context in parallel...")

        def process_context(item):
            try:
                ctx = item["context"]
                matched = item["matched_text"]
                llm_selected = choose_best_llm(ctx)
                llm_instance = llm_selected["llm"]
                model_name = llm_selected["name"]

                logger.info(f"[LMStudio Query] Context: '{ctx}' | Matched Text: '{matched}' | Selected Model: '{model_name}'")
                
                query_engine = index.as_query_engine(
                    llm=llm_instance,
                    text_qa_template=recommendation_query_prompt,
                    embed_model=embed_model,
                    similarity_top_k=1,
                    sparse_top_k=10,
                    enable_hybrid=True,
                )

                logger.info(f"[Recommendation Prompt Call] Calling model '{model_name}' for recommendation reasoning...")
                response = query_engine.query(matched)

                for node in response.source_nodes:
                    similarity_score = node.score
                    metadata = node.node.metadata or {}
                    models_name = metadata.get("Model Name", "N/A")
                    langchain_name = metadata.get("langchain_name", "N/A")

                parsed_reason = None
                for line in response.response.splitlines():
                    if line.startswith("Reason:"):
                        parsed_reason = line.replace("Reason:", "").strip()

                model_input_prompt = f"""You are a specialized model selected for this task.
                                        Input: {matched}
                                        Please provide your response or output based on the input above."""

                logger.info(f"[Final Output Call] Getting final model response from '{model_name}' for input: '{matched}'")
                
                model_response = llm_instance.complete(model_input_prompt).text.strip()

                return {
                    "context": ctx,
                    "matched_text": matched,
                    "selected_model": model_name,
                    "top_model_details": [{
                        "model_info": models_name,
                        "langchain_name": langchain_name or "N/A",
                        "similarity_score": similarity_score or "N/A",
                        "reason": parsed_reason or "N/A",
                        "model_output": model_response
                    }]
                }

            except Exception as e:
                logger.error(f"Failed to process context: {item}\nError: {e}")
                return {
                    "context": item.get("context", "Unknown"),
                    "matched_text": item.get("matched_text", "Unknown"),
                    "top_model_details": [{
                        "model_info": "Error",
                        "langchain_name": "N/A",
                        "similarity_score": "N/A",
                        "reason": str(e),
                        "model_output": "N/A"
                    }]
                }

        final = []
        for item in tqdm(results, desc="Processing contexts"):
            final.append(process_context(item))
        return final
        

    except Exception as e:
        logger.error(f"Error in recommendation engine: {e}")
        raise

# CLI
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your use case (or type 'exit'): ")
        if user_query.lower().strip() in ["exit", "quit"]:
            break

        try:
            results = get_model_recommendations(user_query)

            combined_output = ""
            for item in results:
                context = item.get("context", "Unknown")
                matched = item.get("matched_text", "Unknown")
                model_name = item.get("selected_model", "Unknown")
                model_details = item.get("top_model_details", [{}])[0]
                models_name = model_details.get("model_info", "N/A")
                output = model_details.get("model_output", "")
                similarity_score = model_details.get("similarity_score", "N/A")
                
                combined_output += (
                    f"\n--------------------------\nContext: {context}"
                    f"\nMatched Text: {matched}"
                    f"\nRecommended Model: {model_name}"
                    f"\nModel Name: {models_name}"
                    f"\nSimilarity Score: {similarity_score}"
                    f"\nOutput:\n{output}\n"
                )

            print("\n" + "=" * 60)
            print("Combined Response from All Contexts:")
            print("=" * 60)
            print(combined_output.strip())


        except Exception as e:
            logger.error(f"Failed to process: {e}")

