# llm_backend.py

from sqlalchemy import create_engine
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

engine = create_engine("postgresql+psycopg2://postgres:ranj2005@localhost:5432/argo_db")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="argo_profiles")
model = SentenceTransformer('all-MiniLM-L6-v2')

def route_query(user_input):
    if ">" in user_input or "between" in user_input or "SELECT" in user_input.upper():
        return "sql"
    return "semantic"

def query_sql(user_input):
    query = "SELECT * FROM argo_profiles WHERE salinity > 35 LIMIT 5"
    df = pd.read_sql(query, engine)
    return df

def query_chroma(user_input):
    results = collection.query(query_texts=[user_input], n_results=5)
    return results['documents'][0], results['metadatas'][0]

def generate_response(user_input):
    route = route_query(user_input)
    if route == "sql":
        df = query_sql(user_input)
        return df.to_string(index=False)
    else:
        docs, metas = query_chroma(user_input)
        response = ""
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            response += f"{i+1}. {doc}\n"
            response += f"   → Location: ({meta['latitude']}, {meta['longitude']}) | Salinity: {meta['salinity']} | Temp: {meta['temperature']}\n"
        return response
