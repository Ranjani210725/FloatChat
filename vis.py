import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Step 1: Connect to PostgreSQL and load profile data
db_url = "postgresql+psycopg2://postgres:ranj2005@localhost:5432/argo_db"
engine = create_engine(db_url)

sql_query = """
SELECT 
    cycle_number,
    latitude,
    longitude,
    julian_day,
    data_mode,
    pressure,
    temperature,
    salinity
FROM argo_profiles
LIMIT 100;
"""

df = pd.read_sql(sql_query, engine)
print(f"\nColumns: {df.columns.tolist()}")
print(f"Rows returned: {len(df)}")

# Step 2: Create semantic summaries
def summarize(row):
    return (
        f"Cycle {row['cycle_number']} recorded salinity {row['salinity']} and temperature {row['temperature']} "
        f"at pressure {row['pressure']} near ({row['latitude']}, {row['longitude']}) on Julian day {row['julian_day']}."
    )

df['summary'] = df.apply(summarize, axis=1)

# Step 3: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['summary'].tolist())
embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

# Step 4: Store in Chroma
client = chromadb.PersistentClient(path="./chroma_store")

# Safe delete if collection exists
try:
    client.delete_collection("argo_profiles")
except chromadb.errors.NotFoundError:
    print("Collection 'argo_profiles' does not exist yet—no need to delete.")

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="argo_profiles", embedding_function=embedding_function)

# Convert any datetime columns to strings (if present)
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = df[col].astype(str)

documents = df['summary'].tolist()
metadatas = df.drop(columns=['summary']).to_dict(orient='records')
ids = [f"cycle_{row['cycle_number']}_lat_{row['latitude']}_lon_{row['longitude']}_idx_{i}" for i, row in df.iterrows()]

print(f"\nDocuments: {len(documents)}")
print(f"Embeddings: {len(embeddings_list)}")
print(f"Metadatas: {len(metadatas)}")
print(f"IDs: {len(ids)}")

if documents and embeddings_list:
    collection.add(
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )

    # Step 5: Semantic search example
    results = collection.query(
        query_texts=["Floats recorded during monsoon season"],
        n_results=5
    )

    print("\nTop semantic matches:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"{i+1}. {doc}")
        print(f"   → Location: ({meta['latitude']}, {meta['longitude']}) | Salinity: {meta['salinity']} | Temp: {meta['temperature']}")

    # Step 6: Visualization – Float Locations
    lats = [meta['latitude'] for meta in results['metadatas'][0]]
    lons = [meta['longitude'] for meta in results['metadatas'][0]]

    plt.figure(figsize=(8, 6))
    plt.scatter(lons, lats, c='blue', alpha=0.6, edgecolors='k')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Top Semantic Matches: Float Locations")
    plt.grid(True)
    plt.show()

    # Step 7: Visualization – Salinity vs. Temperature
    salinity = [meta['salinity'] for meta in results['metadatas'][0]]
    temperature = [meta['temperature'] for meta in results['metadatas'][0]]

    plt.figure(figsize=(8, 6))
    plt.scatter(salinity, temperature, c='green', alpha=0.6, edgecolors='k')
    plt.xlabel("Salinity")
    plt.ylabel("Temperature")
    plt.title("Semantic Matches: Salinity vs. Temperature")
    plt.grid(True)
    plt.show()

    # Step 8: Bar Chart – Average Salinity per Cycle
    salinity_df = pd.DataFrame(results['metadatas'][0])
    avg_salinity = salinity_df.groupby('cycle_number')['salinity'].mean()

    plt.figure(figsize=(8, 6))
    avg_salinity.plot(kind='bar', color='purple', edgecolor='black')
    plt.xlabel("Cycle Number")
    plt.ylabel("Average Salinity")
    plt.title("Average Salinity per Cycle (Top Matches)")
    plt.grid(True)
    plt.show()

    # Step 9: Pie Chart – Data Mode Distribution
    mode_counts = salinity_df['data_mode'].value_counts()

    plt.figure(figsize=(6, 6))
    mode_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'salmon'])
    plt.title("Data Mode Distribution (Top Matches)")
    plt.ylabel("")  # Hide y-label
    plt.show()

else:
    print("No documents or embeddings to add. Check your query or summary logic.")
