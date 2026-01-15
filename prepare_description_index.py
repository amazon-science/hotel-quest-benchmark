from langchain.docstore.document import Document
from datasets import load_dataset
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd, tqdm

URI = "./descriptions_hotel.db"                 # ← Milvus‑Lite file or server URL
df = load_dataset("guyhadad01/Hotels_Descriptions")["train"].to_pandas()
required = ['countyName', 'cityName', 'HotelName', 'HotelRating', 'Description']
df = df.dropna(subset=required)
df = df[df['Description'].apply(lambda x: len(str(x).split()) >= 10)]
df.columns = df.columns.str.strip()
docs = []
for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="docs"):
    docs.append(
        Document(
            page_content=f"Description: {row['Description']}",
            metadata={
                "Rating": str(row["HotelRating"]),
                "Name":   str(row["HotelName"]),
                "Country": str(row["countyName"]),
                "City":   str(row["cityName"]),
            },
        )
    )

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Milvus.from_documents(
    docs,
    emb,
    collection_name="hotel_index",
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    drop_old=True,
)



## Example usage:

# URI = "./descriptions_hotel.db"                 # ← Milvus‑Lite file or server URL
# emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector_store_loaded = Milvus(
#     emb,
#     connection_args={"uri": URI},
#     collection_name="hotel_index",
# )

# filtered = vector_store_loaded.similarity_search(
#     "pool",
#     k=30,
#     expr="City like '%Tel Aviv%'",
# )