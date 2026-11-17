from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek


load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url=qdrant_url,
    api_key=qdrant_api_key,
    collection_name="rag-practice"
)

user_query = input("ask anything...")

results = vector_store.similarity_search(user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content} \n Page Number : {result.metadata["page_label"]} \n File Location : {result.metadata["source"]}"
                         for result in results ])

SYSTEM_PROMPT = f""""
You are a helpfull assesstent who only replies from the context data 
you have given.
And retrives the data from given pdf chunk.
You should only give ans to based on user query.
Always navigate the user to right page number to know more

Context: {context}
"""


response = ChatDeepSeek.chat.completions.create(
    model="deepseek-3",
    messages=[
        {"role":"system", "content":SYSTEM_PROMPT},
        {"role":"user", "content":user_query}
    ]
    )


print("response : ", response.choices[0].message.content )