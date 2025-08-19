import os
import weaviate
from dotenv import load_dotenv
from uuid import uuid4
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document


# Load env vars

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
weaviate_cluster_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")


# Connect to Weaviate

auth_config = AuthApiKey(api_key=weaviate_api_key)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_cluster_url,
    auth_credentials=auth_config,
    headers={"X-OpenAI-Api-Key": openai_api_key}
)
print("✅ Connected to Weaviate Cloud")


# Embeddings + LLM

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")


# Run strategy

def run_strategy(name, splitter_or_func, query, pages):
    if callable(splitter_or_func):
        documents = splitter_or_func(pages)
    else:
        documents = splitter_or_func.split_documents(pages)

    collection_name = f"Doc_Combined_{name}"
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    properties = [Property(name="text", data_type=DataType.TEXT)]
    client.collections.create(
        name=collection_name,
        properties=properties,
        vector_index_config=Configure.VectorIndex.hnsw(),
        vectorizer_config=Configure.Vectorizer.none()
    )

    documents_with_ids = [
        Document(page_content=doc.page_content,
                 metadata={**doc.metadata, "uuid": doc.metadata.get("uuid", str(uuid4()))})
        for doc in documents
    ]

    vectorstore = WeaviateVectorStore.from_documents(
        documents=documents_with_ids,
        embedding=embeddings,
        client=client,
        index_name=collection_name,
        text_key="text",
        by_text=False
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    return result, len(documents)
