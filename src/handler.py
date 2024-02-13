import runpod

from pinecone import Pinecone, PodSpec, ServerlessSpec
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI

# VectorStore dependencies 
from llama_index import VectorStoreIndex
from llama_index import StorageContext, load_index_from_storage
from llama_index.vector_stores import PineconeVectorStore

import os
from dotenv import load_dotenv

# Service Context dependencies
from llama_index import set_global_service_context
from llama_index import ServiceContext

# Embeddings wrapper
from llama_index.embeddings import LangchainEmbedding

# HF embeddings - To represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# OpenAI API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# LLM
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")

# Fetching custom embedding model
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder="../embeddingModelCache")
    # HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
)

# Creating new Service Context and setting it to GLOBAL
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

# Setting the service context
set_global_service_context(service_context)



# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def pinecone_init():
    api_key = "e80bd265-1ff4-4ec3-8f03-b7929e7b1011"
    pinecone = Pinecone(api_key=api_key)

    return pinecone

def build_index(pinecone):
    # Connect Pinecone vectorstore with existing embeddings

    pinecone_index = pinecone.Index("mati-index")

    vector_store = PineconeVectorStore(pinecone_index = pinecone_index, namespace="collection_engine_1")
    index = VectorStoreIndex.from_vector_store(vector_store = vector_store)

    return index

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input.get('prompt')

    pinecone = pinecone_init()
    index = build_index(pinecone)

    query_engine = index.as_query_engine()
    st_query_engine = index.as_query_engine(streaming = True)

    query = f"{prompt} Tell me in detail"

    response = query_engine.query(query)
    print(response)

    return f"Hello!! \n\n Response - {response}"


runpod.serverless.start({"handler": handler})
