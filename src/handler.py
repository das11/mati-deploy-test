import runpod

from pinecone import Pinecone, PodSpec, ServerlessSpec
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate


# VectorStore dependencies 
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv

# Service Context dependencies
from llama_index.core import set_global_service_context
from llama_index.core import ServiceContext

# Embeddings wrapper
from llama_index.embeddings.langchain import LangchainEmbedding

# HF embeddings - To represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# ARIS Prompting model
import aris_prompting 

#################################################################################################################################################
# OpenAI API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

print(f"GETENV : {os.getenv("PINECONE_API_KEY")}")
print(f"GETENV : {os.getenv("OPENAI_API_KEY")}")

# Printing secrets to debug
RP_SECRET_NAMESPACE = os.environ.get("NAMESPACE")
print(f"Secret : {RP_SECRET_NAMESPACE} ")
print(f"OS env : {os.environ}")

# LLM
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")

# Fetching custom embedding model
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder="../embeddingModelCache")
)

# Creating new Service Context and setting it to GLOBAL
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

# Setting the service context
set_global_service_context(service_context)

#################################################################################################################################################


# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def pinecone_init():
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone = Pinecone(api_key=pinecone_api_key)

    return pinecone

def build_index(pinecone):
    # Connect Pinecone vectorstore with existing embeddings

    pinecone_index = pinecone.Index("mati-index")
    namespace = RP_SECRET_NAMESPACE if RP_SECRET_NAMESPACE != None else "collection_engine_1"

    vector_store = PineconeVectorStore(pinecone_index = pinecone_index, namespace=namespace)
    index = VectorStoreIndex.from_vector_store(vector_store = vector_store)

    return index

def fetch_dataframes():
    import pandas as pd

    holdings_parquet_url = "https://project-mati-nd-cloudsync.s3.us-east-2.amazonaws.com/holdings.parquet.gzip"
    holdings_df = pd.read_parquet(holdings_parquet_url)

    return holdings_df 

def build_query_engines(index):
    from llama_index.core.query_engine import PandasQueryEngine

    # ARIS Base
    aris_query_engine = index.as_query_engine(
        similarity_top_k = 2,
        text_qa_template= aris_prompting.aris_qa_template_base
    )

    # ARIS Summary
    aris_summary_query_engine = index.as_query_engine(
        # response_mode = "compact",/
        similarity_top_k = 2,
        text_qa_template=aris_prompting.aris_summary_template,
        streaming = True
    )

    # ARIS Holdings 
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. Consider partial matches if relevant\n"
        "5. PRINT ONLY THE EXPRESSION.\n"
        "6. Do not quote the expression.\n" 
    )
    holdings = fetch_dataframes()
    aris_holding_query_engine = PandasQueryEngine(df=holdings, verbose=True, instruction_str=instruction_str, llm=llm, synthesize_response=True)

    return aris_query_engine, aris_summary_query_engine, aris_holding_query_engine

def router_engine(index):
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
    from llama_index.core.selectors import (
        PydanticMultiSelector,
        PydanticSingleSelector,
    )
    from llama_index.core.tools import QueryEngineTool
    import nest_asyncio

    aris_query_engine ,aris_summary_query_engine, aris_holding_query_engine = build_query_engines(index)

    holding_qe_tool = QueryEngineTool.from_defaults(
        query_engine=aris_holding_query_engine,
        description="Useful for shares or holding related questions for Accounts"
    )
    summary_qe_tool = QueryEngineTool.from_defaults(
        query_engine=aris_summary_query_engine,
        description="Useful for summarization questions related to the accounts.",
    )
    general_qe_tool = QueryEngineTool.from_defaults(
        query_engine=aris_query_engine,
        description="Useful for generic questions not involving summarization",
    )

    router_query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            holding_qe_tool,
            summary_qe_tool,
            general_qe_tool,
        ]
    )

    return router_query_engine

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input.get('prompt')

    pinecone = pinecone_init()
    index = build_index(pinecone)

    router_query_engine = router_engine(index)

    query = f"{prompt}"
    response = router_query_engine.query(query)
    print(response)

    return f"Hey!\n{response}"


runpod.serverless.start({"handler": handler})
