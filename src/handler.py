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
from llama_index.core import set_global_handler

# Embeddings wrapper
from llama_index.embeddings.langchain import LangchainEmbedding

# HF embeddings - To represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Node preprocessors
from llama_index.postprocessor.cohere_rerank import CohereRerank

# ARIS Prompting model
import aris_prompting 

# ARIS Source Context
import source_context

#################################################################################################################################################
# OpenAI API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
os.environ['PROMPTLAYER_API_KEY'] = os.getenv("PROMPTLAYER_API_KEY")

ge_p = os.getenv("PINECONE_API_KEY")
ge_o = os.getenv("OPENAI_API_KEY")
ge_c = os.getenv("COHERE_API_KEY")
ge_pl = os.getenv("PROMPTLAYER_API_KEY")

print(f"GETENV : {ge_p}")
print(f"GETENV : {ge_o}")
print(f"GETENV : {ge_c}")
print(f"GETENV : {ge_pl}")

# Printing secrets to debug
RP_SECRET_NAMESPACE = os.environ.get("NAMESPACE")
RP_SECRET_NAMESPACE_DOC_RESEARCH = os.environ.get("NAMESPACE_DOC_RESEARCH")
print(f"Secret : {RP_SECRET_NAMESPACE} ")
print(f"Secret : {RP_SECRET_NAMESPACE_DOC_RESEARCH} ")
print(f"OS env : {os.environ}")

# LLM
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")

# Fetching custom embedding model
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder="../embeddingModelCache")
)

# Promptlayer handler
set_global_handler("promptlayer", pl_tags=["aris-mati"])

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

    doc_research_namespace = RP_SECRET_NAMESPACE_DOC_RESEARCH if RP_SECRET_NAMESPACE_DOC_RESEARCH != None else "doc-research"
    doc_research_vector_store = PineconeVectorStore(pinecone_index = pinecone_index, namespace=doc_research_namespace)
    doc_reseaerch_index = VectorStoreIndex.from_vector_store(vector_store = doc_research_vector_store)

    lead_gen_vector_store = PineconeVectorStore(pinecone_index = pinecone_index, namespace="lead_gen_2")
    lead_gen_index = VectorStoreIndex.from_vector_store(vector_store = lead_gen_vector_store)


    print(f"Namespces used : {namespace}, {doc_research_namespace}")

    return index, doc_reseaerch_index, lead_gen_index

def fetch_dataframes():
    import pandas as pd

    holdings_parquet_url = "https://project-mati-nd-cloudsync.s3.us-east-2.amazonaws.com/holdings.parquet.gzip"
    holdings_df = pd.read_parquet(holdings_parquet_url)

    return holdings_df 

def build_query_engines(index, doc_research_index, lead_gen_index):
    from llama_index.core.query_engine import PandasQueryEngine

    # ARIS Base
    aris_query_engine = index.as_query_engine(
        similarity_top_k = 5,
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
    holdings = fetch_dataframes()
    aris_holding_query_engine = PandasQueryEngine(df=holdings, verbose=True, llm=llm, synthesize_response=True, instruction_str=aris_prompting.holdings_qe_instruction_str, pandas_prompt=aris_prompting.holdings_qe_pandas_prompt)

    # Doc Research
    cohere_api_key = os.environ["COHERE_API_KEY"]
    cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)

    doc_research_query_engine = doc_research_index.as_query_engine(
        # response_mode = "compact",/
        similarity_top_k = 10,
        text_qa_template = aris_prompting.doc_research_template,
        node_postprocessors = [cohere_rerank],
        streaming = True
    )

    # Lead Generation 
    cohere_rerank_lead_gen = CohereRerank(api_key=cohere_api_key, top_n=2)
    lead_gen_query_engine = lead_gen_index.as_query_engine(
        # response_mode = "compact",/
        similarity_top_k = 10,
        text_qa_template = aris_prompting.lead_gen_template,
        node_postprocessors = [cohere_rerank_lead_gen],
        streaming = True
    )


    return aris_query_engine, aris_summary_query_engine, aris_holding_query_engine, doc_research_query_engine, lead_gen_query_engine

def router_engine(index, doc_research_index, lead_gen_index):
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
    from llama_index.core.selectors import (
        PydanticMultiSelector,
        PydanticSingleSelector,
    )
    from llama_index.core.tools import QueryEngineTool
    import nest_asyncio

    aris_query_engine ,aris_summary_query_engine, aris_holding_query_engine, doc_research_query_engine, lead_gen_query_engine = build_query_engines(index, doc_research_index, lead_gen_index)

    holding_qe_tool = QueryEngineTool.from_defaults(
        query_engine=aris_holding_query_engine,
        description="Useful for shares or holding related questions for Accounts"
    )
    summary_qe_tool = QueryEngineTool.from_defaults(
        query_engine=aris_summary_query_engine,
        description="Useful for summarization questions related to the accounts. Especially when asked to summarize an account. ",
    )
    general_qe_tool = QueryEngineTool.from_defaults(
        query_engine=aris_query_engine,
        description="Useful for generic questions regarding account holders or accounts",
    )
    doc_research_qe_tool = QueryEngineTool.from_defaults(
        query_engine=doc_research_query_engine,
        description="Useful for answering questions related to the Callan Institute pdfs. Also useful when asked about \"research\""
    )

    lead_gen_qe_tool = QueryEngineTool.from_defaults(
        query_engine=lead_gen_query_engine,
        description="Useful for answering questions related to lead gen. Especially useful lead gen is mentioned."
    )

    router_query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            holding_qe_tool,
            summary_qe_tool,
            general_qe_tool,
            doc_research_qe_tool,
            lead_gen_qe_tool
        ]
    )

    return router_query_engine

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input.get('prompt')

    pinecone = pinecone_init()
    index, doc_research_index, lead_gen_index = build_index(pinecone)

    router_query_engine = router_engine(index, doc_research_index, lead_gen_index)

    query = f"{prompt}"
    response = router_query_engine.query(query)
    sources = source_context.source_context(response)

    inference = f"{response}\n{sources}"


    return f"\n{inference}"


runpod.serverless.start({"handler": handler})
