
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex


def company_list_builder():
# NOTE 
# This creates a dummy dictionary of comapany_names without 
# the Document object and used as a enumerated list for queryEngines 

    with open("./data_stage/lead_gen/company_names.txt", "r") as file:
        # Read all lines into a list
        lines = file.readlines()

    # Create an empty dictionary
    company_names = {}

    # Loop through each line (name)
    for line in lines:
        # Remove trailing newline character (if present)
        name = line.strip()
        # Add the name as the key and a dummy value (e.g., "dummy") as the value
        company_names[name] = "dummy"


    for company in company_names:
        # print(company)
        pass

    return company_names

def lead_gen_qe_tools(pinecone):
    company_names = company_list_builder()

    lead_gen_query_engine_tools = []
    for idx, company_doc in enumerate(company_names):
        pinecone_index = pinecone.Index("mati-index")

        company_doc = company_doc.replace(" ", "_")
        namespace = f"lead_gen_DA_{company_doc}"

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # vector_index = VectorStoreIndex(nodes = nodes, storage_context = storage_context, show_progress = True)
        # print(f"Indexed data TOKENS : {token_counter.total_embedding_token_count}")

        vector_index = VectorStoreIndex.from_vector_store(vector_store = vector_store)

        # build summary index
        # summary_index = SummaryIndex(nodes)

        # define query engines
        vector_query_engine = vector_index.as_query_engine(llm=llm)
        # summary_query_engine = summary_index.as_query_engine(llm=llm)

        # define tools
        query_engine_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    f"Useful for questions related to {company_doc}. Especially when lead gen and {company_doc} is mentioned"
                ),
            ),
        )

        lead_gen_query_engine_tools.append(query_engine_tool)

    # print((lead_gen_query_engine_tools[97].metadata))
    return lead_gen_query_engine_tools