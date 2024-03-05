######################################################################## Prompting Models - ARIS ########################################################################

# ARIS BASE TEMPLATE
#
# Defining basic persona

from llama_index.core.prompts import PromptTemplate

# Base ARIS Template
aris_qa_template_base_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query in detail. You are an excellent financial analyst named Mati.\n"
    "Query: {query_str}\n"
    "Answer: "
)
aris_qa_template_base = PromptTemplate(aris_qa_template_base_str)


# Summary ARIS Template
aris_summary_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are an excellent financial analyst named Mati.\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Follow these instructions to form the bullets : \n"
    "1. Bullet #1 should include - Mention the financial advisor, strategy and tax sensitivity\n"
    "2. Bullet #2 should include - Mention the Total market value, Cash level\n"
    "3. Bullet #3 should include - Tracking Error, only if available"
    "Query: {query_str}\n"
    "Answer :"
)
aris_summary_template = PromptTemplate(aris_summary_template_str)

# Reranked ARIS Template
reranked_aris_summary_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are an excellent financial analyst named Mati.\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Follow these instructions to form the bullets : \n"
    "1. Bullet #1 should include - Mention the financial advisor, strategy and tax sensitivity\n"
    "2. Bullet #2 should include - Mention the Total market value, Cash level and Tracking Error\n"
    "Query: {query_str}\n"
    "Answer :"
)
reranked_aris_summary_template = PromptTemplate(reranked_aris_summary_template_str)

