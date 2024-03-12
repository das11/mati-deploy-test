######################################################################## Prompting Models - ARIS ########################################################################

# ARIS BASE TEMPLATE
#
# Defining basic persona

from llama_index.core.prompts import PromptTemplate
from llama_index.core import PromptTemplate
import handler

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

# Doc Research
doc_research_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query in detail. You are an excellent financial analyst named Mati.\n"
    "Query: {query_str}\n"
    "Follow these instructions : \n"
    "Strucure the response as a table if possible \n"
    "If its not tabular then structure it multiple points \n"
    "Answer: "
)
doc_research_template = PromptTemplate(doc_research_template_str)

# Holdings Query Engine 
holdings_qe_instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. Consider partial matches if relevant\n"
    "6. PRINT ONLY THE EXPRESSION.\n"
    "7. Do not quote the expression.\n" 
)
pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n"
    "The column names of the dataframe are : \n"
    "{df_columns}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

holdings_df = handler.fetch_dataframes()
holdings_qe_pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str = holdings_qe_instruction_str, 
    df_str=holdings_df.head(5),
    df_columns = list(holdings_df.columns)
)