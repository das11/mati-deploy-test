from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import PromptTemplate
import os
import torch
import json
import pandas as pd
import uuid
import re
from datetime import datetime
import pprint

# ------------------------------------------
# Globals Declaration
# ------------------------------------------
embeddings_model = None
payload = None
llm = None
pinecone_index = None
summaries = {}

def set_embeddings_model(model):
    global embeddings_model
    embeddings_model = model

def set_payload(data):
    global payload
    payload = data
    
def set_llm(object):
    global llm
    llm = object
    
def set_pinecone_index(index):
    global pinecone_index
    pinecone_index = index


# ------------------------------------------
# Feedback Module GLOBALS
# ------------------------------------------

# Global batch_id
current_batch_id = None

# Global DataFrames for feedback and personalization (simulate DB tables)
feedback_df = pd.DataFrame(columns=["user_id", "batch_id", "component_id", "component_text", "score", "count"])
personalization_df = pd.DataFrame(columns=["user_id", "batch_id", "policy_code", "action"])

# Global variable to hold the previous summary components
summary_components = []


# ------------------------------------------
# Data Preprocessing
# ------------------------------------------

# Parse payload data json
def parse_payload_data_json(payload):
    parsed_data = {}

    for account in payload:
        account_id = account["accountId"]
        parsed_data[account_id] = {}

        for scenario in account["scenarios"]:
            account_opt_map_id = scenario["accountOptMapId"]
            policy_data = json.loads(scenario["policyData"])  # Convert JSON string to list of dicts
            
            feasibility_report = json.loads(scenario["feasibilityReport"])  # Convert JSON string to dict
            policy_check_data = [
                {
                    "policy_code": policy["policyCode"],
                    "policy_status": policy["policyStatus"],
                    "reason": policy["reason"],
                    "output_data": policy["outputData"]
                }
                for policy in policy_data
            ]
            
            if "feedback" in scenario:
                feedback = scenario["feedback"]
            
            parsed_data[account_id][account_opt_map_id] = {
                "policy_check_data": policy_check_data,
                "feasibility_report": feasibility_report,
                "feedback" : feedback if "feedback" in scenario else None
            }

    return parsed_data

# Get Policy Context for policy breaks and passes
def get_policy_context(policy_code):
    """
    Retrieves the policy context for a given policy code.

    Parameters:
    - policy_code (str): The policy code to retrieve the context for.

    Returns:
    - dict: The policy context metadata if found, otherwise an empty dictionary.
    """

    embedding = embeddings_model.get_text_embedding(policy_code)  # Generate embedding

    result = pinecone_index.query(vector=embedding, top_k=1, include_metadata=True)
    return result.matches[0].metadata if result.matches else {}


# ------------------------------------------
# Summary Modules
# ------------------------------------------

# --- System prompt for policy summary ---
system_prompt_policy = """
You are a portfolio compliance analyst. Summarize policy check results in this structure:

1. **Status**: 
   - "The portfolio is in full compliance." (if all green)
   - "The policy check report indicates the following issues:" (if red lights)

2. **Critical Issues**: 
   - For each "BREAK" policy: Explain why it happened. Use:
     - Trade IDs (e.g., BF.B-XNYS)
     - Thresholds (e.g., RGL > $0)
     - Financial metrics (e.g., tax cost ratio)

3. **Good Standing**: Collapse "SUCCESS" policies and add concise brief of policy good standing.

Rules:
- Use bold/emoji for highlights.
- Include numbers from `output_data` (e.g., "$1,788.94").
- For tax breaches, explain tax liability impact.
- Feedback/Personalization rules if present must be given more priority.
"""

# --- System prompt for feasibility summary ---
system_prompt_feasibility = """
You are a portfolio compliance analyst. Summarize the following feasibility report failed constraints.
Follow these rules : 
- Highlight the key issues, risk factors, and potential implications.
- Summary should be very brief.
- Present the summary in a concise, bullet-point format using bold text and emojis where appropriate.
- If there are no critical issues, mention that the portfolio constraints are fully feasible.
"""

# Policy Check data
def build_user_prompt_policy(policy_check_data):
    prompt = "Policy Check Results:\n"
    for policy in policy_check_data:
        context = get_policy_context(policy["policy_code"])
        # print("Context : ", context)
        prompt += f"""
        - Policy: {context["policy_code"]} ({context["severity"]})
          Status: {policy["policy_status"]}
          Reason: {policy["reason"]}
          Impact: {context["impact"]}
          Details: {policy["output_data"]}
        """
    return prompt
  
def generate_policy_summary(policy_check_data):
  """
  Generate a summary

  Input:
    - policy_check_data: List[dict] where each dict includes keys:
        'policy_code', 'policy_status', 'reason', 'output_data'

  Output:
    - A summary string generated by the LLM.
  """
  # Build the prompt with policy details
  user_prompt = build_user_prompt_policy(policy_check_data)
  # print("User Prompt - ", user_prompt)

  # Combine system instructions with the policy details
  final_prompt = f"{system_prompt_policy}\n\n{user_prompt}"

  # Use the LlamaIndex predictor to get the summary
  summary = llm.complete(final_prompt)

  return summary

# Feasibility Summary
def extract_failed_feasibility(feasibility_report, portfolio="current Portfolio"):
    """
    Extracts false feasibility constraints from the given feasibility report.

    Parameters:
      - feasibility_report (dict): The feasibility report loaded as a dictionary.
      - portfolio (str): Which portfolio section to check ("current Portfolio" or "optimized Portfolio").

    Returns:
      - List[str]: A list of strings, each describing a failed constraint.
    """
    failed_constraints = []

    # Get the portfolio section (e.g., "current Portfolio")
    portfolio_data = feasibility_report.get(portfolio, {})
    checks = portfolio_data.get("checks", {})

    # Iterate over each check
    for check_name, constraint_obj in checks.items():
        # Each check has one constraint key, e.g., "cash_bounds_constraint"
        for constraint_key, details in constraint_obj.items():
            # Ensure details is a dict and check if "feasible" is False
            if isinstance(details, dict) and not details.get("feasible", True):
                # Get the details (could be a list or a string)
                constraint_details = details.get("details", "No details provided")
                if isinstance(constraint_details, list):
                    constraint_details = "; ".join(constraint_details)
                failed_constraints.append(
                    f"{check_name} ({constraint_key}): {constraint_details}"
                )
    return failed_constraints

def build_feasibility_user_prompt(failed_constraints):
    """
    Build a user prompt from a list of failed constraint strings.

    Input:
      - failed_constraints (List[str]): Each string describes a failed constraint.

    Output:
      - A combined prompt string.
    """
    lines = ["**Feasibility Report - Failed Constraints:**"]
    for fc in failed_constraints:
        lines.append(f"- {fc}")
    return "\n".join(lines)

def generate_feasibility_summary(feasibility_report):
    """
    Generate a summary of feasibility report issues using GPT-4 via LlamaIndex.

    Input:
      - feasibility_report (str): The feasibility report to generate a summary for.

    Output:
      - A summary string generated by the LLM.
    """
    failed_constraints = extract_failed_feasibility(feasibility_report)
    
    # Build the user portion of the prompt
    user_prompt = build_feasibility_user_prompt(failed_constraints)

    # Combine with system instructions
    final_prompt = f"{system_prompt_feasibility}\n\n{user_prompt}"

    # Generate and return the summary using the LLM predictor
    summary = llm.complete(final_prompt)
    
    return summary

def append_feasibility_to_policy(policy_summary, feasibility_summary):
    updated_summary = str(policy_summary) + " \n\n **Feasbility Report Summary **:\n" + str(feasibility_summary)
    return updated_summary

def generate_summary(policy_check_data, feasibility_report):
    policy_summary = generate_policy_summary(policy_check_data)
    feasibility_summary = generate_feasibility_summary(feasibility_report)
    updated_summary = append_feasibility_to_policy(policy_summary, feasibility_summary)
  
    return updated_summary


# ------------------------------------------
# Feedback Modules
# ------------------------------------------

def breakdown_text(text: str) -> list:
    return [block.strip() for block in text.splitlines() if block.strip()]

def set_summary_text(summary: str):
    global summary_components
    summary_components = breakdown_text(summary)

def display_breakdown(summary: str):
    components = breakdown_text(summary)
    for idx, comp in enumerate(components):
        print(f"{idx}. {comp}")

def record_feedback(summary: str, user_id: str, component_id: int, vote: int):
    global feedback_df, summary_components, current_batch_id
    
    # Setting final summary text for component_text extraction
    set_summary_text(str(summary))
    
    # Automatically retrieve component_text based on component_id
    try:
        comp_text = summary_components[component_id]
    except IndexError:
        comp_text = ""
    
    # Timestamp for tracking the latest feedback session
    timestamp = datetime.utcnow()
    
    # Check if record exists for this user, component across ANY batch_id
    mask = ((feedback_df["user_id"] == user_id) & (feedback_df["component_id"] == component_id))
    existing_feedback = feedback_df[mask]
    
    if not existing_feedback.empty:
        # Update existing entry by averaging the scores and increasing count
        latest_entry = existing_feedback.sort_values("timestamp", ascending=False).iloc[0]
        new_count = latest_entry["count"] + 1
        new_score = (latest_entry["score"] * latest_entry["count"] + vote) / new_count
        
        # Update feedback_df (modify row directly)
        feedback_df.loc[mask, "score"] = new_score
        feedback_df.loc[mask, "count"] = new_count
        feedback_df.loc[mask, "timestamp"] = timestamp
        feedback_df.loc[mask, "summary_text"] = summary  # Store the full summary text
    else:
        # If no prior feedback, insert a new row
        new_row = pd.DataFrame({"user_id": [user_id],
                                "batch_id": [current_batch_id],  # Maintain batch tracking
                                "component_id": [component_id],
                                "component_text": [comp_text],
                                "score": [vote],
                                "count": [1],
                                "timestamp": [timestamp],
                                "summary_text": [summary]})  # Store summary text
        feedback_df = pd.concat([feedback_df, new_row], ignore_index=True)

def aggregate_feedback(account_id: str, feedback_df: pd.DataFrame) -> dict:
    """Aggregate feedback across all sessions for a user."""
    df = feedback_df[feedback_df["user_id"] == account_id]
    
    aggregated = {}
    for _, row in df.iterrows():
        comp_id = int(row["component_id"])
        if comp_id in aggregated:
            # Aggregate score & count
            aggregated[comp_id]["score"] = (aggregated[comp_id]["score"] * aggregated[comp_id]["count"] + row["vote"] * row["count"]) / (aggregated[comp_id]["count"] + row["count"])
            aggregated[comp_id]["count"] += row["count"]
        else:
            aggregated[comp_id] = {"score": row["vote"], "count": int(row["count"])}
    
    return aggregated

def extract_policy_code(component_text: str) -> str:
    match = re.search(r"`([^`]+)`", component_text)
    if match:
        return match.group(1)
    
    # Fallback: look for policy codes
    cleaned_text = re.sub(r"[\*\`]", "", component_text)  # Remove ** and ` characters
    match = re.search(r"\b[A-Z0-9_]{3,}\b", cleaned_text)  # Match words with uppercase letters, numbers, and underscores
    return match.group(0) if match else ""

def build_personalization_store(summary: str, user_id: str, batch_id: str, feedback_df: pd.DataFrame, threshold: int = 3) -> dict:
    components = breakdown_text(summary)
    agg = aggregate_feedback(user_id, feedback_df)
    store = {}
    print(f"Inside build_personalization_store : \nFeedback : {feedback_df.head()} \nAgg : {agg}")
    
    print("Starting extraction")
    for idx, comp in enumerate(components):
        print(f"- Component : [{idx}] : {comp}\n")
        code = extract_policy_code(comp)
        if code:
            fb = agg.get(idx, {"vote": 1, "count": 0})
            if fb["count"] >= threshold:
                if fb["score"] < 0.3:
                    store[code] = "omit"
                elif fb["score"] > 0.7:
                    store[code] = "elaborate"
    
    print(f"Store : {store}")
    
    personalization_df = pd.DataFrame(columns=["user_id", "batch_id", "policy_code", "action"])
    for code, action in store.items():
        mask = ((personalization_df["user_id"] == user_id) &
                (personalization_df["batch_id"] == batch_id) &
                (personalization_df["policy_code"] == code))
        if personalization_df[mask].empty:
            new_row = pd.DataFrame({"user_id": [user_id],
                                    "batch_id": [batch_id],
                                    "policy_code": [code],
                                    "action": [action]})
            personalization_df = pd.concat([personalization_df, new_row], ignore_index=True)
        else:
            personalization_df.loc[mask, "action"] = action
            
    print(f"Personalization DF : {personalization_df}")        
    
    return store

def build_personalized_user_prompt(policy_check_data: list, personal_store: dict) -> str:
    prompt = ""
    for policy in policy_check_data:
        code = policy["policy_code"]
        if code in personal_store and personal_store[code] == "omit":
            continue
        context = get_policy_context(code)
        block = (f"- Policy: {context['policy_code']} ({context['severity']})\n"
                 f"  Status: {policy['policy_status']}\n"
                 f"  Reason: {policy['reason']}\n"
                 f"  Impact: {context['impact']}\n"
                 f"  Details: {policy['output_data']}\n\n")
        if code in personal_store and personal_store[code] == "elaborate":
            block += "  [ELABORATE: Provide additional detail on this policy.]\n\n"
        prompt += block
    return prompt

def build_final_prompt(system_prompt_policy: str, policy_check_data: list, summary: str, user_id: str, batch_id: str, feedback_df: pd.DataFrame) -> str:
    personal_store = build_personalization_store(summary, user_id, batch_id, feedback_df)
    user_prompt = build_personalized_user_prompt(policy_check_data, personal_store)
    feedback_section = ("\nPersonalization Rules:\n" +
                        "\n".join([f"{k}: {v}" for k, v in personal_store.items()])
                        if personal_store else "No personalization adjustments.")
    final_prompt = f"""
System Instructions:
{system_prompt_policy}
{feedback_section}

Personalized User Data:
{user_prompt}

Please generate a final summary that strictly adheres to the System Instructions and incorporates the above personalization adjustments.
    """
    return final_prompt


# ------------------------------------------
# Final Summary Modules
# ------------------------------------------

def generate_final_summary(system_prompt_policy: str, policy_check_data: list, feasibility_report_json_data, user_id: str, batch_id: str, feedback:str = None) -> str:
    if feedback is not None:
        feedback_df = pd.read_json(json.dumps(feedback), orient="records")
        
        # Build the modified prompt using personalization rules
        policy_check_summary = generate_policy_summary(policy_check_data)
        
        print(f"Summary : {policy_check_summary}\nBreakdown summary : {breakdown_text(str(policy_check_summary))}")
        
        modified_policy_summary_prompt = build_final_prompt(system_prompt_policy, policy_check_data, str(policy_check_summary), user_id, batch_id, feedback_df)
        policy_summary = llm.complete(modified_policy_summary_prompt)
        feasibility_summary = generate_feasibility_summary(feasibility_report_json_data)
        
        print(f"--> Prompt : {modified_policy_summary_prompt}\n\n")
        
        final_summary = append_feasibility_to_policy(policy_summary, feasibility_summary)
        
    else:
        # Generate a straightforward policy summary
        final_summary = generate_summary(policy_check_data, feasibility_report_json_data)
    
    return final_summary
  
  

# ------------------------------------------
# Main Loop
# ------------------------------------------

def main_loop():
    global current_batch_id, summaries
    current_batch_id = str(uuid.uuid4())
    parsed_data = parse_payload_data_json(payload)

    for account_id, account_data in parsed_data.items():
        summaries[account_id] = {}
        
        for account_opt_map_id, details in account_data.items():
            policy_check_data = [
                {
                    "policy_code": policy["policy_code"],
                    "policy_status": policy["policy_status"],
                    "reason": policy["reason"],
                    "output_data": policy["output_data"]
                }
                for policy in details["policy_check_data"]
            ]
            feasibility_report = details["feasibility_report"]
            feedback = details["feedback"] if "feedback" in details else None

            account_summary = generate_final_summary(system_prompt_policy, policy_check_data, feasibility_report, account_id, current_batch_id, feedback)

            summaries[account_id][account_opt_map_id] = {
                "summary": account_summary,
            }
    
    return summaries