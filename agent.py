import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()
model_name = os.getenv("MODEL")

# --- Tool: Save User Prompt ---

def add_prompt_to_state(tool_context: ToolContext, prompt: str) -> dict[str, str]:
    """Stores user prompt in state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] PROMPT: {prompt}")
    return {"status": "success"}

# --- Wikipedia Tool (Optional External Knowledge) ---

wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# --- 1. Question Understanding + Research Agent ---

qa_agent = Agent(
    name="qa_agent",
    model=model_name,
    description="Answers user questions using reasoning and external knowledge if needed.",
    instruction="""
    You are an intelligent question-answering assistant.

    Your job:
    - Understand the user's PROMPT clearly.
    - Decide if external knowledge (Wikipedia) is needed.
    - If needed, use the Wikipedia tool.
    - Otherwise, answer using your own knowledge.
    - Provide a clear, accurate, and helpful answer.

    Keep answers:
    - Concise but informative
    - Structured when helpful
    - Easy to understand

    PROMPT:
    { PROMPT }
    """,
    tools=[wikipedia_tool],
    output_key="answer"
)

# --- 2. Response Formatter (Optional but cleaner UX) ---

formatter_agent = Agent(
    name="formatter_agent",
    model=model_name,
    description="Formats the final answer cleanly.",
    instruction="""
    You are a helpful assistant that formats answers for clarity.

    Take the ANSWER and:
    - Improve readability
    - Add bullet points if needed
    - Keep it clean and user-friendly

    ANSWER:
    { answer }
    """
)

# --- Workflow ---

qa_workflow = SequentialAgent(
    name="qa_workflow",
    description="Handles general question answering",
    sub_agents=[
        qa_agent,
        formatter_agent
    ]
)

# --- Root Agent ---

root_agent = Agent(
    name="qa_entry",
    model=model_name,
    description="Entry point for question answering system.",
    instruction="""
    - Ask the user what question they have.
    - When the user responds, store it using 'add_prompt_to_state'.
    - Then pass control to 'qa_workflow'.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[qa_workflow]
)