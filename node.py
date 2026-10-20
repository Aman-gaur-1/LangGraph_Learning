from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful customer support assistant for an Indian e-commerce company.
Use the tools when you need live information or exact calculations.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content":SYSTEM_MESSAGE}, *state["messages"]])

    return {"messages": [response]}

tool_node = ToolNode(tools)
