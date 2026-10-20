from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import    TavilySearch

load_dotenv()

@tool()
def calculate_compensation(order_amount: float, percent: float) -> float:
    """
    Calculate the compensation amount for a late or faulty order.

    :param order_amount: the total order value in rupees, for example 2499
    :param percent: the compensation percentage, for example 10 for 10%
    :return: the compensation amount in rupees, rounded to 2 decimals
    """

    return round(float(order_amount) * float(percent) / 100, 2)


tools = [TavilySearch(max_results=3), calculate_compensation]

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    temperature=0,
    timeout=120

).bind_tools(tools)


if __name__ == "__main__":
    print("Tool loaded:", [t.name for t in tools])
    print("Quick tool test (2499, 10%): ",calculate_compensation.invoke({"order_amount": 2499, "percent":10}))











