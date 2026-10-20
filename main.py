from dotenv import load_dotenv
load_dotenv()


from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from node import run_agent_reasoning, tool_node


AGENT_REASON = 'agent_reason'
ACT = 'act'
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state['messages'][LAST].tool_calls:
        return END

    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue,{END: END,ACT:ACT})

flow.add_edge(ACT,AGENT_REASON)

app = flow.compile()

try:
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")
    print("Graph picture saved as Flow.png")

except Exception as e:
    print("Could not save flow.png (needs internet). Here is the graph as text instead: ")
    print(app.get_graph().draw_mermaid())



if __name__ == "__main__":
    print('hello ReAct LangGraph with Function Calling')

    question = (
        "My order of Rs 2499 is 6 days late with Delhivery. "
        "What is Delhivery's policy on delayed deliveries? "
        "Also, how much is 10% compensation on my order?"
    )

    res = app.invoke({"messages": [HumanMessage(content=question)]})

    print("\n ----- Every message in the state -----")

    for msg in res["messages"]:
        print(f"\n [{msg.type.upper()}]")

        if getattr(msg, "tool_calls", None):

            for call in msg.tool_calls:
                print(f"tool_call -> {call['name']} ({call['args']})")


        else:
            print(str(msg.content)[:300])



    print("\n ---- Final Answer -----")
    print(res["messages"][LAST].content)














