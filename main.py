import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflection_chain

from typing import Sequence, List

load_dotenv(override=True)


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


# the key difference between this and the other is involves the HumanMessage
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("hello there")
    inputs = HumanMessage(
        content="""Make this tweet better:"
        @LangChainAI
        - newly Tool Calling feature is seriously underrated.
        After a long wait, it's here-making the implementation of agents across different models with function calling -super easy.
        Made a video covering their newest blog post.
        """
    )

    inputs1 = HumanMessage(
        content="""Make this tweet better:"
        @jordanhasnolife5163
        - Jordan Deserves more subscribers @NeetCode levels.
        - For simply the best system design videos out there, you must turn to Jordan Has No Life.
        -If you want to have a real understanding of the fundamentals of systems design, an understanding that would impress even Martin Kleppmann, look no further.
        www.youtube.com/@jordanhasnolife5163
        """
    )

    # response = graph.invoke(inputs)
    response = graph.invoke(inputs1)
