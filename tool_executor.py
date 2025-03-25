from dotenv import load_dotenv
from typing import List

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode


from schemas import AnswerQuestion, ReviseAnswer, Reflection
from chains_reflex import parser

load_dotenv(override=True)

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)



if __name__ == "__main__":
    print("hello there")

    human_message = HumanMessage(
        content="Write about the future of AI agents and list some of the most important new startups in the field."

    )

    answer = AnswerQuestion(
    answer = "",
    reflection = Reflection(missing="", superfluous=""),
    search_queries=[
                "AI Agents Futures",
                "AI Agents Startups",
                "AI Agents Trends",
                "AI Agents Innovations",
                "AI Agents Predictions",
                ]

        )
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
             content = "",
             tool_calls=[
                 {
                     "name": AnswerQuestion.__name__,
                     "args": answer.dict(),
                     "id":  "somecallid",
                 }
             ],
         ),
          ]
          )
    print(raw_res)
