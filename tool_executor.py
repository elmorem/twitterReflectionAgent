from dotenv import load_dotenv
from typing import List

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import Tool


from schemas import AnswerQuestion, ReviseAnswer, Reflection
from chains_reflex import parser

load_dotenv(override=True)

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolExecutor([tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    tool_invocation = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocation.append(ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query,
            )
            )
            ids.append(parsed_call["id"])
    outputs = tool_executor.batch(tool_invocation)

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
