import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4-turbo")

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
        Current time: {time}
        1. {first_instruction}
        2. Reflect and critique your answer.  Be severe to maximize improvement
        3. Recomment search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format"),
    ]
).partial(time=datetime.datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed answer approximately 250 words long."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools={AnswerQuestion}, tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer with the new information.
    - You should use the previous critique to add important information to your answer.
        -you MUST incldue numerical citations in your revised answer to ensure it can be verified.
        - add a "References" section to the bottom of your answer (which does not count toward the word limit.)
            - [1] https://example.com
            - [2] https://example.com
        You should use the previous critique yo remove superfluous information from your answer and make SURE that it doesn't go over 250 words

"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    # Example usage
    user_question = "Write about the future of AI agents and list some of the most important new startups in the field."
    human_message = [HumanMessage(content=user_question)]

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    res = chain.invoke(input={"messages": human_message})
    print(res)
