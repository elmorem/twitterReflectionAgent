from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
        "Always provide detailed recommendations, including requests for length, virality,style, and tone"),

        MessagesPlaceholder(variable_name="messages"),
    ]

)

generation_prompt = ChatPromptTemplate.from_messages(

    [
        ("system", "You are a viral twitter influencer assistant tasked with writing excellent twitter posts"
        "Generate the best twitter post possible for the user's request."
         "If the user provides criqtiue, respond with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model_name="gpt-4-mini")
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
