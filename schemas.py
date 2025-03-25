from typing import Optional, List

from langchain_core.pydantic_v1 import BaseModel, Field

class Reflection(BaseModel):
    """
    A class to represent a reflection.
    """

    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """
    A class to represent an answer to a question.
    """

    answer: str = Field(description="Answer to the question")
    reflection: Reflection = Field(description="Your reflection on the answer")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")