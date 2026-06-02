from app.chain.steps import PromptBuilder, LLMRunner, ResponseParser
from app.chain.runnable import RunnableSequence

oraklet = RunnableSequence(
    PromptBuilder(),
    LLMRunner(),
    ResponseParser()
)
