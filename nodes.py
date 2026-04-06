from .llm import getLLm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .prompts import plannerPrompt

llm=getLLm()

parser=JsonOutputParser() 

def plannerNode(state):

    prompt=ChatPromptTemplate.from_messages([
            ("system",plannerPrompt),
        ("human","user query: {query}")
    ])

    chain = prompt | llm | parser
    return {
        "category": chain.run(query=state["query"])["category"],
        "needs_decomposition": chain.run(query=state["query"])["needs_decomposition"],
        "reasoning": chain.run(query=state["query"])["reasoning"]
    }