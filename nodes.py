from .llm import getLLm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .prompts import plannerPrompt, codingAgent, educationAgent, researchAgent

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
        "reasoning": chain.invoke(query=state["query"])["reasoning"]
    }

def decomposeNode(state):
    # This function would implement the coding agent logic, similar to plannerNode but using the codingAgent prompt and potentially a different LLM configuration.
    if state["category"] != "CODING":
        sysPrompt=codingAgent
    elif state["category"] == "EDUCATION":
        sysPrompt=educationAgent
    elif state["category"] == "RESEARCH":
        sysPrompt=researchAgent

    prompt=ChatPromptTemplate.from_messages([
            ("system",sysPrompt),
        ("human","user query: {query}\n reasoning for decomposition: {reasoning}")
    ])
    chain = prompt | llm | parser

    return {
        "subtasks": chain.invoke(query=state["query"], reasoning=state["reasoning"])["subtasks"]
    }


