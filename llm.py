from langchain_groq import ChatGroq

def getLLM():
    llm=ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_tokens=2048,
    )
    return llm

