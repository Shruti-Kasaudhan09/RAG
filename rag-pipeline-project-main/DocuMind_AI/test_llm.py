from utils.llm import load_llm

llm = load_llm()

response = llm.invoke("What is Machine Learning?")

print(response.content)