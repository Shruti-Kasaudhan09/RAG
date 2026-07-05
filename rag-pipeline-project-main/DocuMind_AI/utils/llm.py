import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_llm():

    llm = ChatGoogleGenerativeAI(

        model="gemini-2.5-flash",

        temperature=0

    )

    return llm