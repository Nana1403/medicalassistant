import requests
# from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool


search = DuckDuckGoSearchRun()

def search_internet(topic: str) -> str:
    """Search the internet for relevant information on a topic"""
    return search.run(topic)

def search_medlineplus_web(topic: str) -> str:
    """Search MedlinePlus Web for relevant information on a topic"""

    url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=title:{topic}&rettype=topic"

    response = requests.get(url)

    info = response.text

    return info 