from openai import OpenAI
import os
from dotenv import load_dotenv
from services.ai_tools import search_internet, search_medlineplus_web
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


instructions="""
        You are a helpful medical assistant. Provide clear, accurate, and easy-to-understand health information.

        - Use a professional and supportive tone
        - Avoid unnecessary medical jargon; explain terms when needed
        - Do not provide diagnoses or replace professional medical advice
        - Encourage users to consult a qualified healthcare professional for serious concerns
        - Focus on general guidance and educational information

"""

tools = [
    {
        "type": "function",
        "name": "search_internet",
        "description": "Search the internet for relevant information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }, 
    {
     "type": "function",
        "name": "earch_medlineplus_web",
        "description": "Search the internet for relevant information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

available_tools = {
    "search_internet": search_internet,
    "search_medlineplus_web": search_medlineplus_web
}

def ask_llm(user_question: str, search_results: str, context: str, search_medlineplus_web) -> str:

    response = client.responses.create(
        model="gpt-4.1-mini", 
        instructions=instructions,
        tools=tools,
        input=f"""
        Question:
        {user_question}

        Document context:
        {context}

        Search results:
        {search_results}
       
        Additional Information MedicalPlus: 
        {search_medlineplus_web}


        """
          )

    llm_response = response.output_text

     
    if response.output[0].type == "tool_call":
        tool_name = response.output[0].name
        args = json.loads(response.output[0].arguments)

        result = available_tools[tool_name](**args)

        second_response = client.responses.create(
            model="gpt-4.1-mini",
            input=result
        )

        return second_response.output_text
    
    return llm_response 