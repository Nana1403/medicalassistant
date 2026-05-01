from services.llm_services import ask_llm
from services.ai_tools import search_internet, search_medlineplus_web
from services.rag import rag

def main():
    print()
    print("""Hello 👋 welcome, I am an AI-powered Medical Assistant designed to provide 
      information on health-related queries. Please enter your question below to begin.""")
    print()

    while True: 
         user_question = input("⭐️ Enter your question her (If finished type (exit) or e):")

         if user_question.lower().strip() in ["exit", "e"]:
            break
         
         context = rag(user_question)
         search_net = search_internet(user_question)
         search_mp = search_medlineplus_web(user_question)
         
         response = ask_llm(user_question, search_net, context, search_mp)
        
         print()
         print("---" * 80)
         print(f" 🤖 : {response}") 
         print("---" * 80)
         print()

main()
