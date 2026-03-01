import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import re
import fitz


load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
) 

application_info = {
    "name": None,
    "email": None,
    "skills": None
}

# Store conversation history
conversation_history = []

def extract_application_info(text: str) -> str: 
    name_match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE) 
    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)  
    skills_match = re.search(r"(?:skills are|i know|i can use|i am)\s+(.+)", text, re.IGNORECASE) 

    response = [] 

    if name_match: 
        application_info["name"] = name_match.group(1).title()
        response.append("✅ Name saved.") 


    if email_match:
        application_info["email"] = email_match.group(0)
        response.append("✅ Email saved.")
    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()
        response.append("✅ Skills saved.")

    if not any([name_match, email_match, skills_match]):
        return "❓ I couldn't extract any info. Could you please provide your name, email, or skills?"

    return " ".join(response) + " Let me check what else I need."



def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return f"✅ You're ready! Name: {application_info['name']}, Email: {application_info['email']}, Skills: {application_info['skills']}."
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"⏳ Still need: {', '.join(missing)}. Please ask the user to provide this."

# Define tools using the @tool decorator
@tool
def extract_info(text: str) -> str:
    """Use this to extract name, email, and skills from the user's message."""
    return extract_application_info(text)

@tool
def check_goal(text: str) -> str:
    """Check if name, email, and skills are provided. If not, tell the user what is missing."""
    return check_application_goal(text)

tools = [extract_info, check_goal]

SYSTEM_PROMPT = """You are a helpful job application assistant. 
Your goal is to collect the user's name, email, and skills. 
Use the tools provided to extract this information and check whether all required data is collected.
Once everything is collected, inform the user that the application info is complete and stop.
"""

# Create agent using create_agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


print("📝 Hi! I'm your job application assistant. Please tell me your name, email, and skills.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Bye! Good luck.")
        break

    response = None
    try:
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Invoke agent with full conversation history
        response = agent.invoke({"messages": conversation_history})
        
        # Extract and display bot output
        if isinstance(response, dict) and "messages" in response:
            # Get all messages from response and add to history
            response_messages = response["messages"]
            conversation_history.extend(response_messages)
            
            # Get the last assistant message to display
            last_message = response_messages[-1]
            output = getattr(last_message, "content", str(last_message))
            print("Bot:", output)
        else:
            print("Bot:", response)
    except Exception as e:
        print(f"Error: {e}")
        # Remove the user message from history if there was an error
        if conversation_history and isinstance(conversation_history[-1], HumanMessage):
            conversation_history.pop()
        continue

    # If goal achieved, stop
    if response and "you're ready" in str(response).lower():
        print("🎉 Application info complete!")
        break