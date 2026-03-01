import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import re
import requests
from typing import Optional

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
) 

application_info = {
    "name": None,
    "email": None,
    "skills": None,
    # Career-related fields
    "current_role": None,
    "years_experience": None,
    "career_goals": None,
    "career_challenges": None,
}

def extract_application_info(text: str) -> str: 
    name_match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE) 
    email_match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)  
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

# Extract text from PDF using pypdf (more reliable)
def extract_text_from_pdf(uploaded_file):
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_info_from_cv(text: str):
    extracted_info = {"name": None, "email": None, "skills": None}
    name_match = re.search(r"(?:Full Name:|Name:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    email_match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)
    skills_match = re.search(r"Skills\s*-+\s*(.*?)\n(?:Projects|Certifications|$)", text, re.DOTALL)

    if name_match:
        extracted_info["name"] = name_match.group(1).strip()
    if email_match:
        extracted_info["email"] = email_match.group(0).strip()
    if skills_match:
        skills = skills_match.group(1).replace("\n", ", ").replace("\u2022", "").replace("-", "")
        extracted_info["skills"] = re.sub(r"\s+", " ", skills.strip())

    return extracted_info

def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return f"✅ You're ready! Name: {application_info['name']}, Email: {application_info['email']}, Skills: {application_info['skills']}."
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"⏳ Still need: {', '.join(missing)}. Please ask the user to provide this."

def generate_smart_search_queries() -> dict:
    """Generate contextual search queries based on career profile."""
    queries = {}
    role = application_info.get("current_role") or ""
    goals = application_info.get("career_goals") or ""
    challenges = application_info.get("career_challenges") or ""
    years = application_info.get("years_experience") or 0
    skills = application_info.get("skills") or ""
    
    # Courses & Learning
    if goals:
        queries["courses"] = f"online courses {goals} {role}".strip()
    elif skills:
        queries["courses"] = f"advanced courses {skills}".strip()
    else:
        queries["courses"] = "professional development courses"
    
    # Job Listings
    if role:
        queries["jobs"] = f"job listings {role} positions hiring"
    else:
        queries["jobs"] = "job board openings"
    
    # Skills & Technologies
    if goals:
        queries["skills"] = f"learn {goals} tutorials resources"
    elif challenges:
        keywords = challenges.split()[0] if challenges else "skills"
        queries["skills"] = f"improve {keywords} guide"
    else:
        queries["skills"] = "in-demand tech skills 2026"
    
    # Salary & Market Trends
    if role:
        queries["salary"] = f"{role} salary report 2026"
    else:
        queries["salary"] = "tech industry salary trends 2026"
    
    # Networking & Communities
    if skills:
        skill_list = skills.split(",")[0] if "," in skills else skills
        queries["networking"] = f"{skill_list} developer community forums"
    else:
        queries["networking"] = "professional networking communities"
    
    # Challenge-specific search
    if challenges:
        queries["challenges"] = f"solutions for {challenges[:50]}"
    
    return queries


# Define tools using the @tool decorator
@tool
def extract_info(text: str) -> str:
    """Use this to extract name, email, and skills from the user's message."""
    return extract_application_info(text)

@tool
def check_goal(text: str) -> str:
    """Check if name, email, and skills are provided. If not, tell the user what is missing."""
    return check_application_goal(text)

@tool
def search_web(query: str) -> str:
    """Search the web for information. Use this to find resources, courses, job market trends, and career development tips."""
    try:
        # Try using duckduckgo-search if available
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                if results:
                    output = f"🔍 Search results for '{query}':\n"
                    for i, result in enumerate(results[:5], 1):
                        output += f"{i}. **{result.get('title', 'No title')}**\n"
                        output += f"   {result.get('body', 'No description')[:200]}...\n"
                        output += f"   🔗 {result.get('href', 'No link')}\n\n"
                    return output
        except ImportError:
            # Fallback: Use simple web search via requests (mock approach)
            pass
        
        # Fallback message
        return f"💡 For '{query}': I recommend searching on Google, GitHub, Coursera, LinkedIn Learning, or industry-specific job boards like Stack Overflow Jobs or specialized platforms. Use keywords like '{query}' to find relevant resources."
    except Exception as e:
        return f"⚠️ Could not complete web search: {str(e)}. Try searching manually or I can provide general recommendations based on your profile."

tools = [extract_info, check_goal, search_web]

SYSTEM_PROMPT = """You are a helpful job application and career development assistant. 
Your goal is to collect the user's name, email, and skills. 
Use the tools provided to extract this information and check whether all required data is collected.
Once everything is collected, inform the user that the application info is complete and stop.
Additionally, ask the user about their career life: current role/title, years of experience, career goals, and main career challenges.
When career information is provided, offer concise, prioritized, and actionable suggestions to help the user improve their career. Suggestions should include short-term steps, longer-term learning recommendations, resume/LinkedIn improvements, networking tips, and resources (courses, books, communities) where applicable.
WHEN PROVIDING CAREER SUGGESTIONS, USE THE search_web TOOL to find:
- Online courses and certifications relevant to their skills and goals
- Current job market trends for their desired role
- Salary benchmarks and industry insights
- Networking communities and forums
- Learning resources (books, blogs, tutorials)
- Job boards and opportunities
Always cite sources from web search results when available. Be empathetic, practical, and provide clear next steps with measurable outcomes.
If the user asks about specific skills, technologies, or career paths, proactively search the web to provide current, relevant information.
"""

# Create agent using create_agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

# Streamlit UI
st.set_page_config(page_title="🎯 Job Application Assistant", layout="centered")
st.title("🧠 Goal-Based Agent: Job Application Assistant")
st.markdown("Tell me your **name**, **email**, and **skills** to complete your application!")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []  # LangChain message history
if "goal_complete" not in st.session_state:
    st.session_state.goal_complete = False
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False
if "application_summary" not in st.session_state:
    st.session_state.application_summary = ""
if "career_suggestions" not in st.session_state:
    st.session_state.career_suggestions = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Upload resume
st.sidebar.header("📤 Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt", "docx"])

if resume:
    st.sidebar.success("Resume uploaded!")
    if resume.type == "application/pdf":
        text = extract_text_from_pdf(resume)
    else:
        try:
            text = resume.read().decode("utf-8")
        except UnicodeDecodeError:
            # Try common encodings if UTF-8 fails
            try:
                text = resume.read().decode("latin-1")
            except:
                resume.seek(0)  # Reset file pointer
                text = resume.read().decode("utf-8", errors="replace")
    
    extracted = extract_info_from_cv(text)
    for key in application_info:
        if extracted[key]:
            application_info[key] = extracted[key]
    st.sidebar.info("🔍 Extracted info from resume:")
    for key, value in extracted.items():
        st.sidebar.markdown(f"**{key.capitalize()}:** {value}")

# Career life inputs
st.sidebar.header("💼 Career Life (Optional)")
current_role = st.sidebar.text_input("Current role / title", value=application_info.get("current_role") or "")
years_exp = st.sidebar.number_input("Years of experience", min_value=0, max_value=60, value=0)
career_goals = st.sidebar.text_area("Career goals", value=application_info.get("career_goals") or "", height=100)
career_challenges = st.sidebar.text_area("Career challenges", value=application_info.get("career_challenges") or "", height=100)

# Save career inputs into application_info
if current_role:
    application_info["current_role"] = current_role
application_info["years_experience"] = years_exp if years_exp and years_exp > 0 else None
if career_goals:
    application_info["career_goals"] = career_goals
if career_challenges:
    application_info["career_challenges"] = career_challenges

# Web Search Section
st.sidebar.header("🔍 Smart Web Search")

# Generate smart queries from career profile
smart_queries = generate_smart_search_queries()

if any([application_info.get("current_role"), application_info.get("career_goals"), application_info.get("skills")]):
    st.sidebar.info("📍 **Smart searches from your profile:**")
    
    # Create columns for auto-search buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🎓 Courses", use_container_width=True):
            with st.sidebar.spinner("Searching courses..."):
                try:
                    query = smart_queries.get("courses", "professional development")
                    search_result = search_web.invoke({"query": query})
                    st.session_state.search_results.append({
                        "query": query,
                        "category": "Courses & Learning",
                        "result": search_result
                    })
                    st.sidebar.success("✅ Done!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        
        if st.button("💼 Jobs", use_container_width=True):
            with st.sidebar.spinner("Searching jobs..."):
                try:
                    query = smart_queries.get("jobs", "job openings")
                    search_result = search_web.invoke({"query": query})
                    st.session_state.search_results.append({
                        "query": query,
                        "category": "Job Listings",
                        "result": search_result
                    })
                    st.sidebar.success("✅ Done!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    
    with col2:
        if st.button("💰 Salary", use_container_width=True):
            with st.sidebar.spinner("Searching salary data..."):
                try:
                    query = smart_queries.get("salary", "salary trends")
                    search_result = search_web.invoke({"query": query})
                    st.session_state.search_results.append({
                        "query": query,
                        "category": "Salary & Market Trends",
                        "result": search_result
                    })
                    st.sidebar.success("✅ Done!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        
        if st.button("🤝 Network", use_container_width=True):
            with st.sidebar.spinner("Searching communities..."):
                try:
                    query = smart_queries.get("networking", "communities")
                    search_result = search_web.invoke({"query": query})
                    st.session_state.search_results.append({
                        "query": query,
                        "category": "Networking & Communities",
                        "result": search_result
                    })
                    st.sidebar.success("✅ Done!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    
    # Skills search
    if st.button("🛠️ Skills", use_container_width=True):
        with st.sidebar.spinner("Searching skills..."):
            try:
                query = smart_queries.get("skills", "skill development")
                search_result = search_web.invoke({"query": query})
                st.session_state.search_results.append({
                    "query": query,
                    "category": "Skills & Technologies",
                    "result": search_result
                })
                st.sidebar.success("✅ Done!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    
    # Challenge-specific search
    if application_info.get("career_challenges"):
        if st.button("⚠️ Address Challenges", use_container_width=True):
            with st.sidebar.spinner("Searching solutions..."):
                try:
                    query = smart_queries.get("challenges", "problem solving")
                    search_result = search_web.invoke({"query": query})
                    st.session_state.search_results.append({
                        "query": query,
                        "category": "Challenge Solutions",
                        "result": search_result
                    })
                    st.sidebar.success("✅ Done!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
else:
    st.sidebar.info("💡 Fill in your career details above to enable smart searches!")

# Custom search
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Custom Search")
custom_query = st.sidebar.text_input("Enter custom search query")
if st.sidebar.button("🌐 Search"):
    if custom_query.strip():
        with st.sidebar.spinner("Searching..."):
            try:
                search_result = search_web.invoke({"query": custom_query})
                st.session_state.search_results.append({
                    "query": custom_query,
                    "category": "Custom",
                    "result": search_result
                })
                st.sidebar.success("✅ Search complete!")
            except Exception as e:
                st.sidebar.error(f"Search error: {e}")
    else:
        st.sidebar.warning("Please enter a search query.")


# Career suggestions button
if st.sidebar.button("💡 Get career suggestions"):
    prompt_text = (
        f"User career profile:\nCurrent role: {application_info.get('current_role')}\n"
        f"Years experience: {application_info.get('years_experience')}\n"
        f"Goals: {application_info.get('career_goals')}\n"
        f"Challenges: {application_info.get('career_challenges')}\n\n"
        "Please provide concise, prioritized, and actionable suggestions to improve the user's career. "
        "Include short-term next steps, upskilling recommendations, resume/LinkedIn tips, networking strategies, and useful resources. "
        "Search the web to find current courses, job market trends, and relevant resources for this profile."
    )
    try:
        response = agent.invoke({"messages": [HumanMessage(content=prompt_text)]})
        if isinstance(response, dict) and "messages" in response:
            resp_msgs = response["messages"]
            career_reply = resp_msgs[-1].content if resp_msgs else str(response)
        else:
            career_reply = str(response)
        st.session_state.career_suggestions = career_reply
        st.sidebar.success("Career suggestions generated — see main view below.")
    except Exception as e:
        st.sidebar.error(f"Error generating suggestions: {e}")

# Reset chat
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.messages = []  # Reset message history
    st.session_state.goal_complete = False
    st.session_state.download_ready = False
    st.session_state.application_summary = ""
    st.session_state.career_suggestions = ""
    st.session_state.search_results = []
    for key in application_info:
        application_info[key] = None
    st.rerun()

# Chat input
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    try:
        # Add user message to conversation history (stored in session state)
        st.session_state.messages.append(HumanMessage(content=user_input))
        extract_application_info(user_input)
        
        # Invoke agent with full conversation history from session state
        response = agent.invoke({"messages": st.session_state.messages})
        
        # Extract bot response
        if isinstance(response, dict) and "messages" in response:
            response_messages = response["messages"]
            # Add all response messages to session state history
            st.session_state.messages.extend(response_messages)
            bot_reply = response_messages[-1].content if response_messages else "No response"
        else:
            bot_reply = str(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        
        # Check goal status
        goal_status = check_application_goal("check")
        
        if "you're ready" in goal_status.lower():
            st.session_state.goal_complete = True
            # Build application summary including optional career info
            summary_lines = []
            summary_lines.append(f"✅ Name: {application_info.get('name')}")
            summary_lines.append(f"📧 Email: {application_info.get('email')}")
            summary_lines.append(f"🛠️ Skills: {application_info.get('skills')}")
            if application_info.get('current_role'):
                summary_lines.append(f"💼 Current role: {application_info.get('current_role')}")
            if application_info.get('years_experience'):
                summary_lines.append(f"⏳ Years experience: {application_info.get('years_experience')}")
            if application_info.get('career_goals'):
                summary_lines.append(f"🎯 Career goals: {application_info.get('career_goals')}")
            if application_info.get('career_challenges'):
                summary_lines.append(f"⚠️ Career challenges: {application_info.get('career_challenges')}")
            if st.session_state.career_suggestions:
                summary_lines.append("\n💡 Career suggestions:\n" + st.session_state.career_suggestions)

            st.session_state.application_summary = "\n".join(summary_lines)
            st.session_state.download_ready = True
    except Exception as e:
        st.error(f"Error: {e}")

# Chat UI with avatars
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Final message
if st.session_state.goal_complete:
    st.success("🎉 All information collected! You're ready to apply!")

# Show career suggestions if available
if st.session_state.career_suggestions:
    st.subheader("💡 Career Suggestions")
    st.markdown(st.session_state.career_suggestions)

# Show search results if available
if st.session_state.search_results:
    st.subheader("🌐 Web Search Results")
    for search_item in st.session_state.search_results:
        category_badge = f"📂 {search_item.get('category', 'Search')}" if 'category' in search_item else "📂 Search"
        with st.expander(f"{category_badge} • {search_item['query']}"):
            st.markdown(search_item['result'])

# Download summary
if st.session_state.download_ready:
    st.download_button(
        label="📥 Download Application Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt",
        mime="text/plain"
    )