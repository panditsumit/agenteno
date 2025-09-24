import streamlit as st
import asyncio
import json
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Define tool functions
def load_books():
    with open("data/books.json", "r", encoding="utf-8") as f:
        return json.load(f)

async def search_book_by_author(author: str) -> str:
    books = load_books()
    filtered = [b for b in books if b.get("author") == author]
    return json.dumps(filtered or {"message": "No books found."}, ensure_ascii=False, indent=2)

async def search_book_by_category(category: str) -> str:
    books = load_books()
    filtered = [b for b in books if b.get("category") == category]
    return json.dumps(filtered or {"message": "No books found."}, ensure_ascii=False, indent=2)

# Azure OpenAI client
model_client = AzureOpenAIChatCompletionClient(
    model=AZURE_OPENAI_MODEL,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION
)

# Initialize agent with tools
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[search_book_by_author, search_book_by_category],
    system_message="You can search books by author or category."
)

# Helper to run async calls
def run_agent(query: str) -> str:
    async def _run():
        resp = await agent.on_messages([
            TextMessage(content=query, source="User")
        ], CancellationToken())
        return resp.chat_message.to_text()
    return asyncio.run(_run())

# Streamlit UI
st.title("Book Search Assistant")
mode = st.radio("Choose search mode:", ["By Author", "By Category"])
input_label = "Author name:" if mode == "By Author" else "Category name:"
user_input = st.text_input(input_label)

if st.button("Search Books"):
    if not user_input.strip():
        st.error("Please enter a value.")
    else:
        # Construct prompt for agent
        if mode == "By Author":
            prompt = f"search_book_by_author:{user_input}"  # function syntax recognized by agent
        else:
            prompt = f"search_book_by_category:{user_input}"
        result = run_agent(prompt)
        st.text(result)

# Ensure script only runs in Streamlit context
if __name__ == "__main__":
    import sys
    if "streamlit" not in sys.modules:
        print("This script is intended to be run with Streamlit.")
        sys.exit(1)
    else:
        st.write("Running in Streamlit context.")
