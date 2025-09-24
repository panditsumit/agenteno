import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Load environment variables
load_dotenv(override=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Safe async runner
def run_async(coro, timeout=50):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()

# Translation tool implementations (simple functions)
def spanish_tool_fn(input: str) -> str:
    return f"[Spanish] Hola! Translation of: '{input}'"

def french_tool_fn(input: str) -> str:
    return f"[French] Bonjour! Translation of: '{input}'"

def italian_tool_fn(input: str) -> str:
    return f"[Italian] Ciao! Translation of: '{input}'"

# Unified assistant using simple tools
async def run_translator_agent(text: str):
    client = AzureOpenAIChatCompletionClient(
        model=AZURE_OPENAI_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    agent = AssistantAgent(
        name="Translator",
        model_client=client,
        tools=[spanish_tool_fn, french_tool_fn, italian_tool_fn],
        system_message=(
            "You are a multilingual assistant. Given an English sentence, call all your translation tools to produce translations into Spanish, French, and Italian."
        )
    )

    user_msg = TextMessage(content=text, source="user")
    response = await agent.on_messages([user_msg], cancellation_token=CancellationToken())
    await client.close()
    return response.chat_message.to_text()

# Streamlit UI
st.set_page_config(page_title="🌍 Unified Translator Agent", layout="centered")
st.title("🌍 Unified Translator with Tools")

user_input = st.text_input("Enter sentence to translate into Spanish, French, and Italian:")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter a valid sentence.")
    else:
        with st.spinner("Translating via tools..."):
            reply = run_async(run_translator_agent(user_input))
            st.subheader("🌟 Translations")
            st.write(reply)
