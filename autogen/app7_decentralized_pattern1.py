import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination

# Load environment variables
load_dotenv(override=True)

# Azure OpenAI Config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Safe async runner
# This is a workaround for Streamlit's async limitations
# Streamlit's run_async is not available in all versions
# and can cause issues with asyncio event loops.
# This function creates a new event loop and runs the coroutine until completion.
def run_async(coro, timeout=50):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()

# Core translation logic
async def translate_with_manager(task: str) -> list:
    client = AzureOpenAIChatCompletionClient(
        model=AZURE_OPENAI_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Specialized agents
    spanish_agent = AssistantAgent(
        name="Spanish_Agent",
        model_client=client,
        system_message="Translate any given English sentence into Spanish."
    )
    french_agent = AssistantAgent(
        name="French_Agent",
        model_client=client,
        system_message="Translate any given English sentence into French."
    )
    italian_agent = AssistantAgent(
        name="Italian_Agent",
        model_client=client,
        system_message="Translate any given English sentence into Italian."
    )

    # Manager agent
    manager = AssistantAgent(
        name="Manager",
        model_client=client,
        system_message=(
            "You are a manager. Your task is:\n"
            "1. Assign translation tasks to Spanish_Agent, French_Agent, Italian_Agent.\n"
            "2. After collecting all translations, reply with: 'ALL_TRANSLATIONS_COMPLETED'."
        )
    )

    # GroupChat: Manager + Specialists
    termination = TextMentionTermination("ALL_TRANSLATIONS_COMPLETED")

    team = RoundRobinGroupChat(
        [manager, spanish_agent, french_agent, italian_agent],
        termination_condition=termination
    )

    
    result = await team.run(task=task)
    #await client.close()
    return result.messages

# Streamlit UI
st.set_page_config(page_title="🌍 Manager Pattern Translator", layout="centered")
st.title("🌍 Manager Pattern Multilingual Translator")

user_task = st.text_input("Enter sentence to translate into Spanish, French, and Italian:")

if st.button("Translate"):
    if not user_task.strip():
        st.warning("Please enter a valid sentence.")
    else:
        with st.spinner("Manager and specialists are working..."):
            messages = run_async(translate_with_manager(user_task))

            st.subheader("🌟 Translations Collected")
            for m in messages:
                role = getattr(m, "source", getattr(m, "role", ""))
                content = getattr(m, "content", "")
                st.markdown(f"**{role}:** {content}")
