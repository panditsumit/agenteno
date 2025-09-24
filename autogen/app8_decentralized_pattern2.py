import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# ---------------- Environment Setup ----------------
load_dotenv(override=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ---------------- Helper Functions ----------------
def run_async(coro, timeout=60):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()

async def triage_app(task: str) -> list:
    client = AzureOpenAIChatCompletionClient(
        model=AZURE_OPENAI_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Specialist agents
    orders_agent = AssistantAgent(
        name="Orders_Agent",
        model_client=client,
        system_message="You are an order support specialist. Answer questions about order tracking, shipping, and delivery. End your final reply with 'FINAL_ANSWER'."
    )

    sales_agent = AssistantAgent(
        name="Sales_Agent",
        model_client=client,
        system_message="You are a sales specialist. Answer questions about product pricing, availability, and promotions. End your final reply with 'FINAL_ANSWER'."
    )

    issues_agent = AssistantAgent(
        name="Issues_Repairs_Agent",
        model_client=client,
        system_message="You are a repair and issues specialist. Handle product complaints, damages, or repairs. End your final reply with 'FINAL_ANSWER'."
    )

    # Triage agent
    triage_agent = AssistantAgent(
        name="Support_Agent",
        model_client=client,
        system_message=(
            "You are a triage bot. When a user query comes:\n"
            "- If it mentions order, shipping, delivery → Forward to Orders_Agent.\n"
            "- If it mentions price, buying, discount → Forward to Sales_Agent.\n"
            "- If it mentions broken, damage, repair → Forward to Issues_Repairs_Agent.\n"
            "After forwarding, do not reply yourself."
        )
    )

    # Termination Condition
    termination = TextMentionTermination("FINAL_ANSWER")

    # Team (Decentralized)
    team = RoundRobinGroupChat(
        [triage_agent, orders_agent, sales_agent, issues_agent],
        termination_condition=termination
    )

    result = await team.run(task=task)
    await client.close()
    return result.messages

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📦 Customer Triage Bot", layout="centered")
st.title("📦 Customer Support Triage Bot")

user_query = st.text_input("Ask your question (e.g., 'Where is my order?'):")

if st.button("Ask Specialist"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Triage bot analyzing and routing..."):
            messages = run_async(triage_app(user_query))

            st.subheader("📢 Response")
            for m in messages:
                role = getattr(m, "source", getattr(m, "role", ""))
                content = getattr(m, "content", "")
                st.markdown(f"**{role}:** {content.replace('FINAL_ANSWER', '').strip()}")
