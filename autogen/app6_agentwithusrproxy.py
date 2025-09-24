from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import  TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.ui import Console
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv
import os
import asyncio

load_dotenv(override=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

async def main(task:str, cancellation_token:CancellationToken):
    model_client = AzureOpenAIChatCompletionClient(
        model=AZURE_OPENAI_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment= AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        max_tokens=2000,
        temperature=0.7
    )

    def user_input_func(prompt: str) -> str:
        print("Enter (APPROVE) to stop, type (no) to regenerate new story:")
        return input(prompt)
    
    # Create the agents.
    assistant = AssistantAgent("assistant", model_client=model_client)
    user_proxy = UserProxyAgent("user_proxy", input_func=user_input_func)  
    # Use input() to get user input from console.
    # The user_proxy agent will take the user input and send it to the assistant agent.
    # The assistant agent will generate a response based on the user input.
    # The user_proxy agent will then send the response back to the user.
    # The user_proxy agent will also handle the termination condition.
    
    # Create the termination condition which will end the conversation when the user says "APPROVE".
    termination = TextMentionTermination("APPROVE")

    # Create the team.
    # The team will consist of the assistant and user_proxy agents.
    # The termination condition will be used to terminate the conversation when the user says "APPROVE".
    team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination)

    # Run the conversation and stream to the console.
    # The task is to write a short story about a monkey and farmer.
    # The cancellation token will be used to cancel the conversation.
    stream = team.run_stream(task=task, cancellation_token=cancellation_token)
    await Console(stream)
    await model_client.close()


cancellation_token = CancellationToken()
asyncio.run(main("Write a short story about a monkey and farmer", cancellation_token))
