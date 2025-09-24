# Import necessary libraries
import streamlit as st  # Streamlit for building the web app interface
from openai import AzureOpenAI  # Azure OpenAI client for interacting with the OpenAI API
from dotenv import load_dotenv  # To load environment variables from a .env file
import os  # For accessing environment variables

# Load environment variables from the .env file
load_dotenv()

# Initialize the Azure OpenAI client with credentials and configuration
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # API key for authentication
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # API version to use
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Azure endpoint URL
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    # Deployment name for the model
)

# Set the title of the Streamlit app
st.title("Azure OpenAI Model Demo")

# Create a text input field for the user to enter their message
user_input = st.text_input("Enter your message:")

# Check if the user has entered any input
if user_input:
    # Call the Azure OpenAI model to generate a response
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # Specify the model deployment name
        stream=False,  #Disable streaming for simplicity | If set to true, the model response data will be streamed to the client as it is generated using server-sent events. 
        messages=[{"role": "user", "content": user_input}]  # Pass the user input as a message
    )
    st.write("**Model Response:**", response.choices[0].message.content)
