#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Instantiate Ollama client pointing to local server
ollama_client = OllamaChatCompletionClient(
    model=OLLAMA_MODEL,                        # e.g., "ollama-model"
    host=OLLAMA_HOST,
)

# # Validate that required credentials are set
# if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
#             AZURE_OPENAI_MODEL, AZURE_OPENAI_DEPLOYMENT_NAME,
#             GEMINI_API_KEY, GEMINI_MODEL_NAME, OLLAMA_MODEL]):
#     st.error("One or more environment variables are missing. Please check your .env file.")
#     st.stop()

# Instantiate Azure OpenAI client for chat completions
# azure_client = AzureOpenAIChatCompletionClient(
#     model=AZURE_OPENAI_MODEL,                  # e.g., "gpt-4o-mini"
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version=AZURE_OPENAI_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
#     max_tokens=1000,
#     temperature=0.7,
# )

# Instantiate Gemini client for chat completions
# gemini_client = OpenAIChatCompletionClient(
#     model=GEMINI_MODEL_NAME,                  # e.g., "gemini-pro"
#     api_key=GEMINI_API_KEY,
# )

# Instantiate Ollama client pointing to local server
ollama_client = OllamaChatCompletionClient(
    model=OLLAMA_MODEL,                        # e.g., "ollama-model"
    host=OLLAMA_HOST,
)

# Streamlit UI setup
st.title("Multi-Model Chat Demo: Azure OpenAI | Gemini | Ollama")

# Text input for user's message
user_input = st.text_input("Enter your message:")

if user_input:
    # Echo user message
    st.write("**User:**", user_input)

    # 1️⃣ Azure OpenAI response
    # Run the async call in the event loop
    # azure_response = asyncio.run(
    #     azure_client.create(
    #         messages=[UserMessage(content=user_input, source="user")]
    #     )
    # )
    # st.write("**Azure OpenAI Response:**", azure_response.content)

    # 2️⃣ Gemini response
    # gemini_response = asyncio.run(
    #     gemini_client.create(
    #         messages=[UserMessage(content=user_input, source="user")]
    #     )
    # )
    # st.write("**Gemini Response:**", gemini_response.content)

    # 3️⃣ Ollama response
    ollama_response = asyncio.run(
        ollama_client.create(
            messages=[UserMessage(content=user_input, source="user")]
        )
    )
    st.write("**Ollama Response:**", ollama_response.content)
