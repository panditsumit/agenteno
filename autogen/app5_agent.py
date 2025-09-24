import streamlit as st
import asyncio
import json
from dotenv import load_dotenv
import os
import aiohttp
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
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Configure Azure OpenAI client
model_client = AzureOpenAIChatCompletionClient(
    model=AZURE_OPENAI_MODEL,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Async weather function
async def get_current_weather(city: str) -> str:
    """Fetch weather data from OpenWeatherMap API"""
    if not OPENWEATHER_API_KEY:
        return json.dumps({"error": "Missing API credentials"})
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                return json.dumps({
                    "city": data["name"],
                    "temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "conditions": data["weather"][0]["description"],
                    "wind": data["wind"]["speed"]
                }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

# Initialize agent with weather tool
weather_agent = AssistantAgent(
    name="WeatherExpert",
    model_client=model_client,
    tools=[get_current_weather],
    system_message="""You are a weather specialist. Use get_current_weather for data.
    Format responses as: City | Conditions | Temperature(°C) | Humidity(%) | Wind(m/s)
    Add relevant weather insights.""",
)

# Async execution helper
def run_agent_query(query: str) -> str:
    async def _run():
        response = await weather_agent.on_messages([
            TextMessage(content=query, source="User")
        ], CancellationToken())
        return response.chat_message.to_text()
    return asyncio.run(_run())

# Streamlit interface
st.title("AI Weather Assistant")
st.markdown("### Get real-time weather updates")

city_input = st.text_input("Enter city name:", placeholder="e.g., London, Tokyo")
search_btn = st.button("Get Weather Report")

if search_btn:
    if not city_input.strip():
        st.error("Please enter a valid city name")
    else:
        with st.spinner("Analyzing weather patterns..."):
            try:
                # Format query for agent tool recognition
                result = run_agent_query(f"get_current_weather:{city_input}")
                weather_data = json.loads(result)
                
                if "error" in weather_data:
                    st.error(f"Error: {weather_data['error']}")
                else:
                    st.subheader(f"Weather in {weather_data['city']}")
                    cols = st.columns(4)
                    metrics = [
                        ("🌤️ Conditions", weather_data["conditions"]),
                        ("🌡️ Temperature", f"{weather_data['temp']}°C"),
                        ("💧 Humidity", f"{weather_data['humidity']}%"),
                        ("🍃 Wind Speed", f"{weather_data['wind']} m/s")
                    ]
                    
                    for col, (label, value) in zip(cols, metrics):
                        col.metric(label, value)
                        
            except json.JSONDecodeError:
                st.error("Invalid response format")
            except Exception as e:
                st.error(f"Service error: {str(e)}")
