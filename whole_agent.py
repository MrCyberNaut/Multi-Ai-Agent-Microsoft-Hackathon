import getpass
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from amadeus import Client
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch
from langchain_ollama import ChatOllama
ollama = ChatOllama(model="llama3.2")
# Load environment variables from .env file
load_dotenv()

input_text = input("Enter your query here: ")

# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.aimlapi.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize ChatOpenAI for agents
chat_model = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.aimlapi.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)

# Initialize Amadeus client using environment variables
amadeus_client = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID"),
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET")
)

# Rebuild the models after Client is defined
AmadeusToolkit.model_rebuild()
AmadeusClosestAirport.model_rebuild()
AmadeusFlightSearch.model_rebuild()

# Create AmadeusToolkit instance with client
toolkit = AmadeusToolkit(client=amadeus_client)
tools = toolkit.get_tools()

# Create agents with the tools
flight_assistant = create_react_agent(
    model=ChatOllama(model="llama3.2"),
    tools=tools,
    prompt="You are a flight booking assistant",
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model=ChatOllama(model="llama3.2"),
    tools=tools,
    prompt="You are a hotel booking assistant",
    name="hotel_assistant"
)

supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatOllama(model="llama3.2"),
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    )
).compile()

# Prepare the messages for the conversation
messages = [
    {"role": "system", "content": "You are a helpful travel planning assistant that can help with flights and hotels."},
    {"role": "user", "content": input_text}
]

try:
    for chunk in supervisor.stream({"messages": messages}):
        # Iterate over the top-level keys like 'supervisor', 'flight_assistant', etc.
        for assistant, data in chunk.items():
            if isinstance(data, dict) and "messages" in data:
                for msg in data["messages"]:
                    if hasattr(msg, "content") and msg.content is not None:
                        print(f"{assistant}: {msg.content}")
                        print("\n")
                    elif isinstance(msg, dict) and "content" in msg and msg["content"] is not None:
                        print(f"{assistant}: {msg['content']}")
                        print("\n")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please try a different query or check your API configuration.")