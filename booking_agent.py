"""Consolidated multi-agent travel booking system."""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, START, END, Parallel
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 1500
CACHE_DIR = "cache"
HITL_ENABLED = True
HITL_TIMEOUT = 300
SERPAPI_TIMEOUT = 30
SERPAPI_CACHE_DURATION = 3600

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# State Management
class TravelState(BaseModel):
    """State for the travel assistant."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    flight_options: Optional[List[Dict[str, Any]]] = None
    hotel_options: Optional[List[Dict[str, Any]]] = None
    itinerary: Optional[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    human_feedback: List[str] = Field(default_factory=list)

# Agent Prompts
SUPERVISOR_PROMPT = """You are the supervisor agent for a travel planning system. Your responsibilities include:

1. Initial request analysis and task delegation
2. Coordination between specialist agents
3. Error handling and recovery
4. Managing human-in-the-loop interactions
5. Ensuring all travel requirements are met

Use the available tools to search for travel options and coordinate with other agents.
Always validate critical decisions with human operators when HITL is enabled.
Maintain conversation context and handle transitions between agents smoothly."""

FLIGHT_AGENT_PROMPT = """You are a flight booking specialist agent. Your key responsibilities include:

1. Searching for optimal flight options using SerpAPI
2. Analyzing and filtering results based on user preferences
3. Presenting options clearly for human selection
4. Handling booking-related queries and issues
5. Coordinating with other agents for complete travel planning

Always verify flight availability and pricing before presenting options.
Get human approval for final flight selections.
Handle errors gracefully and suggest alternatives when needed."""

HOTEL_AGENT_PROMPT = """You are a hotel booking specialist agent. Your key responsibilities include:

1. Finding suitable accommodations using SerpAPI
2. Matching hotels to user preferences and budget
3. Verifying availability and amenities
4. Presenting options clearly for human selection
5. Coordinating with other agents for complete travel planning

Consider location, ratings, and reviews when selecting options.
Get human approval for final hotel selections.
Handle errors gracefully and suggest alternatives when needed."""

ITINERARY_AGENT_PROMPT = """You are an itinerary planning specialist. Your key responsibilities include:

1. Creating comprehensive travel schedules
2. Incorporating flight and hotel bookings
3. Suggesting activities and attractions
4. Optimizing timing and logistics
5. Getting human approval for plans

Use SerpAPI to find local attractions and activities.
Consider travel times, check-in/out times, and local conditions.
Allow for flexibility and human customization of plans."""

# SerpAPI Utility Functions
def _get_cache_path(query_hash: str) -> str:
    """Get the cache file path for a query."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{query_hash}.json")

def _cache_results(query_hash: str, results: Dict[str, Any]) -> None:
    """Cache search results."""
    cache_path = _get_cache_path(query_hash)
    with open(cache_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().timestamp(),
            'results': results
        }, f)

def _get_cached_results(query_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached results if they exist and are not expired."""
    cache_path = _get_cache_path(query_hash)
    if not os.path.exists(cache_path):
        return None
    
    with open(cache_path, 'r') as f:
        cached = json.load(f)
    
    if datetime.now().timestamp() - cached['timestamp'] > SERPAPI_CACHE_DURATION:
        return None
    
    return cached['results']

def search_flights_serp(origin: str, destination: str, departure_date: str, return_date: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for flights using SerpAPI."""
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY is not set in environment variables")
    
    query = f"flights from {origin} to {destination} on {departure_date}"
    query_hash = f"flights_{origin}_{destination}_{departure_date}"
    
    cached = _get_cached_results(query_hash)
    if cached:
        return cached
    
    params = {
        "engine": "google_flights",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us",
        "departure_id": origin,
        "arrival_id": destination,
        "date": departure_date,
        "return_date": return_date,
        "currency": "USD",
        "type": "2"  # Round trip
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    flights = []
    if 'flights_results' in results:
        for flight in results['flights_results']:
            flight_info = {
                "airline": flight.get('airline', 'Unknown'),
                "flight_number": flight.get('flight_id', 'Unknown'),
                "price": flight.get('price', 'Unknown'),
                "departure": flight.get('departure_time', 'Unknown'),
                "arrival": flight.get('arrival_time', 'Unknown'),
                "duration": flight.get('duration', 'Unknown'),
                "stops": flight.get('stops', 'Unknown')
            }
            if budget and flight.get('price'):
                if float(flight['price'].replace('$', '')) <= float(budget.replace('$', '')):
                    flights.append(flight_info)
            else:
                flights.append(flight_info)
    
    _cache_results(query_hash, flights)
    return flights

def search_hotels_serp(location: str, check_in: str, check_out: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for hotels using SerpAPI."""
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY is not set in environment variables")
    
    query = f"hotels in {location} check in {check_in} check out {check_out}"
    query_hash = f"hotels_{location}_{check_in}_{check_out}"
    
    cached = _get_cached_results(query_hash)
    if cached:
        return cached
    
    params = {
        "engine": "google_hotels",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us",
        "check_in_date": check_in,
        "check_out_date": check_out,
        "location": location,
        "currency": "USD"
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    hotels = []
    if 'hotels_results' in results:
        for hotel in results['hotels_results']:
            hotel_info = {
                "name": hotel.get('name', 'Unknown'),
                "price": hotel.get('price', 'Unknown'),
                "rating": hotel.get('rating', 0),
                "amenities": hotel.get('amenities', []),
                "address": hotel.get('address', 'Unknown'),
                "website": hotel.get('website', 'Unknown'),
                "reviews": hotel.get('reviews', [])
            }
            if budget and hotel.get('price'):
                if float(hotel['price'].replace('$', '').split('/')[0]) <= float(budget.replace('$', '')):
                    hotels.append(hotel_info)
            else:
                hotels.append(hotel_info)
    
    _cache_results(query_hash, hotels)
    return hotels

def get_destination_info_serp(destination: str) -> Dict[str, Any]:
    """Get destination information using SerpAPI."""
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY is not set in environment variables")
    
    query = f"travel guide {destination} tourist attractions"
    query_hash = f"destination_{destination}"
    
    cached = _get_cached_results(query_hash)
    if cached:
        return cached
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us",
        "location": "United States",
        "google_domain": "google.com",
        "num": 10
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    info = {
        "name": destination,
        "description": "",
        "attractions": [],
        "weather": "",
        "best_time_to_visit": "",
        "local_tips": []
    }
    
    if 'knowledge_graph' in results:
        kg = results['knowledge_graph']
        info.update({
            "description": kg.get('description', ''),
            "weather": kg.get('weather', ''),
            "timezone": kg.get('timezone', ''),
            "currency": kg.get('currency', ''),
            "languages": kg.get('languages', [])
        })
    
    if 'organic_results' in results:
        attractions = []
        tips = []
        for result in results['organic_results'][:5]:
            if 'attractions' in result['title'].lower():
                attractions.extend(result.get('attractions', []))
            if 'tips' in result['title'].lower() or 'guide' in result['title'].lower():
                tips.append(result['snippet'])
        
        info['attractions'] = attractions[:5]
        info['local_tips'] = tips[:3]
    
    _cache_results(query_hash, info)
    return info

# Human-in-the-Loop Functions
def get_human_approval(data: Dict[str, Any], context: str) -> bool:
    """Get human approval for a decision."""
    if not HITL_ENABLED:
        return True
    
    print("\n=== Human Approval Required ===")
    print(f"Context: {context}")
    print("Data to approve:")
    print(json.dumps(data, indent=2))
    
    response = input("\nDo you approve? (yes/no): ").lower().strip()
    return response in ['y', 'yes']

def get_human_selection(options: List[Dict[str, Any]], context: str) -> Optional[Dict[str, Any]]:
    """Get human selection from a list of options."""
    if not HITL_ENABLED:
        return options[0] if options else None
    
    print("\n=== Human Selection Required ===")
    print(f"Context: {context}")
    print("\nAvailable options:")
    
    for i, option in enumerate(options, 1):
        print(f"\n{i}. {json.dumps(option, indent=2)}")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def handle_error(error: Union[str, Dict[str, Any]], context: str) -> Dict[str, Any]:
    """Handle errors with human intervention."""
    if not HITL_ENABLED:
        return {"status": "error", "message": str(error)}
    
    print("\n=== Error Resolution Required ===")
    print(f"Context: {context}")
    print(f"Error: {error}")
    
    resolution = input("How would you like to resolve this error? ").strip()
    
    return {
        "status": "resolved" if resolution else "error",
        "original_error": error,
        "resolution": resolution,
        "timestamp": datetime.now().isoformat()
    }

# Agent Functions
def supervisor_agent(state: TravelState) -> Dict[str, Any]:
    """Supervisor agent implementation."""
    messages = state.messages.copy()
    messages.append({"role": "system", "content": SUPERVISOR_PROMPT})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return {"messages": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}

def flight_agent(state: TravelState) -> Dict[str, Any]:
    """Flight agent implementation."""
    messages = state.messages.copy()
    messages.append({"role": "system", "content": FLIGHT_AGENT_PROMPT})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return {"messages": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}

def hotel_agent(state: TravelState) -> Dict[str, Any]:
    """Hotel agent implementation."""
    messages = state.messages.copy()
    messages.append({"role": "system", "content": HOTEL_AGENT_PROMPT})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return {"messages": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}

def itinerary_agent(state: TravelState) -> Dict[str, Any]:
    """Itinerary agent implementation."""
    messages = state.messages.copy()
    messages.append({"role": "system", "content": ITINERARY_AGENT_PROMPT})
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return {"messages": messages + [{"role": "assistant", "content": response.choices[0].message.content}]}

def create_travel_graph() -> StateGraph:
    """Create the travel assistant graph."""
    workflow = StateGraph(TravelState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("flight_agent", flight_agent)
    workflow.add_node("hotel_agent", hotel_agent)
    workflow.add_node("itinerary_agent", itinerary_agent)
    
    # Add parallel search node
    parallel_search = Parallel(
        "parallel_search",
        {
            "flight": flight_agent,
            "hotel": hotel_agent
        }
    )
    workflow.add_node("parallel_search", parallel_search)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "flight_agent")
    workflow.add_edge("supervisor", "hotel_agent")
    workflow.add_edge("supervisor", "parallel_search")
    workflow.add_edge("supervisor", "itinerary_agent")
    workflow.add_edge("supervisor", END)
    
    workflow.add_edge("flight_agent", "supervisor")
    workflow.add_edge("hotel_agent", "supervisor")
    workflow.add_edge("parallel_search", "supervisor")
    workflow.add_edge("itinerary_agent", "supervisor")
    workflow.add_edge("itinerary_agent", END)
    
    return workflow.compile()

def main():
    """Main entry point for the travel assistant."""
    # Initialize the graph
    graph = create_travel_graph()
    
    print("Travel Assistant initialized. Type 'quit' to exit.")
    print("Example: I want to book a trip from NYC to Miami from May 15-20, 2025")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
        
        # Initialize state
        state = TravelState(messages=[{"role": "user", "content": user_input}])
        
        # Run the graph
        try:
            for event, state in graph.run(state):
                if event.name == "agent":
                    print(f"\nAgent ({event.content}): {state.messages[-1]['content']}")
        except Exception as e:
            error_info = handle_error(str(e), "Error in graph execution")
            print(f"\nError: {error_info['message']}")
            if error_info.get('resolution'):
                print(f"Resolution: {error_info['resolution']}")

if __name__ == "__main__":
    main()
