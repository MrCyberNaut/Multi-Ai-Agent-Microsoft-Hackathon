"""Consolidated multi-agent travel booking system."""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import streamlit as st
import sys
import time
import requests  # For Ollama API calls

# Load environment variables
load_dotenv()

# Import configuration
from config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    SERPAPI_TIMEOUT,
    HITL_ENABLED,
    HITL_MODE,
    HITL_TIMEOUT,
    DEBUG_MODE,
)

# Configuration
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Kept for backward compatibility
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CACHE_DIR = "cache"
DEBUG = True  # Enable debug output

def debug_print(message):
    """Print debug messages if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

# Initialize API clients
debug_print(f"Initializing with LLM provider: {LLM_PROVIDER}")
if LLM_PROVIDER.lower() == "openai":
    debug_print("Using OpenAI client")
    client = OpenAI(api_key=OPENAI_API_KEY)
elif LLM_PROVIDER.lower() == "ollama":
    debug_print(f"Using Ollama client with model: {OLLAMA_MODEL}")
    # No client initialization needed for Ollama as we'll use the call_ollama function
    class OllamaWrapper:
        def __init__(self):
            self.chat = type('obj', (object,), {
                'completions': type('obj', (object,), {
                    'create': lambda **kwargs: call_ollama(
                        kwargs.get('messages', []), 
                        kwargs.get('temperature', TEMPERATURE), 
                        kwargs.get('max_tokens', MAX_TOKENS)
                    )
                })
            })
    client = OllamaWrapper()
else:
    debug_print(f"Unknown LLM provider: {LLM_PROVIDER}, defaulting to OpenAI")
    client = OpenAI(api_key=OPENAI_API_KEY)

# Function to call Ollama API
def call_ollama(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """Call Ollama API with messages."""
    debug_print(f"Calling Ollama API with {len(messages)} messages")
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        debug_print(f"Sending request to Ollama at {OLLAMA_URL}/api/chat")
        debug_print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make the API request
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        debug_print(f"Ollama API status code: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            debug_print(f"Ollama response: {json.dumps(response_json, indent=2)}")
            
            # Extract the assistant's message from the response
            if "message" in response_json and "content" in response_json["message"]:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": response_json["message"]["content"]
                            }
                        }
                    ]
                }
            else:
                debug_print("Unexpected response structure from Ollama API")
                debug_print(f"Full response: {response_json}")
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Error: Unexpected response structure from Ollama"
                            }
                        }
                    ]
                }
        else:
            debug_print(f"Error from Ollama API: {response.text}")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Error calling Ollama API: {response.status_code} - {response.text}"
                        }
                    }
                ]
            }
    except Exception as e:
        debug_print(f"Exception in call_ollama: {str(e)}")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    }
                }
            ]
        }

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
4. Managing user interactions
5. Ensuring all travel requirements are met

Use the available tools to search for travel options and coordinate with other agents.
""" + ("""Always validate critical decisions with human operators since Human-in-the-Loop is enabled.""" if HITL_ENABLED else """Process requests automatically without human intervention since Human-in-the-Loop is disabled.""") + """
Maintain conversation context and handle transitions between agents smoothly."""

FLIGHT_AGENT_PROMPT = """You are a flight booking specialist agent. Your key responsibilities include:

1. Searching for optimal flight options using SerpAPI
2. Analyzing and filtering results based on user preferences
3. Presenting options clearly for user selection
4. Handling booking-related queries and issues
5. Coordinating with other agents for complete travel planning

Always verify flight availability and pricing before presenting options.
""" + ("""Get human approval for final flight selections.""" if HITL_ENABLED else """Process flight selections automatically without human intervention.""") + """
Handle errors gracefully and suggest alternatives when needed."""

HOTEL_AGENT_PROMPT = """You are a hotel booking specialist agent. Your key responsibilities include:

1. Finding suitable accommodations using SerpAPI
2. Matching hotels to user preferences and budget
3. Verifying availability and amenities
4. Presenting options clearly for user selection
5. Coordinating with other agents for complete travel planning

Consider location, ratings, and reviews when selecting options.
""" + ("""Get human approval for final hotel selections.""" if HITL_ENABLED else """Process hotel selections automatically without human intervention.""") + """
Handle errors gracefully and suggest alternatives when needed."""

ITINERARY_AGENT_PROMPT = """You are an itinerary planning specialist. Your key responsibilities include:

1. Creating comprehensive travel schedules
2. Incorporating flight and hotel bookings
3. Suggesting activities and attractions
4. Optimizing timing and logistics
5. Providing a complete travel plan

Use SerpAPI to find local attractions and activities.
Consider travel times, check-in/out times, and local conditions.
""" + ("""Allow for flexibility and human customization of plans.""" if HITL_ENABLED else """Generate complete itineraries automatically.""") + """
"""

# SerpAPI Utility Functions
def _get_cache_path(query_hash: str) -> str:
    """Get the cache file path for a query."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{query_hash}.json")

def _cache_results(query_hash: str, results: Dict[str, Any]) -> None:
    """Cache search results."""
    cache_path = _get_cache_path(query_hash)
    debug_print(f"Caching results to: {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().timestamp(),
            'results': results
        }, f)

def _get_cached_results(query_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached results if they exist and are not expired."""
    cache_path = _get_cache_path(query_hash)
    if not os.path.exists(cache_path):
        debug_print(f"No cache found for: {query_hash}")
        return None
    
    debug_print(f"Found cache file: {cache_path}")
    with open(cache_path, 'r') as f:
        cached = json.load(f)
    
    if datetime.now().timestamp() - cached['timestamp'] > SERPAPI_CACHE_DURATION:
        debug_print(f"Cache expired for: {query_hash}")
        return None
    
    debug_print(f"Using cached results for: {query_hash}")
    return cached['results']

def search_flights_serp(origin: str, destination: str, departure_date: str, return_date: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for flights using SerpAPI."""
    debug_print(f"search_flights_serp called with: origin={origin}, destination={destination}, departure_date={departure_date}, return_date={return_date}")
    
    if not SERPAPI_API_KEY:
        error_msg = "SERPAPI_API_KEY is not set in environment variables"
        debug_print(error_msg)
        raise ValueError(error_msg)
    
    query_hash = f"flights_{origin}_{destination}_{departure_date}"
    debug_print(f"Query hash: {query_hash}")
    
    cached = _get_cached_results(query_hash)
    if cached:
        debug_print(f"Returning {len(cached.get('results', []))} cached flight results")
        return cached["results"]
    
    # Updated parameters based on SerpAPI documentation
    params = {
        "engine": "google_flights",
        "departure_id": origin.upper(),  # Ensure uppercase IATA code
        "arrival_id": destination.upper(),  # Ensure uppercase IATA code
        "outbound_date": departure_date,
        "return_date": return_date,
        "currency": "INR",  # Default to INR
        "hl": "en",
        "gl": "in",
        "type": "1",  # 1 = Round trip, 2 = One way
        "api_key": SERPAPI_API_KEY
    }
    
    # Default budget is 5000 INR per person if not specified
    if budget:
        budget_value = budget if isinstance(budget, int) else int(''.join(filter(str.isdigit, str(budget))) or 5000)
    else:
        budget_value = 5000
    
    debug_print(f"Searching flights with params: {params}")
    
    try:
        # Use GoogleSearch instead of SerpApiClient
        debug_print("Initializing GoogleSearch for flights")
        search = GoogleSearch(params)
        debug_print("Executing get_dict() for flight search")
        response = search.get_dict()
        debug_print(f"Flight search response keys: {list(response.keys())}")
        
        # Save full response for debugging
        debug_query_hash = f"debug_{query_hash}"
        _cache_results(debug_query_hash, response)
        
        # Extract and process flight data
        flight_results = []
        
        # Check if we got flights_results and process them
        if "flights_results" in response:
            flights_data = response["flights_results"]
            debug_print(f"Flights results keys: {list(flights_data.keys())}")
            
            # Process different response structures from SerpAPI
            if "flights" in flights_data:
                flights = flights_data["flights"]
                debug_print(f"Found {len(flights)} flights in response")
                
                for flight in flights:
                    # Extract price from different possible locations
                    price = 0
                    if "price" in flight and "total" in flight["price"] and "amount" in flight["price"]["total"]:
                        price = flight["price"]["total"]["amount"]
                    
                    # Check if flight is within budget
                    if price and price <= budget_value:
                        debug_print(f"Adding flight: {flight.get('airlines', [{}])[0].get('name', 'Unknown')}, price: {price}")
                        
                        # Process airline information
                        airline = "Unknown Airline"
                        if "airlines" in flight and flight["airlines"] and len(flight["airlines"]) > 0:
                            airline = flight["airlines"][0].get("name", "Unknown Airline")
                        
                        # Process times
                        departure_time = ""
                        arrival_time = ""
                        duration = ""
                        
                        if "departure" in flight and "datetime" in flight["departure"]:
                            departure_time = flight["departure"]["datetime"].get("timestamp", "")
                        
                        if "arrival" in flight and "datetime" in flight["arrival"]:
                            arrival_time = flight["arrival"]["datetime"].get("timestamp", "")
                        
                        if "duration" in flight:
                            duration = flight["duration"].get("text", "")
                        
                        flight_results.append({
                            "airline": airline,
                            "departure_time": departure_time,
                            "arrival_time": arrival_time,
                            "duration": duration,
                            "price": price,
                            "currency": "INR",
                            "link": flight.get("booking_token", "")
                        })
            
            # Also check for best_flights if no regular flights
            elif "best_flights" in response:
                best_flights = response["best_flights"]
                debug_print(f"Found {len(best_flights)} best flights in response")
                
                for flight in best_flights:
                    # Process these flights differently
                    price = flight.get("price", 0)
                    
                    if price and price <= budget_value:
                        # Get flight details
                        flights_info = flight.get("flights", [{}])
                        first_flight = flights_info[0] if flights_info else {}
                        last_flight = flights_info[-1] if flights_info else {}
                        
                        airline = first_flight.get("airline", "Unknown Airline")
                        departure_time = first_flight.get("departure_airport", {}).get("time", "")
                        arrival_time = last_flight.get("arrival_airport", {}).get("time", "")
                        duration = flight.get("total_duration", "")
                        
                        flight_results.append({
                            "airline": airline,
                            "departure_time": departure_time,
                            "arrival_time": arrival_time,
                            "duration": duration,
                            "price": price,
                            "currency": "INR",
                            "link": flight.get("booking_token", "")
                        })
        
        # If no flights found through regular paths, create sample data for development
        if not flight_results and DEBUG:
            debug_print("No flights found in response, creating sample data for development")
            flight_results = [
                {
                    "airline": "Sample Airline",
                    "departure_time": departure_date + " 08:00",
                    "arrival_time": departure_date + " 10:00",
                    "duration": "2h 0m",
                    "price": 4500,
                    "currency": "INR",
                    "link": "#sample-link"
                },
                {
                    "airline": "Test Airways",
                    "departure_time": departure_date + " 12:30",
                    "arrival_time": departure_date + " 14:45",
                    "duration": "2h 15m",
                    "price": 4800,
                    "currency": "INR",
                    "link": "#sample-link"
                }
            ]
            debug_print("Created sample flight data for development purposes")
        
        # Cache the processed results
        results = {"timestamp": time.time(), "results": flight_results}
        _cache_results(query_hash, results)
        
        debug_print(f"Flight search results count: {len(flight_results)}")
        return flight_results
    
    except Exception as e:
        error_msg = f"Error searching flights: {str(e)}"
        debug_print(f"Exception in search_flights_serp: {error_msg}")
        # Return empty list on error
        return []

def search_hotels_serp(location: str, check_in: str, check_out: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for hotels using SerpAPI."""
    debug_print(f"search_hotels_serp called with: location={location}, check_in={check_in}, check_out={check_out}")
    
    if not SERPAPI_API_KEY:
        error_msg = "SERPAPI_API_KEY is not set in environment variables"
        debug_print(error_msg)
        raise ValueError(error_msg)
    
    query_hash = f"hotels_{location}_{check_in}_{check_out}"
    debug_print(f"Query hash: {query_hash}")
    
    cached = _get_cached_results(query_hash)
    if cached:
        debug_print(f"Returning {len(cached.get('results', []))} cached hotel results")
        return cached["results"]
    
    # Updated parameters based on SerpAPI documentation
    params = {
        "engine": "google_hotels",
        "q": f"hotels in {location}",
        "check_in_date": check_in,
        "check_out_date": check_out,
        "currency": "INR",  # Default to INR
        "adults": 2,  # Default number of adults
        "children": 0,
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_API_KEY
    }
    
    # Default budget is 5000 INR per night if not specified
    if budget:
        budget_value = budget if isinstance(budget, int) else int(''.join(filter(str.isdigit, str(budget))) or 5000)
    else:
        budget_value = 5000
    
    debug_print(f"Searching hotels with params: {params}")
    
    try:
        # Use GoogleSearch instead of SerpApiClient
        debug_print("Initializing GoogleSearch for hotels")
        search = GoogleSearch(params)
        debug_print("Executing get_dict() for hotel search")
        response = search.get_dict()
        debug_print(f"Hotel search response keys: {list(response.keys())}")
        
        # Save full response for debugging
        debug_query_hash = f"debug_{query_hash}"
        _cache_results(debug_query_hash, response)
        
        # Extract and process hotel data
        hotel_results = []
        
        if "properties" in response:
            hotels = response["properties"]
            debug_print(f"Found {len(hotels)} properties in response")
            
            for hotel in hotels:
                # Check if hotel is a vacation rental
                if hotel.get("type") != "hotel":
                    continue
                
                # Get the nightly rate or total rate
                price = 0
                if "rate_per_night" in hotel and "extracted_lowest" in hotel["rate_per_night"]:
                    price = hotel["rate_per_night"]["extracted_lowest"]
                elif "total_rate" in hotel and "extracted_lowest" in hotel["total_rate"]:
                    price = hotel["total_rate"]["extracted_lowest"]
                
                # Skip hotels outside budget
                if price and price <= budget_value:
                    debug_print(f"Adding hotel: {hotel.get('name', 'Unknown')}, price: {price}")
                    hotel_results.append({
                        "name": hotel.get("name", "Unknown Hotel"),
                        "description": hotel.get("description", ""),
                        "address": "",  # Address not directly available in response
                        "price_per_night": price,
                        "currency": "INR",
                        "rating": hotel.get("overall_rating", 0),
                        "amenities": hotel.get("amenities", []),
                        "image_url": hotel.get("images", [{}])[0].get("original_image", "") if hotel.get("images") else "",
                    })
        else:
            debug_print("No properties found in response")
            if "error" in response:
                debug_print(f"API Error: {response['error']}")
        
        # Cache the processed results
        results = {"timestamp": time.time(), "results": hotel_results}
        _cache_results(query_hash, results)
        
        debug_print(f"Hotel search results count: {len(hotel_results)}")
        return hotel_results
    
    except Exception as e:
        error_msg = f"Error searching hotels: {str(e)}"
        debug_print(f"Exception in search_hotels_serp: {error_msg}")
        # Return empty list on error
        return []

def get_destination_info_serp(destination: str) -> Dict[str, Any]:
    """Get destination information using SerpAPI."""
    debug_print(f"get_destination_info_serp called with: destination={destination}")
    
    if not SERPAPI_API_KEY:
        error_msg = "SERPAPI_API_KEY is not set in environment variables"
        debug_print(error_msg)
        raise ValueError(error_msg)
    
    query_hash = f"destination_{destination}"
    debug_print(f"Query hash: {query_hash}")
    
    cached = _get_cached_results(query_hash)
    if cached:
        debug_print(f"Returning cached destination info for: {destination}")
        return cached
    
    # Updated parameters for better destination information
    params = {
        "engine": "google",
        "q": f"travel guide {destination} tourist attractions things to do",
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us",
        "location": "United States",
        "google_domain": "google.com",
        "num": 10
    }
    
    try:
        message = "Looking up destination information..."
        if 'st' in globals():
            st.write(message)
        debug_print(message)
        
        debug_print("Initializing GoogleSearch for destination info")
        search = GoogleSearch(params)
        debug_print("Executing get_dict() for destination search")
        results = search.get_dict()
        debug_print(f"Destination search response keys: {list(results.keys())}")
        
        # Check for API error response
        if "error" in results:
            debug_print(f"SerpAPI error: {results['error']}")
            _cache_results(query_hash, {})
            return {}
        
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
            debug_print("Found knowledge graph in results")
            info.update({
                "description": kg.get('description', ''),
                "weather": kg.get('weather', ''),
                "timezone": kg.get('timezone', ''),
                "currency": kg.get('currency', ''),
                "languages": kg.get('languages', [])
            })
        else:
            debug_print("No knowledge graph found in results")
        
        if 'organic_results' in results:
            debug_print(f"Processing {len(results['organic_results'])} organic results")
            attractions = []
            tips = []
            for result in results['organic_results'][:5]:
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '')
                
                # Extract attractions
                if any(word in title for word in ['attractions', 'things to do', 'places to visit']):
                    # Try to extract bulleted list items
                    if 'list' in result:
                        attractions.extend([item.get('title', '') for item in result.get('list', [])])
                    elif snippet:
                        # Extract potential attractions from snippet
                        lines = snippet.split('.')
                        attractions.extend([line.strip() for line in lines if len(line.strip()) > 15 and len(line.strip()) < 100])
                
                # Extract tips
                if any(word in title for word in ['tips', 'guide', 'advice', 'travel']):
                    tips.append(snippet)
            
            # Fallback to getting attractions from knowledge graph
            if not attractions and 'attractions' in results.get('knowledge_graph', {}):
                kg = results['knowledge_graph']
                attractions = kg.get('attractions', [])
            
            # Clean up attractions and tips
            attractions = [a for a in attractions if a and len(a.strip()) > 3]
            tips = [t for t in tips if t and len(t.strip()) > 20]
            
            info['attractions'] = attractions[:5]
            info['local_tips'] = tips[:3]
            
            debug_print(f"Extracted {len(info['attractions'])} attractions and {len(info['local_tips'])} tips")
        else:
            debug_print("No organic results found")
        
        # Save debug information if we didn't find useful content
        if not info['description'] and not info['attractions']:
            debug_print("No useful content found, saving debug info")
            with open(f"cache/debug_{query_hash}.json", "w") as f:
                json.dump(results, f)
        
        _cache_results(query_hash, info)
        return info
        
    except Exception as e:
        error_msg = f"Error in SerpAPI destination search: {str(e)}"
        debug_print(f"Exception in get_destination_info_serp: {error_msg}")
        _cache_results(query_hash, {})
        return {}

# Human-in-the-Loop Functions
def get_human_approval(data: Dict[str, Any], context: str) -> bool:
    """Get human approval for a decision."""
    debug_print(f"get_human_approval called with context: {context}")
    
    if not HITL_ENABLED:
        debug_print("HITL disabled, automatically approving")
        return True
    
    # Check if running in Streamlit mode
    in_streamlit = 'streamlit' in sys.modules and HITL_MODE == "streamlit"
    
    if in_streamlit:
        import streamlit as st
        # Store the approval request in session state
        if 'hitl_approval_request' not in st.session_state:
            st.session_state.hitl_approval_request = {
                'data': data,
                'context': context,
                'timestamp': datetime.now().timestamp(),
                'type': 'approval',
                'pending': True,
                'result': None
            }
            debug_print("Stored HITL approval request in session state")
            # Force a rerun to display the approval UI
            st.rerun()
        else:
            # Check if there's a result from a previous approval
            if not st.session_state.hitl_approval_request['pending']:
                result = st.session_state.hitl_approval_request['result']
                # Clear the request
                st.session_state.hitl_approval_request = None
                debug_print(f"Retrieved HITL approval result: {result}")
                return result
            
            # If we're still waiting for approval but timeout has passed
            current_time = datetime.now().timestamp()
            if current_time - st.session_state.hitl_approval_request['timestamp'] > HITL_TIMEOUT:
                debug_print("HITL approval timed out, automatically approving")
                st.session_state.hitl_approval_request = None
                return True
            
            # Otherwise, we're still waiting
            debug_print("Still waiting for HITL approval")
            time.sleep(1)  # Small delay to avoid tight loop
            st.rerun()  # Force a rerun to check again
    else:
        # CLI mode
        print("\n=== Human Approval Required ===")
        print(f"Context: {context}")
        print("Data to approve:")
        print(json.dumps(data, indent=2))
        
        response = input("\nDo you approve? (yes/no): ").lower().strip()
        debug_print(f"Human approval response: {response}")
        return response in ['y', 'yes']

def get_human_selection(options: List[Dict[str, Any]], context: str) -> Optional[Dict[str, Any]]:
    """Get human selection from a list of options."""
    debug_print(f"get_human_selection called with {len(options)} options, context: {context}")
    
    if not HITL_ENABLED:
        debug_print("HITL disabled, automatically selecting first option")
        return options[0] if options else None
    
    # Check if running in Streamlit mode
    in_streamlit = 'streamlit' in sys.modules and HITL_MODE == "streamlit"
    
    if in_streamlit:
        # Store the selection request in session state
        if 'hitl_selection_request' not in st.session_state:
            st.session_state.hitl_selection_request = {
                'options': options,
                'context': context,
                'timestamp': datetime.now().timestamp(),
                'type': 'selection',
                'pending': True,
                'result': None
            }
            debug_print("Stored HITL selection request in session state")
            # Force a rerun to display the selection UI
            st.rerun()
        else:
            # Check if there's a result from a previous selection
            if not st.session_state.hitl_selection_request['pending']:
                result = st.session_state.hitl_selection_request['result']
                # Clear the request
                st.session_state.hitl_selection_request = None
                debug_print(f"Retrieved HITL selection result: {options.index(result) if result in options else 'none'}")
                return result
            
            # If we're still waiting for selection but timeout has passed
            current_time = datetime.now().timestamp()
            if current_time - st.session_state.hitl_selection_request['timestamp'] > HITL_TIMEOUT:
                debug_print("HITL selection timed out, automatically selecting first option")
                st.session_state.hitl_selection_request = None
                return options[0] if options else None
            
            # Otherwise, we're still waiting
            debug_print("Still waiting for HITL selection")
            time.sleep(1)  # Small delay to avoid tight loop
            st.rerun()  # Force a rerun to check again
    else:
        # CLI mode
        print("\n=== Human Selection Required ===")
        print(f"Context: {context}")
        print("\nAvailable options:")
        
        for i, option in enumerate(options, 1):
            print(f"\n{i}. {json.dumps(option, indent=2)}")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (number): "))
                if 1 <= choice <= len(options):
                    debug_print(f"Human selected option {choice}")
                    return options[choice - 1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

def handle_error(error: Union[str, Dict[str, Any]], context: str) -> Dict[str, Any]:
    """Handle errors with human intervention."""
    debug_print(f"handle_error called with context: {context}, error: {error}")
    
    if not HITL_ENABLED:
        debug_print("HITL disabled, returning error status")
        return {"status": "error", "message": str(error)}
    
    # Check if running in Streamlit mode
    in_streamlit = 'streamlit' in sys.modules and HITL_MODE == "streamlit"
    
    if in_streamlit:
        import streamlit as st
        # Store the error resolution request in session state
        if 'hitl_error_request' not in st.session_state:
            st.session_state.hitl_error_request = {
                'error': error,
                'context': context,
                'timestamp': datetime.now().timestamp(),
                'type': 'error',
                'pending': True,
                'resolution': None
            }
            debug_print("Stored HITL error resolution request in session state")
            # Force a rerun to display the error resolution UI
            st.rerun()
        else:
            # Check if there's a resolution from a previous error
            if not st.session_state.hitl_error_request['pending']:
                resolution = st.session_state.hitl_error_request['resolution']
                # Clear the request
                st.session_state.hitl_error_request = None
                debug_print(f"Retrieved HITL error resolution: {resolution}")
                return {
                    "status": "resolved" if resolution else "error",
                    "original_error": error,
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat()
                }
            
            # If we're still waiting for resolution but timeout has passed
            current_time = datetime.now().timestamp()
            if current_time - st.session_state.hitl_error_request['timestamp'] > HITL_TIMEOUT:
                debug_print("HITL error resolution timed out, returning error status")
                st.session_state.hitl_error_request = None
                return {"status": "error", "message": str(error)}
            
            # Otherwise, we're still waiting
            debug_print("Still waiting for HITL error resolution")
            time.sleep(1)  # Small delay to avoid tight loop
            st.rerun()  # Force a rerun to check again
    else:
        # CLI mode
        print("\n=== Error Resolution Required ===")
        print(f"Context: {context}")
        print(f"Error: {error}")
        
        resolution = input("How would you like to resolve this error? ").strip()
        debug_print(f"Human resolution: {resolution}")
        
        return {
            "status": "resolved" if resolution else "error",
            "original_error": error,
            "resolution": resolution,
            "timestamp": datetime.now().isoformat()
        }

# Agent Functions
def supervisor_agent(state: TravelState) -> Dict[str, Any]:
    """Supervisor agent implementation."""
    debug_print("supervisor_agent called")
    
    messages = state.messages.copy()
    messages.append({"role": "system", "content": SUPERVISOR_PROMPT})
    
    debug_print(f"Calling supervisor LLM with {len(messages)} messages")
    try:
        if LLM_PROVIDER.lower() == "ollama":
            response = call_ollama(messages, TEMPERATURE, MAX_TOKENS)
            assistant_message = response["choices"][0]["message"]["content"]
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            assistant_message = response.choices[0].message.content
            
        debug_print(f"Received response from LLM: {assistant_message[:50]}...")
        
        # Update messages with the assistant's response
        updated_messages = messages + [{"role": "assistant", "content": assistant_message}]
        debug_print("Supervisor agent completed")
        return {"messages": updated_messages}
    except Exception as e:
        error_msg = f"Error calling LLM API: {str(e)}"
        debug_print(error_msg)
        return {"messages": messages, "error": error_msg}

def flight_agent(state: TravelState) -> Dict[str, Any]:
    """Flight agent implementation."""
    debug_print("flight_agent called")
    
    messages = state.messages.copy()
    messages.append({"role": "system", "content": FLIGHT_AGENT_PROMPT})
    
    debug_print(f"Calling flight agent LLM with {len(messages)} messages")
    try:
        if LLM_PROVIDER.lower() == "ollama":
            response = call_ollama(messages, TEMPERATURE, MAX_TOKENS)
            assistant_message = response["choices"][0]["message"]["content"]
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            assistant_message = response.choices[0].message.content
            
        debug_print(f"Received response from LLM: {assistant_message[:50]}...")
        
        # Update messages with the assistant's response
        updated_messages = messages + [{"role": "assistant", "content": assistant_message}]
        debug_print("Flight agent completed")
        return {"messages": updated_messages}
    except Exception as e:
        error_msg = f"Error calling LLM API: {str(e)}"
        debug_print(error_msg)
        return {"messages": messages, "error": error_msg}

def hotel_agent(state: TravelState) -> Dict[str, Any]:
    """Hotel agent implementation."""
    debug_print("hotel_agent called")
    
    messages = state.messages.copy()
    messages.append({"role": "system", "content": HOTEL_AGENT_PROMPT})
    
    debug_print(f"Calling hotel agent LLM with {len(messages)} messages")
    try:
        if LLM_PROVIDER.lower() == "ollama":
            response = call_ollama(messages, TEMPERATURE, MAX_TOKENS)
            assistant_message = response["choices"][0]["message"]["content"]
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            assistant_message = response.choices[0].message.content
            
        debug_print(f"Received response from LLM: {assistant_message[:50]}...")
        
        # Update messages with the assistant's response
        updated_messages = messages + [{"role": "assistant", "content": assistant_message}]
        debug_print("Hotel agent completed")
        return {"messages": updated_messages}
    except Exception as e:
        error_msg = f"Error calling LLM API: {str(e)}"
        debug_print(error_msg)
        return {"messages": messages, "error": error_msg}

def itinerary_agent(state: TravelState) -> Dict[str, Any]:
    """Itinerary agent implementation."""
    debug_print("itinerary_agent called")
    
    messages = state.messages.copy()
    
    # Enhanced prompt to include instructions about ending the conversation
    enhanced_prompt = ITINERARY_AGENT_PROMPT + """

When you've completed creating a full itinerary or answered the user's question completely, 
end your message with 'This completes your final itinerary. Is there anything else you'd like to know?'
This will help ensure the conversation reaches a natural conclusion.
"""
    
    messages.append({"role": "system", "content": enhanced_prompt})
    
    debug_print(f"Calling itinerary agent LLM with {len(messages)} messages")
    try:
        if LLM_PROVIDER.lower() == "ollama":
            response = call_ollama(messages, TEMPERATURE, MAX_TOKENS)
            assistant_message = response["choices"][0]["message"]["content"]
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            assistant_message = response.choices[0].message.content
            
        debug_print(f"Received response from LLM: {assistant_message[:50]}...")
        
        # Update messages with the assistant's response
        updated_messages = messages + [{"role": "assistant", "content": assistant_message}]
        debug_print("Itinerary agent completed")
        return {"messages": updated_messages}
    except Exception as e:
        error_msg = f"Error calling LLM API: {str(e)}"
        debug_print(error_msg)
        return {"messages": messages, "error": error_msg}

def router(state: TravelState) -> str:
    """Route messages based on content."""
    if not state.messages or len(state.messages) == 0:
        debug_print("No messages found, routing to supervisor")
        return "supervisor"
        
    last_message = state.messages[-1]["content"].lower() if state.messages[-1].get("content") else ""
    debug_print(f"Routing based on message: {last_message[:50]}...")
    
    # More comprehensive check for terminal conditions
    terminal_phrases = [
        "final itinerary", 
        "thank you", 
        "goodbye", 
        "thank you for your help",
        "anything else you'd like to know",
        "is there anything else you'd like to know",
        "is there anything else i can help you with",
        "this completes your",
        "completed your request",
        "all set",
        "all done"
    ]
    
    # Check if any terminal phrase is in the message
    if any(phrase in last_message for phrase in terminal_phrases):
        debug_print(f"Terminal condition detected: '{next((phrase for phrase in terminal_phrases if phrase in last_message), '')}'")
        return END
    
    # Add recursion counter to prevent infinite loops
    if len(state.messages) > 15:  # Reduced threshold to prevent long conversations
        debug_print(f"Message count threshold exceeded ({len(state.messages)} > 15), ending graph to prevent recursion")
        return END
    
    # Check for search completion patterns
    if ("found" in last_message and any(word in last_message for word in ["flight", "hotel"])) or \
       "search results" in last_message or \
       "search completed" in last_message:
        debug_print("Search completion detected, ending graph")
        return END
    
    # More specific routing based on message content
    if any(flight_term in last_message for flight_term in ["flight", "book a trip", "travel from", "fly to", "booking", "airline"]):
        debug_print("Routing to flight_agent")
        return "flight_agent"
    elif any(hotel_term in last_message for hotel_term in ["hotel", "accommodation", "place to stay", "lodging", "room"]):
        debug_print("Routing to hotel_agent")
        return "hotel_agent"
    elif any(itinerary_term in last_message for itinerary_term in ["itinerary", "plan", "schedule", "activities", "attractions"]):
        debug_print("Routing to itinerary_agent")
        return "itinerary_agent"
    else:
        debug_print("No specific route found, defaulting to supervisor")
        return "supervisor"

def create_travel_graph() -> StateGraph:
    """Create the travel assistant graph."""
    debug_print("Creating travel assistant graph")
    workflow = StateGraph(TravelState)
    
    # Set recursion limit directly on the StateGraph instance
    workflow.recursion_limit = 200  # Increased from 50 to 200
    
    # Add nodes
    debug_print("Adding graph nodes")
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("flight_agent", flight_agent)
    workflow.add_node("hotel_agent", hotel_agent)
    workflow.add_node("itinerary_agent", itinerary_agent)

    # Define router function is outside this function (see above)
    
    # Edges
    debug_print("Adding graph edges")
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor to other agents and END
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "itinerary_agent": "itinerary_agent",
            "supervisor": "supervisor",
            END: END  # Add explicit END option
        }
    )
    
    # Add edges from agents back to supervisor
    workflow.add_edge("flight_agent", "supervisor")
    workflow.add_edge("hotel_agent", "supervisor")
    workflow.add_edge("itinerary_agent", "supervisor")
    
    debug_print("Graph created and compiled")
    return workflow.compile()


def main():
    """Main entry point for the travel assistant."""
    debug_print("main() function started")
    
    # Initialize the graph
    debug_print("Initializing travel graph")
    graph = create_travel_graph()
    
    # Check if running in Streamlit
    in_streamlit = 'streamlit' in sys.modules
    debug_print(f"Running in Streamlit: {in_streamlit}")
    
    if in_streamlit:
        st.write("Travel Assistant API initialized and ready to use")
        debug_print("Returning graph for Streamlit")
        return graph
    
    # Test invocation of the graph with a sample query
    print("Invoking graph with test query...")
    test_state = TravelState(
        messages=[{
            "role": "user",
            "content": "I want to book a trip from Mumbai to Mangalore from May 3-6, 2025"
        }]
    )
    
    try:
        # Run the graph with the test state - use the updated method
        debug_print("Running graph with test query")
        for event in graph.stream(test_state):
            if event.get("type") == "agent":
                print(f"\nAgent ({event.get('agent')}): {event.get('state').messages[-1]['content']}")
                debug_print(f"Agent event: {event.get('agent')}")
        print("\nGraph execution completed successfully")
        debug_print("Test graph execution completed")
    except Exception as e:
        debug_print(f"Error in test graph execution: {str(e)}")
        error_info = handle_error(str(e), "Error in graph execution")
        print(f"\nError: {error_info['message']}")
        if error_info.get('resolution'):
            print(f"Resolution: {error_info['resolution']}")
    
    # CLI mode
    print("\nTravel Assistant initialized. Type 'quit' to exit.")
    print("Example: I want to book a trip from NYC to Miami from May 15-20, 2025")
    debug_print("Entering CLI mode")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            debug_print("User requested to quit")
            break
        
        # Initialize state
        debug_print(f"Creating new state with user input: {user_input}")
        state = TravelState(messages=[{"role": "user", "content": user_input}])
        
        # Run the graph
        try:
            debug_print("Running graph with user input")
            for event in graph.stream(state):
                if event.get("type") == "agent":
                    print(f"\nAgent ({event.get('agent')}): {event.get('state').messages[-1]['content']}")
                    debug_print(f"Agent event: {event.get('agent')}")
            debug_print("Graph execution completed")
        except Exception as e:
            debug_print(f"Error in graph execution: {str(e)}")
            error_info = handle_error(str(e), "Error in graph execution")
            print(f"\nError: {error_info['message']}")
            if error_info.get('resolution'):
                print(f"Resolution: {error_info['resolution']}")
    
    debug_print("main() function completed")
    return graph

if __name__ == "__main__":
    debug_print("Script executed directly")
    main()
