"""Flight booking agent implementation."""
from typing import Dict, Any
from tools import search_flights

flight_agent = {
    "name": "flight_agent",
    "instructions": """You are a flight booking specialist agent. Your responsibility is to find flight options 
    based on the user's travel preferences. When you receive travel details, use the search_flights tool to find 
    suitable flight options. If you need information that's better handled by another agent, use the handoff tool.
    
    If you encounter any issues with finding flights, provide helpful information about alternatives.""",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search for flights based on origin, destination, and dates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Departure city or airport code"},
                        "destination": {"type": "string", "description": "Arrival city or airport code"},
                        "departure_date": {"type": "string", "description": "Date of departure (YYYY-MM-DD)"},
                        "return_date": {"type": "string", "description": "Date of return (YYYY-MM-DD)"},
                        "budget": {"type": "string", "description": "Optional budget constraint"}
                    },
                    "required": ["origin", "destination", "departure_date", "return_date"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "handoff",
                "description": "Hand off the conversation to another agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "enum": ["hotel_agent", "itinerary_agent"]}
                    },
                    "required": ["to"]
                }
            }
        }
    ]
}

def handle_search_flights(params: Dict[str, Any]) -> str:
    """Execute the flight search and format results."""
    flights = search_flights(
        params["origin"],
        params["destination"],
        params["departure_date"],
        params["return_date"],
        params.get("budget")
    )
    
    result = f"Found {len(flights)} flight options:\n\n"
    for i, flight in enumerate(flights, 1):
        result += f"Option {i}:\n"
        result += f"- Airline: {flight['airline']}\n"
        result += f"- Flight: {flight['flight_number']}\n"
        result += f"- Price: {flight['price']}\n"
        result += f"- Departure: {flight['departure']}\n"
        result += f"- Arrival: {flight['arrival']}\n\n"
    
    return result
