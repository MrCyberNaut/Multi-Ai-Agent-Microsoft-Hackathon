"""Hotel booking agent implementation."""
from typing import Dict, Any
from tools import search_hotels

hotel_agent = {
    "name": "hotel_agent",
    "instructions": """You are a hotel booking specialist agent. Your responsibility is to find hotel options 
    based on the user's travel preferences. When you receive travel details, use the search_hotels tool to find 
    suitable hotel accommodations. If you need information that's better handled by another agent, use the handoff tool.
    
    If you encounter any issues with finding hotels, provide helpful information about alternatives.""",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_hotels",
                "description": "Search for hotels based on location and dates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City or area for hotel search"},
                        "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                        "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                        "budget": {"type": "string", "description": "Optional budget constraint"}
                    },
                    "required": ["location", "check_in", "check_out"]
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
                        "to": {"type": "string", "enum": ["flight_agent", "itinerary_agent"]}
                    },
                    "required": ["to"]
                }
            }
        }
    ]
}

def handle_search_hotels(params: Dict[str, Any]) -> str:
    """Execute the hotel search and format results."""
    hotels = search_hotels(
        params["location"],
        params["check_in"],
        params["check_out"],
        params.get("budget")
    )
    
    result = f"Found {len(hotels)} hotel options:\n\n"
    for i, hotel in enumerate(hotels, 1):
        result += f"Option {i}:\n"
        result += f"- Name: {hotel['name']}\n"
        result += f"- Price: {hotel['price']}\n"
        result += f"- Rating: {hotel['rating']} stars\n"
        result += f"- Amenities: {', '.join(hotel['amenities'])}\n\n"
    
    return result
