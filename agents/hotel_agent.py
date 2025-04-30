"""Hotel booking agent implementation."""
from typing import Dict, Any
from tools import search_hotels
from hitl import get_human_selection, handle_error

hotel_agent = {
    "name": "hotel_agent",
    "instructions": """You are a hotel booking specialist agent. Your responsibility is to find hotel options 
    based on the user's travel preferences. When you receive travel details, use the search_hotels tool to find 
    suitable hotel accommodations. If you need information that's better handled by another agent, use the handoff tool.
    
    Always present hotel options to the human for selection when available. If you encounter any issues with 
    finding hotels, work with the human to find alternatives or resolve errors.""",
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
    """Execute the hotel search and handle results with human interaction."""
    try:
        hotels = search_hotels(
            params["location"],
            params["check_in"],
            params["check_out"],
            params.get("budget")
        )
        
        if not hotels:
            error_context = f"No hotels found in {params['location']}"
            error_resolution = handle_error(
                "No hotels found matching your criteria",
                error_context
            )
            return f"Error: {error_resolution['message']}\nResolution: {error_resolution.get('resolution', 'No resolution provided')}"
        
        # Let human select from available hotels
        selected_hotel = get_human_selection(
            hotels,
            f"Please select a hotel in {params['location']}"
        )
        
        if not selected_hotel:
            return "No hotel was selected."
        
        result = "Selected hotel details:\n"
        result += f"- Name: {selected_hotel['name']}\n"
        result += f"- Price: {selected_hotel['price']}\n"
        result += f"- Rating: {selected_hotel['rating']} stars\n"
        if 'address' in selected_hotel:
            result += f"- Address: {selected_hotel['address']}\n"
        if 'website' in selected_hotel:
            result += f"- Website: {selected_hotel['website']}\n"
        if 'amenities' in selected_hotel:
            result += f"- Amenities: {', '.join(selected_hotel['amenities'])}\n"
        if 'reviews' in selected_hotel and selected_hotel['reviews']:
            result += "\nTop Review:\n"
            result += f"{selected_hotel['reviews'][0]}\n"
        
        return result
    
    except Exception as e:
        error_resolution = handle_error(
            str(e),
            "Error occurred while searching for hotels"
        )
        return f"Error: {error_resolution['message']}\nResolution: {error_resolution.get('resolution', 'No resolution provided')}"
