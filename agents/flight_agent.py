"""Flight booking agent implementation."""
from typing import Dict, Any
from tools import search_flights
from hitl import get_human_selection, handle_error

flight_agent = {
    "name": "flight_agent",
    "instructions": """You are a flight booking specialist agent. Your responsibility is to find flight options 
    based on the user's travel preferences. When you receive travel details, use the search_flights tool to find 
    suitable flight options. If you need information that's better handled by another agent, use the handoff tool.
    
    Always present flight options to the human for selection when available. If you encounter any issues with 
    finding flights, work with the human to find alternatives or resolve errors.""",
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
    """Execute the flight search and handle results with human interaction."""
    try:
        flights = search_flights(
            params["origin"],
            params["destination"],
            params["departure_date"],
            params["return_date"],
            params.get("budget")
        )
        
        if not flights:
            error_context = f"No flights found from {params['origin']} to {params['destination']}"
            error_resolution = handle_error(
                "No flights found matching your criteria",
                error_context
            )
            return f"Error: {error_resolution['message']}\nResolution: {error_resolution.get('resolution', 'No resolution provided')}"
        
        # Let human select from available flights
        selected_flight = get_human_selection(
            flights,
            f"Please select a flight from {params['origin']} to {params['destination']}"
        )
        
        if not selected_flight:
            return "No flight was selected."
        
        result = "Selected flight details:\n"
        result += f"- Airline: {selected_flight['airline']}\n"
        result += f"- Flight: {selected_flight['flight_number']}\n"
        result += f"- Price: {selected_flight['price']}\n"
        result += f"- Departure: {selected_flight['departure']}\n"
        result += f"- Arrival: {selected_flight['arrival']}\n"
        if 'duration' in selected_flight:
            result += f"- Duration: {selected_flight['duration']}\n"
        if 'stops' in selected_flight:
            result += f"- Stops: {selected_flight['stops']}\n"
        
        return result
    
    except Exception as e:
        error_resolution = handle_error(
            str(e),
            "Error occurred while searching for flights"
        )
        return f"Error: {error_resolution['message']}\nResolution: {error_resolution.get('resolution', 'No resolution provided')}"
