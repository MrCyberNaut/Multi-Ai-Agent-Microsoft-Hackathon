"""Itinerary creation agent implementation."""
from config import MODEL_NAME

itinerary_agent = {
    "name": "itinerary_agent",
    "instructions": """You are an itinerary creation specialist. Your responsibility is to create comprehensive travel 
    itineraries based on flight and hotel information provided by other agents or the user. Create a well-formatted, 
    detailed itinerary that includes flight details, hotel information, check-in/check-out times, and a daily schedule.
    
    Always check if you have all the necessary information before creating the itinerary. If you need more information, 
    use the handoff tool to request it from the appropriate specialist agent.""",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "handoff",
                "description": "Hand off the conversation to another agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "enum": ["flight_agent", "hotel_agent"]}
                    },
                    "required": ["to"]
                }
            }
        }
    ]
}
