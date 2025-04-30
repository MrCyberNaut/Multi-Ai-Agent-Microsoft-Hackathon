"""Itinerary creation agent implementation."""
from typing import Dict, Any
from tools import get_destination_info
from hitl import validate_itinerary, get_human_feedback, handle_error

itinerary_agent = {
    "name": "itinerary_agent",
    "instructions": """You are an itinerary creation specialist. Your responsibility is to create comprehensive travel 
    itineraries based on flight and hotel information provided by other agents or the user. Create a well-formatted, 
    detailed itinerary that includes flight details, hotel information, check-in/check-out times, and a daily schedule.
    
    Always get human validation for the itineraries you create. If the human requests changes, incorporate their feedback
    and present the updated itinerary for approval. Use destination information to suggest activities and attractions.
    
    If you need more information, use the handoff tool to request it from the appropriate specialist agent.""",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_destination_info",
                "description": "Get detailed information about a destination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "Name of the destination"}
                    },
                    "required": ["destination"]
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
                        "to": {"type": "string", "enum": ["flight_agent", "hotel_agent"]}
                    },
                    "required": ["to"]
                }
            }
        }
    ]
}

def create_itinerary(destination: str, flight_info: Dict[str, Any], hotel_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create and validate an itinerary with human interaction."""
    try:
        # Get destination information
        dest_info = get_destination_info(destination)
        
        # Create initial itinerary
        itinerary = {
            "destination": destination,
            "destination_info": dest_info,
            "flight": flight_info,
            "hotel": hotel_info,
            "daily_schedule": []
        }
        
        # Add suggested activities based on destination info
        if 'attractions' in dest_info:
            for attraction in dest_info['attractions']:
                itinerary['daily_schedule'].append({
                    "activity": f"Visit {attraction}",
                    "duration": "2-3 hours",
                    "type": "attraction"
                })
        
        # Get human validation and feedback
        validated_itinerary = validate_itinerary(itinerary)
        
        # If changes are needed, get specific feedback for each section
        while validated_itinerary.get('needs_revision', False):
            feedback = get_human_feedback("Please provide specific changes needed for the itinerary:")
            
            # Update itinerary based on feedback
            validated_itinerary['revision_history'] = validated_itinerary.get('revision_history', [])
            validated_itinerary['revision_history'].append({
                "feedback": feedback,
                "previous_state": validated_itinerary.copy()
            })
            
            # Clear the needs_revision flag for the next validation
            validated_itinerary['needs_revision'] = False
            
            # Validate the updated itinerary
            validated_itinerary = validate_itinerary(validated_itinerary)
        
        return validated_itinerary
    
    except Exception as e:
        error_resolution = handle_error(
            str(e),
            "Error occurred while creating itinerary"
        )
        return {
            "status": "error",
            "message": error_resolution['message'],
            "resolution": error_resolution.get('resolution', 'No resolution provided')
        }
