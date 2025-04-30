"""Supervisor agent implementation."""
from typing import Dict, Any, List, Tuple, Optional
import os
from openai import OpenAI
from tools import search_flights, search_hotels, get_destination_info
from hitl import get_human_approval, handle_error
from config import MODEL_NAME, OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# State variables to store conversation context
conversation_state = {
    "flight_options": None,
    "hotel_options": None,
    "user_preferences": {},
    "error_history": [],
    "human_feedback": []
}

triage_agent = {
    "name": "triage_agent",
    "instructions": """You are the initial triage agent for a travel planning system. Your job is to:
    
    1. Analyze the user's request to determine what kind of travel assistance they need
    2. Extract key travel details like origin, destination, dates, and preferences
    3. Route the request to the appropriate specialist agent:
       - flight_agent: For flight booking and information
       - hotel_agent: For hotel booking and accommodation
       - itinerary_agent: For creating a complete travel itinerary (only when flight and hotel info are known)
    
    Always be helpful and warm in your responses, but focus on efficiently routing requests.
    Get human approval for critical decisions and handle errors gracefully.""",
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
                "name": "get_destination_info",
                "description": "Get information about a destination",
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
                        "to": {"type": "string", "enum": ["flight_agent", "hotel_agent", "itinerary_agent"]}
                    },
                    "required": ["to"]
                }
            }
        }
    ]
}

def process_tool_calls(tool_calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Process tool calls from an agent and return the tool outputs."""
    tool_outputs = []
    from .flight_agent import handle_search_flights
    from .hotel_agent import handle_search_hotels
    
    # Map tool names to their handler functions
    tool_handlers = {
        "search_flights": handle_search_flights,
        "search_hotels": handle_search_hotels,
        "get_destination_info": get_destination_info
    }
    
    for tool_call in tool_calls:
        try:
            function_name = tool_call["function"]["name"]
            function_args = eval(tool_call["function"]["arguments"])
            
            if function_name == "handoff":
                # Get human approval for handoff
                handoff_approved = get_human_approval(
                    {"to": function_args['to']},
                    f"Approve handoff to {function_args['to']}?"
                )
                
                if not handoff_approved:
                    tool_outputs.append({
                        "tool_call_id": tool_call["id"],
                        "output": "Handoff was not approved by human operator."
                    })
                    continue
                
                # Handle agent handoff
                tool_outputs.append({
                    "tool_call_id": tool_call["id"],
                    "output": f"Handing off to {function_args['to']}"
                })
                return tool_outputs, function_args["to"]
            
            elif function_name in tool_handlers:
                # Call the appropriate tool handler
                result = tool_handlers[function_name](function_args)
                
                # Store results in conversation state
                if function_name == "search_flights":
                    conversation_state["flight_options"] = result
                    # Store user preferences
                    conversation_state["user_preferences"].update({
                        "origin": function_args["origin"],
                        "destination": function_args["destination"],
                        "departure_date": function_args["departure_date"],
                        "return_date": function_args["return_date"]
                    })
                
                elif function_name == "search_hotels":
                    conversation_state["hotel_options"] = result
                    # Store user preferences
                    conversation_state["user_preferences"].update({
                        "destination": function_args["location"],
                        "check_in": function_args["check_in"],
                        "check_out": function_args["check_out"]
                    })
                
                tool_outputs.append({
                    "tool_call_id": tool_call["id"],
                    "output": result
                })
        
        except Exception as e:
            error_resolution = handle_error(
                str(e),
                f"Error processing tool call: {function_name}"
            )
            
            # Store error in history
            conversation_state["error_history"].append({
                "tool": function_name,
                "error": str(e),
                "resolution": error_resolution
            })
            
            tool_outputs.append({
                "tool_call_id": tool_call["id"],
                "output": f"Error: {error_resolution['message']}\nResolution: {error_resolution.get('resolution', 'No resolution provided')}"
            })
    
    return tool_outputs, None
