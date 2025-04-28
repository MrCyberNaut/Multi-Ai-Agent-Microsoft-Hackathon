"""Supervisor agent implementation."""
from typing import Dict, Any, List, Tuple, Optional
import os
from openai import OpenAI
from tools import search_flights, search_hotels
from config import MODEL_NAME, OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# State variables to store conversation context
conversation_state = {
    "flight_options": None,
    "hotel_options": None,
    "user_preferences": {}
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
    
    Always be helpful and warm in your responses, but focus on efficiently routing requests.""",
    "tools": [
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
        "search_hotels": handle_search_hotels
    }
    
    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        function_args = eval(tool_call["function"]["arguments"])
        
        if function_name == "handoff":
            # Handle agent handoff (no output needed)
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
                conversation_state["flight_options"] = search_flights(
                    function_args["origin"],
                    function_args["destination"],
                    function_args["departure_date"],
                    function_args["return_date"],
                    function_args.get("budget")
                )
                # Store user preferences
                conversation_state["user_preferences"].update({
                    "origin": function_args["origin"],
                    "destination": function_args["destination"],
                    "departure_date": function_args["departure_date"],
                    "return_date": function_args["return_date"]
                })
            
            elif function_name == "search_hotels":
                conversation_state["hotel_options"] = search_hotels(
                    function_args["location"],
                    function_args["check_in"],
                    function_args["check_out"],
                    function_args.get("budget")
                )
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
    
    return tool_outputs, None
