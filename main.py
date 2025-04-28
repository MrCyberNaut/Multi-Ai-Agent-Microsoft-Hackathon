"""Main entry point for the travel assistant application."""
import json
import os
from openai import OpenAI
from agents import flight_agent, hotel_agent, itinerary_agent, supervisor, process_tool_calls, conversation_state
from config import OPENAI_API_KEY, MODEL_NAME

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def main():
    """Run the travel assistant."""
    # Store conversation messages
    messages = []
    current_agent = "supervisor"
    agent_definitions = {
        "supervisor": supervisor,
        "flight_agent": flight_agent,
        "hotel_agent": hotel_agent,
        "itinerary_agent": itinerary_agent
    }
    
    print("Travel Assistant initialized. Enter 'quit' to exit.")
    print("Example query: 'I want to book a trip from NYC to Miami from May 15-20, 2025. Budget is $1500.'")
    
    # Add initial system message
    messages.append({
        "role": "system",
        "content": agent_definitions[current_agent]["instructions"]
    })
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break
        
        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})
        
        # Continue conversation until complete
        while True:
            # Get response from current agent
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=agent_definitions[current_agent]["tools"],
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            
            # Check if the message has content to display
            if assistant_message.content:
                print(f"\nTravel Assistant ({current_agent}): {assistant_message.content}")
            
            # Check if agent used tools
            if assistant_message.tool_calls:
                # Process tool calls and get outputs
                tool_outputs, next_agent = process_tool_calls(assistant_message.tool_calls)
                
                # If handoff requested, switch agents
                if next_agent:
                    print(f"\n[System: Switching from {current_agent} to {next_agent}]")
                    current_agent = next_agent
                    
                    # Add new system message for the new agent
                    messages.append({
                        "role": "system",
                        "content": agent_definitions[current_agent]["instructions"]
                    })
                    
                    # Add context from the conversation state
                    context_message = "Here's what we know so far:\n"
                    if conversation_state["flight_options"]:
                        context_message += f"- Flight options are available\n"
                    if conversation_state["hotel_options"]:
                        context_message += f"- Hotel options are available\n"
                    if conversation_state["user_preferences"]:
                        context_message += f"- User preferences: {json.dumps(conversation_state['user_preferences'], indent=2)}\n"
                    
                    messages.append({"role": "system", "content": context_message})
                    
                    # Continue to next iteration with new agent
                    continue
                
                # Add tool outputs to messages
                for tool_output in tool_outputs:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_output["tool_call_id"],
                        "content": tool_output["output"]
                    })
                
                # Continue conversation with same agent
                continue
            
            # If no tool calls and no handoff, break out of the inner loop
            break

if __name__ == "__main__":
    main()
