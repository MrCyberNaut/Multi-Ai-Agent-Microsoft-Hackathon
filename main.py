"""Main entry point for the multi-agent travel system."""
import os
import sys
from dotenv import load_dotenv
from booking_agent import (
    create_travel_graph,
    TravelState,
    debug_print
)

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the CLI interface."""
    debug_print("Initializing the travel assistant")
    
    # Check if required environment variables are set
    if not os.getenv("SERPAPI_API_KEY"):
        print("Error: SERPAPI_API_KEY environment variable is not set.")
        print("Please set it in the .env file or as an environment variable.")
        return
    
    if not (os.getenv("AIMLAPI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("Error: Neither AIMLAPI_API_KEY nor OPENAI_API_KEY environment variables are set.")
        print("Please set one of them in the .env file or as an environment variable.")
        return
    
    try:
        # Initialize the graph
        debug_print("Creating travel graph")
        graph = create_travel_graph()
        debug_print("Graph created")
        
        # Check if we're running in Streamlit
        in_streamlit = 'streamlit' in sys.modules
        if in_streamlit:
            debug_print("Running in Streamlit, returning graph")
            return graph
        
        # Start CLI interaction
        print("\n===== Travel Assistant CLI =====")
        print("Welcome to the Travel Assistant! How can I help you plan your trip today?")
        print("(Type 'quit' or 'exit' to end the session)")
        
        # CLI interaction loop
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Thank you for using the Travel Assistant. Goodbye!")
                break
            
            # Create state with user message
            state = TravelState(
                messages=[{"role": "user", "content": user_input}]
            )
            
            # Process through the graph
            debug_print("Running message through agent graph")
            try:
                has_response = False
                # Use the stream method instead of run
                for event in graph.stream(state):
                    if event.get("type") == "agent":
                        agent_name = event.get("agent", "Assistant")
                        response_state = event.get("state")
                        if response_state and response_state.messages and len(response_state.messages) > 0:
                            last_message = response_state.messages[-1]
                            if last_message["role"] == "assistant":
                                print(f"\nAssistant: {last_message['content']}")
                                has_response = True
                
                # Fallback if no response
                if not has_response:
                    print("\nAssistant: I'm processing your request. Can you please provide more details?")
                    
            except Exception as e:
                error_msg = f"Error processing your request: {str(e)}"
                print(f"\nAssistant: {error_msg}")
                debug_print(f"Exception in graph execution: {error_msg}")
    
    except Exception as e:
        error_msg = f"System error: {str(e)}"
        print(f"Error: {error_msg}")
        debug_print(f"Exception in main: {error_msg}")

if __name__ == "__main__":
    main()
