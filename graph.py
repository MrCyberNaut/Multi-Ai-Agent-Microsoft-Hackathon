"""Graph construction for the travel assistant."""
from typing import Literal
from langgraph.graph import StateGraph, START, END, Parallel
from langgraph.types import Command
from state import TravelState
from agents import flight_agent, hotel_agent, itinerary_agent, supervisor

def create_travel_assistant_graph() -> StateGraph:
    """Create and return the travel assistant graph."""
    # Create the graph
    builder = StateGraph(TravelState)
    
    # Add individual nodes
    builder.add_node("supervisor", supervisor)
    builder.add_node("flight_agent", flight_agent)
    builder.add_node("hotel_agent", hotel_agent)
    builder.add_node("itinerary_agent", itinerary_agent)
    
    # Define a parallel node for flight and hotel search
    parallel_search = Parallel(
        "__parallel_search",
        {
            "flight": flight_agent,
            "hotel": hotel_agent
        }
    )
    builder.add_node("parallel_search", parallel_search)
    
    # Add human intervention node
    def human_intervention(state: TravelState) -> Command[Literal["supervisor"]]:
        """Handle cases where human intervention is needed."""
        # In a real application, this would be an API endpoint or UI element
        human_feedback = input("Human feedback required: ")
        
        return Command(
            goto="supervisor",
            update={"messages": [{"type": "human", "content": human_feedback}]}
        )
    
    builder.add_node("human", human_intervention)
    
    # Add edges
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "flight_agent")
    builder.add_edge("supervisor", "hotel_agent")
    builder.add_edge("supervisor", "parallel_search")
    builder.add_edge("supervisor", "itinerary_agent")
    builder.add_edge("supervisor", "human")
    builder.add_edge("supervisor", END)
    
    builder.add_edge("flight_agent", "supervisor")
    builder.add_edge("hotel_agent", "supervisor")
    builder.add_edge("parallel_search", "supervisor")
    builder.add_edge("itinerary_agent", "supervisor")
    builder.add_edge("itinerary_agent", END)
    builder.add_edge("human", "supervisor")
    
    # Compile the graph
    return builder.compile()
