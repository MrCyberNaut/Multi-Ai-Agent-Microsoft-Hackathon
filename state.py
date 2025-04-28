"""State definitions for the travel assistant."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class TravelState(MessagesState):
    """State for the travel assistant."""
    flight_options: Optional[List[Dict[str, Any]]] = Field(default=None)
    hotel_options: Optional[List[Dict[str, Any]]] = Field(default=None)
    itinerary: Optional[Dict[str, Any]] = Field(default=None)
    user_preferences: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None)
