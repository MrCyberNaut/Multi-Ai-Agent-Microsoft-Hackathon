# Travel Assistant LangGraph Structure

## Overview

The travel assistant uses LangGraph to create a directed graph of specialized agents that work together to plan and book travel arrangements. The graph is designed to handle complex travel planning tasks with human-in-the-loop validation at critical decision points.

## State Management

The system uses a `TravelState` Pydantic model to maintain the conversation state:

```python
class TravelState(BaseModel):
    messages: List[Dict[str, Any]]  # Conversation history
    flight_options: Optional[List[Dict[str, Any]]]  # Available flights
    hotel_options: Optional[List[Dict[str, Any]]]  # Available hotels
    itinerary: Optional[Dict[str, Any]]  # Final itinerary
    user_preferences: Dict[str, Any]  # User preferences
    error: Optional[str]  # Current error state
    error_history: List[Dict[str, Any]]  # Error tracking
    human_feedback: List[str]  # Human feedback history
```

## Graph Structure

### Nodes

1. **Supervisor Agent**

   - Entry point for user requests
   - Coordinates between specialist agents
   - Handles task delegation and error recovery

2. **Flight Agent**

   - Searches for flight options using SerpAPI
   - Handles flight-related queries
   - Gets human approval for selections

3. **Hotel Agent**

   - Searches for hotel options using SerpAPI
   - Handles accommodation queries
   - Gets human approval for selections

4. **Itinerary Agent**

   - Creates comprehensive travel plans
   - Incorporates flight and hotel selections
   - Gets human approval for final itinerary

5. **Parallel Search Node**
   - Runs flight and hotel searches concurrently
   - Improves response time for initial searches

### Edges

```
START → supervisor
supervisor → flight_agent
supervisor → hotel_agent
supervisor → parallel_search
supervisor → itinerary_agent
supervisor → END

flight_agent → supervisor
hotel_agent → supervisor
parallel_search → supervisor
itinerary_agent → supervisor
itinerary_agent → END
```

## Flow Control

1. User input enters through the supervisor
2. Supervisor analyzes request and delegates to appropriate agents
3. Agents can work in parallel when appropriate
4. Results flow back through supervisor for coordination
5. Final itinerary requires human approval before completion

## Human-in-the-Loop Integration

- Human validation points are integrated at critical decisions
- Approval required for:
  - Flight selections
  - Hotel selections
  - Final itinerary
  - Error resolution
- Can be disabled via configuration

## Error Handling

- All errors flow through supervisor for recovery
- Human intervention available for complex issues
- Error history maintained in state
- Graceful degradation when services unavailable

## Caching

- SerpAPI results are cached to reduce API calls
- Cache duration configurable
- Cache includes timestamp for expiration
- Separate caches for flights, hotels, and destination info
