"""Tools for API interactions."""
from typing import Optional, List, Dict, Any
from tools.serpapi_utils import search_flights_serp, search_hotels_serp, get_destination_info_serp

def search_flights(origin: str, destination: str, departure_date: str, return_date: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for flights using SerpAPI."""
    return search_flights_serp(origin, destination, departure_date, return_date, budget)

def search_hotels(location: str, check_in: str, check_out: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for hotels using SerpAPI."""
    return search_hotels_serp(location, check_in, check_out, budget)

def get_destination_info(destination: str) -> Dict[str, Any]:
    """Get information about a destination using SerpAPI."""
    return get_destination_info_serp(destination)
