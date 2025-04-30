"""SerpAPI utility functions for travel search."""
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from serpapi import GoogleSearch
from config import SERPAPI_API_KEY, SERPAPI_TIMEOUT, SERPAPI_CACHE_DURATION, CACHE_DIR

def _get_cache_path(query_hash: str) -> str:
    """Get the cache file path for a query."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{query_hash}.json")

def _cache_results(query_hash: str, results: Dict[str, Any]) -> None:
    """Cache search results."""
    cache_path = _get_cache_path(query_hash)
    with open(cache_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().timestamp(),
            'results': results
        }, f)

def _get_cached_results(query_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached results if they exist and are not expired."""
    cache_path = _get_cache_path(query_hash)
    if not os.path.exists(cache_path):
        return None
    
    with open(cache_path, 'r') as f:
        cached = json.load(f)
    
    if datetime.now().timestamp() - cached['timestamp'] > SERPAPI_CACHE_DURATION:
        return None
    
    return cached['results']

def search_flights_serp(origin: str, destination: str, departure_date: str, return_date: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for flights using SerpAPI."""
    query = f"flights from {origin} to {destination} on {departure_date}"
    query_hash = f"flights_{origin}_{destination}_{departure_date}"
    
    # Check cache first
    cached = _get_cached_results(query_hash)
    if cached:
        return cached
    
    # Perform search
    params = {
        "engine": "google_flights",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us"
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract and format flight results
    flights = []
    if 'flights_results' in results:
        for flight in results['flights_results']:
            flight_info = {
                "airline": flight.get('airline', 'Unknown'),
                "flight_number": flight.get('flight_id', 'Unknown'),
                "price": flight.get('price', 'Unknown'),
                "departure": flight.get('departure_time', 'Unknown'),
                "arrival": flight.get('arrival_time', 'Unknown'),
                "duration": flight.get('duration', 'Unknown'),
                "stops": flight.get('stops', 'Unknown')
            }
            if budget and flight.get('price'):
                if float(flight['price'].replace('$', '')) <= float(budget.replace('$', '')):
                    flights.append(flight_info)
            else:
                flights.append(flight_info)
    
    # Cache results
    _cache_results(query_hash, flights)
    
    return flights

def search_hotels_serp(location: str, check_in: str, check_out: str, budget: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for hotels using SerpAPI."""
    query = f"hotels in {location} check in {check_in} check out {check_out}"
    query_hash = f"hotels_{location}_{check_in}_{check_out}"
    
    # Check cache first
    cached = _get_cached_results(query_hash)
    if cached:
        return cached
    
    # Perform search
    params = {
        "engine": "google_hotels",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us"
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract and format hotel results
    hotels = []
    if 'hotels_results' in results:
        for hotel in results['hotels_results']:
            hotel_info = {
                "name": hotel.get('name', 'Unknown'),
                "price": hotel.get('price', 'Unknown'),
                "rating": hotel.get('rating', 0),
                "amenities": hotel.get('amenities', []),
                "address": hotel.get('address', 'Unknown'),
                "website": hotel.get('website', 'Unknown'),
                "reviews": hotel.get('reviews', [])
            }
            if budget and hotel.get('price'):
                if float(hotel['price'].replace('$', '').split('/')[0]) <= float(budget.replace('$', '')):
                    hotels.append(hotel_info)
            else:
                hotels.append(hotel_info)
    
    # Cache results
    _cache_results(query_hash, hotels)
    
    return hotels

def get_destination_info_serp(destination: str) -> Dict[str, Any]:
    """Get destination information using SerpAPI."""
    query = f"travel guide {destination} tourist attractions"
    query_hash = f"destination_{destination}"
    
    # Check cache first
    cached = _get_cached_results(query_hash)
    if cached:
        return cached
    
    # Perform search
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us"
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract and format destination information
    info = {
        "name": destination,
        "description": "",
        "attractions": [],
        "weather": "",
        "best_time_to_visit": "",
        "local_tips": []
    }
    
    if 'knowledge_graph' in results:
        kg = results['knowledge_graph']
        info.update({
            "description": kg.get('description', ''),
            "weather": kg.get('weather', ''),
            "timezone": kg.get('timezone', ''),
            "currency": kg.get('currency', ''),
            "languages": kg.get('languages', [])
        })
    
    if 'organic_results' in results:
        attractions = []
        tips = []
        for result in results['organic_results'][:5]:
            if 'attractions' in result['title'].lower():
                attractions.extend(result.get('attractions', []))
            if 'tips' in result['title'].lower() or 'guide' in result['title'].lower():
                tips.append(result['snippet'])
        
        info['attractions'] = attractions[:5]  # Top 5 attractions
        info['local_tips'] = tips[:3]  # Top 3 tips
    
    # Cache results
    _cache_results(query_hash, info)
    
    return info 