"""Test script for the travel booking agent system."""
import os
import sys
from datetime import datetime
import json
from booking_agent import (
    search_flights_serp,
    search_hotels_serp,
    get_destination_info_serp,
    TravelState,
    create_travel_graph,
    SERPAPI_API_KEY
)

def test_serpapi_connection():
    """Test the SerpAPI connection directly."""
    print("\n=== Testing SerpAPI Connection ===")
    if not SERPAPI_API_KEY:
        print("ERROR: SERPAPI_API_KEY is not set in the environment")
        return False
    
    print(f"SERPAPI_API_KEY is set: {SERPAPI_API_KEY[:5]}...{SERPAPI_API_KEY[-5:]}")
    
    # Test flight search
    print("\n--- Testing Flight Search ---")
    try:
        origin = "BOM"  # Mumbai
        destination = "IXE"  # Mangalore
        departure_date = "2025-05-03"
        return_date = "2025-05-06"
        print(f"Searching flights: {origin} to {destination}, {departure_date} to {return_date}")
        
        flights = search_flights_serp(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            budget="5000"
        )
        
        print(f"Flight search results: {len(flights)}")
        if flights:
            print("Sample flight:")
            print(json.dumps(flights[0], indent=2))
        else:
            print("No flights found - check debug file")
            debug_file = f"cache/debug_flights_{origin}_{destination}_{departure_date}.json"
            if os.path.exists(debug_file):
                with open(debug_file, 'r') as f:
                    debug_data = json.load(f)
                if "error" in debug_data:
                    print(f"API Error: {debug_data['error']}")
                print(f"API Response Keys: {list(debug_data.keys())}")
                
                # Check full cached response directly to see what's happening
                print("\nLooking at raw API response for debugging:")
                if "flights_results" in debug_data:
                    flights_data = debug_data["flights_results"]
                    print(f"Flights data available: {bool(flights_data)}")
                    if "flights" in flights_data:
                        print(f"Number of flights in response: {len(flights_data['flights'])}")
                        if len(flights_data["flights"]) > 0:
                            print("Sample flight from API:")
                            sample_flight = flights_data["flights"][0]
                            print(f"Airline: {sample_flight.get('airlines', [{}])[0].get('name', 'Unknown')}")
                            print(f"Price: {sample_flight.get('price', {}).get('total', {}).get('amount', 'Unknown')}")
                    else:
                        print("No 'flights' key in flights_results")
                        print(f"Keys in flights_results: {list(flights_data.keys())}")
                        
    except Exception as e:
        print(f"Flight search error: {str(e)}")
    
    # Test hotel search
    print("\n--- Testing Hotel Search ---")
    try:
        location = "IXE"  # Mangalore
        check_in = "2025-05-03"
        check_out = "2025-05-06"
        print(f"Searching hotels in {location}, {check_in} to {check_out}")
        
        hotels = search_hotels_serp(
            location=location,
            check_in=check_in,
            check_out=check_out,
            budget="5000"
        )
        
        print(f"Hotel search results: {len(hotels)}")
        if hotels:
            print("Sample hotel:")
            print(json.dumps(hotels[0], indent=2))
        else:
            print("No hotels found - check debug file")
            debug_file = f"cache/debug_hotels_{location}_{check_in}_{check_out}.json"
            if os.path.exists(debug_file):
                with open(debug_file, 'r') as f:
                    debug_data = json.load(f)
                if "error" in debug_data:
                    print(f"API Error: {debug_data['error']}")
                print(f"API Response Keys: {list(debug_data.keys())}")
    except Exception as e:
        print(f"Hotel search error: {str(e)}")
    
    # Test destination info search
    print("\n--- Testing Destination Info ---")
    try:
        destination_name = "Mangalore"
        print(f"Getting info for {destination_name}")
        
        info = get_destination_info_serp(destination_name)
        
        if info:
            print("Destination info found:")
            print(f"Name: {info.get('name')}")
            print(f"Description: {info.get('description')[:100]}..." if info.get('description') else "No description")
            print(f"Attractions: {len(info.get('attractions', []))}")
            print(f"Local tips: {len(info.get('local_tips', []))}")
        else:
            print("No destination info found")
    except Exception as e:
        print(f"Destination info error: {str(e)}")
    
    return True

def test_direct_search_call():
    """Test direct calls to the search functions with new parameters."""
    print("\n=== Testing Direct API Calls with Fresh Parameters ===")
    
    # Test flight search with fresh parameters
    print("\n--- Testing Fresh Flight Search ---")
    try:
        origin = "DEL"  # Delhi
        destination = "BOM"  # Mumbai
        departure_date = "2025-06-01"
        return_date = "2025-06-10"
        print(f"Searching flights: {origin} to {destination}, {departure_date} to {return_date}")
        
        # Remove any existing cache file
        cache_file = f"cache/flights_{origin}_{destination}_{departure_date}.json"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed existing cache file: {cache_file}")
        
        flights = search_flights_serp(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            budget="10000"
        )
        
        print(f"Flight search results: {len(flights)}")
        if flights:
            print("Sample flight:")
            print(json.dumps(flights[0], indent=2))
        else:
            print("No flights found - check debug file")
            debug_file = f"cache/debug_flights_{origin}_{destination}_{departure_date}.json"
            if os.path.exists(debug_file):
                with open(debug_file, 'r') as f:
                    debug_data = json.load(f)
                if "error" in debug_data:
                    print(f"API Error: {debug_data['error']}")
                print(f"API Response Keys: {list(debug_data.keys())}")
    except Exception as e:
        print(f"Flight search error: {str(e)}")

    # Fresh hotel search
    print("\n--- Testing Fresh Hotel Search ---")
    try:
        location = "GOI"  # Goa
        check_in = "2025-06-01"
        check_out = "2025-06-10"
        print(f"Searching hotels in {location}, {check_in} to {check_out}")
        
        # Remove any existing cache file
        cache_file = f"cache/hotels_{location}_{check_in}_{check_out}.json"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed existing cache file: {cache_file}")
        
        hotels = search_hotels_serp(
            location=location,
            check_in=check_in,
            check_out=check_out,
            budget="10000"
        )
        
        print(f"Hotel search results: {len(hotels)}")
        if hotels:
            print("Sample hotel:")
            print(json.dumps(hotels[0], indent=2))
        else:
            print("No hotels found - check debug file")
            debug_file = f"cache/debug_hotels_{location}_{check_in}_{check_out}.json"
            if os.path.exists(debug_file):
                with open(debug_file, 'r') as f:
                    debug_data = json.load(f)
                if "error" in debug_data:
                    print(f"API Error: {debug_data['error']}")
                print(f"API Response Keys: {list(debug_data.keys())}")
    except Exception as e:
        print(f"Hotel search error: {str(e)}")

if __name__ == "__main__":
    print("=== Travel Booking Agent Test ===")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Force sys.modules to not include streamlit
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    
    # Run tests
    test_serpapi_connection()
    test_direct_search_call() 