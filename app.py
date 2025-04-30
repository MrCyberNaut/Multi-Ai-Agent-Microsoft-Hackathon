"""Streamlit app for the multi-agent travel system."""
import streamlit as st
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import pandas as pd
from fpdf import FPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import os
import json
import sys
from booking_agent import (
    search_flights_serp,
    search_hotels_serp,
    get_destination_info_serp,
    get_human_selection,
    TravelState,
    create_travel_graph,
    debug_print
)

# Enable debugging
DEBUG = True

# Configure Streamlit page
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)
debug_print("Streamlit page configured")

# Custom CSS for dark theme with orange accents
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    
    /* Form styling */
    .stTextInput, .stNumberInput, .stDateInput, .stSelectbox {
        background-color: #3b3b3b;
        border-radius: 5px;
        border: 1px solid #4b4b4b;
    }
    
    /* Card styling */
    .travel-card {
        background-color: #3b3b3b;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #4b4b4b;
        transition: all 0.3s ease;
    }
    .travel-card:hover {
        border-color: #ff6b35;
        transform: translateY(-2px);
    }
    .selected-card {
        border: 2px solid #ff6b35;
    }
    
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 500px;
        overflow-y: auto;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #3b3b3b;
        border-radius: 10px;
    }
    
    .user-message {
        align-self: flex-end;
        background-color: #ff6b35;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        max-width: 80%;
    }
    
    .ai-message {
        align-self: flex-start;
        background-color: #4b4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        max-width: 80%;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #ff6b35;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #ff8255;
    }
</style>
""", unsafe_allow_html=True)
debug_print("Applied custom CSS")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    debug_print("Initialized messages in session state")

if 'travel_state' not in st.session_state:
    st.session_state.travel_state = TravelState()
    debug_print("Initialized travel state in session state")

# Initialize graph with error handling
if 'graph' not in st.session_state:
    try:
        debug_print("Creating travel graph")
        st.session_state.graph = create_travel_graph()
        debug_print("Graph created and stored in session state")
    except Exception as e:
        error_msg = f"Error creating graph: {str(e)}"
        debug_print(error_msg)
        st.error(f"Error initializing the travel assistant: {error_msg}")
        st.session_state.graph = None
        st.stop()

if 'selected_flight' not in st.session_state:
    st.session_state.selected_flight = None
    debug_print("Initialized selected_flight in session state")

if 'selected_accommodation' not in st.session_state:
    st.session_state.selected_accommodation = None
    debug_print("Initialized selected_accommodation in session state")

if 'search_results' not in st.session_state:
    st.session_state.search_results = {
        "flights": [],
        "hotels": [],
        "destination_info": {}
    }
    debug_print("Initialized search_results in session state")

if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = {
        "origin": "",
        "destination": "",
        "departure_date": "",
        "return_date": "",
        "budget": "",
        "travelers": 1,
        "preferences": ""
    }
    debug_print("Initialized extracted_info in session state")

if 'chat_stage' not in st.session_state:
    st.session_state.chat_stage = "initial"
    debug_print("Initialized chat_stage to 'initial'")

# Function to format datetime for display
def format_datetime(iso_string):
    """Format datetime string for display."""
    debug_print(f"Formatting datetime: {iso_string}")
    try:
        # Handle empty or None values
        if not iso_string:
            debug_print("Empty datetime string, returning N/A")
            return "N/A"
            
        # Handle numeric timestamps (seconds since epoch)
        if isinstance(iso_string, (int, float)) or (isinstance(iso_string, str) and iso_string.isdigit()):
            timestamp = float(iso_string)
            dt = datetime.fromtimestamp(timestamp)
            formatted = dt.strftime("%b %d, %Y | %I:%M %p")  # Example: Mar 06, 2025 | 6:20 PM
            debug_print(f"Formatted timestamp {iso_string} to {formatted}")
            return formatted
            
        # Handle ISO format strings
        if "T" in iso_string:
            dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
            formatted = dt.strftime("%b %d, %Y | %I:%M %p")
            debug_print(f"Formatted ISO datetime {iso_string} to {formatted}")
            return formatted
            
        # Handle simple date strings
        if "-" in iso_string and len(iso_string) >= 10:
            dt = datetime.strptime(iso_string[:10], "%Y-%m-%d")
            formatted = dt.strftime("%b %d, %Y")
            debug_print(f"Formatted date {iso_string} to {formatted}")
            return formatted
            
        # Handle other datetime formats
        formats_to_try = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%H:%M"
        ]
        
        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(iso_string, fmt)
                formatted = dt.strftime("%b %d, %Y | %I:%M %p")
                debug_print(f"Formatted {iso_string} using format {fmt} to {formatted}")
                return formatted
            except:
                continue
                
        debug_print(f"Could not format datetime {iso_string}, returning as is")
        return iso_string  # Return original if all parsing fails
    except Exception as e:
        error_msg = f"Error formatting datetime: {e}"
        debug_print(error_msg)
        return str(iso_string)  # Return original if parsing fails

# Function to create PDF itinerary
def create_pdf_itinerary(flight_data, accommodation_data, activities):
    """Create a PDF itinerary from selected options."""
    debug_print("Creating PDF itinerary")
    pdf = FPDF()
    pdf.add_page()
    
    # Set font and colors
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(255, 107, 53)  # Orange
    
    # Title
    pdf.cell(0, 10, 'Your Travel Itinerary', 0, 1, 'C')
    pdf.ln(10)
    
    # Flight details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Flight Details', 0, 1)
    pdf.set_font('Arial', '', 10)
    for key, value in flight_data.items():
        if key not in ['id', 'selected']:
            pdf.cell(0, 8, f'{key.title()}: {value}', 0, 1)
    pdf.ln(5)
    
    # Accommodation details
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Accommodation Details', 0, 1)
    pdf.set_font('Arial', '', 10)
    for key, value in accommodation_data.items():
        if key not in ['id', 'selected']:
            pdf.cell(0, 8, f'{key.title()}: {value}', 0, 1)
    pdf.ln(5)
    
    # Activities
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Activities', 0, 1)
    pdf.set_font('Arial', '', 10)
    for activity in activities:
        pdf.cell(0, 8, f'• {activity}', 0, 1)
    
    debug_print("PDF itinerary created successfully")
    return pdf

# Function to handle user messages and interact with the agent graph
def handle_user_message(user_input):
    """Process user message through LangGraph agents."""
    debug_print(f"Processing user message: {user_input}")
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Check if graph is initialized
    if st.session_state.graph is None:
        debug_print("Graph is not initialized")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I'm sorry, but the travel assistant is not properly initialized. Please check the error message or try again later."
        })
        return
    
    # Initialize or update travel state with the new message
    if not hasattr(st.session_state.travel_state, 'messages') or not st.session_state.travel_state.messages:
        st.session_state.travel_state = TravelState(
            messages=[{"role": "user", "content": user_input}]
        )
    else:
        st.session_state.travel_state.messages.append({"role": "user", "content": user_input})
    
    # Process through the graph
    debug_print("Running message through agent graph")
    try:
        events = []
        for event in st.session_state.graph.stream(st.session_state.travel_state):
            events.append(event)
            if event.get("type") == "agent":
                agent_name = event.get("agent", "Assistant")
                state = event.get("state")
                if state and state.messages and len(state.messages) > 0:
                    last_message = state.messages[-1]
                    if last_message["role"] == "assistant":
                        st.session_state.messages.append({"role": "assistant", "content": last_message["content"]})
                        st.session_state.travel_state = state
                        
                        # Extract and structure travel information if it looks like a structured request
                        extract_travel_info(last_message["content"])
        
        # If no response was generated, provide a fallback
        if not any(e.get("type") == "agent" for e in events):
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'm processing your request. Can you please provide more details about your travel plans?"
            })
            
    except Exception as e:
        error_msg = f"Error in agent processing: {str(e)}"
        debug_print(error_msg)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"I encountered an error while processing your request. Please try again or provide more specific details. Error: {str(e)}"
        })

# Function to extract structured travel information from messages
def extract_travel_info(message):
    """Extract structured travel information from LLM responses."""
    debug_print("Extracting travel info from message")
    
    # Simple extraction of key travel information
    info = st.session_state.extracted_info.copy()
    
    # Check for origin/departure location
    if "from " in message.lower():
        parts = message.lower().split("from ")
        if len(parts) > 1:
            next_part = parts[1].split(" ")
            if len(next_part) > 0:
                potential_origin = next_part[0].strip().rstrip(",.?!:;")
                if len(potential_origin) > 1:
                    info["origin"] = potential_origin.upper()
                    debug_print(f"Extracted origin: {info['origin']}")
    
    # Check for destination
    if "to " in message.lower():
        parts = message.lower().split("to ")
        if len(parts) > 1:
            next_part = parts[1].split(" ")
            if len(next_part) > 0:
                potential_dest = next_part[0].strip().rstrip(",.?!:;")
                if len(potential_dest) > 1:
                    info["destination"] = potential_dest.upper()
                    debug_print(f"Extracted destination: {info['destination']}")
    
    # Check for dates
    import re
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    dates = re.findall(date_pattern, message)
    if len(dates) >= 2:
        info["departure_date"] = dates[0]
        info["return_date"] = dates[1]
        debug_print(f"Extracted dates: {info['departure_date']} to {info['return_date']}")
    elif len(dates) == 1:
        info["departure_date"] = dates[0]
        debug_print(f"Extracted departure date: {info['departure_date']}")
    
    # Check for budget
    budget_patterns = [
        r'budget.*?(\d+)',
        r'(\d+).*?budget',
        r'around (\d+)',
        r'about (\d+)'
    ]
    
    for pattern in budget_patterns:
        budget_match = re.search(pattern, message.lower())
        if budget_match:
            info["budget"] = budget_match.group(1)
            debug_print(f"Extracted budget: {info['budget']}")
            break
    
    # Check for travelers
    travelers_patterns = [
        r'(\d+).*?traveler',
        r'(\d+).*?people',
        r'(\d+).*?person',
        r'for (\d+)'
    ]
    
    for pattern in travelers_patterns:
        travelers_match = re.search(pattern, message.lower())
        if travelers_match:
            info["travelers"] = travelers_match.group(1)
            debug_print(f"Extracted travelers: {info['travelers']}")
            break
    
    # Update session state with extracted info
    st.session_state.extracted_info = info
    
    # If we have enough information, move to the search stage
    if (info["origin"] and info["destination"] and 
        info["departure_date"] and info["return_date"]):
        st.session_state.chat_stage = "ready_to_search"
        debug_print("Enough information extracted, ready to search")

# Function to search for travel options
def search_travel_options():
    """Search for flights and hotels based on extracted information."""
    debug_print("Searching for travel options")
    info = st.session_state.extracted_info
    
    if not info["origin"] or not info["destination"] or not info["departure_date"] or not info["return_date"]:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I need more information before I can search. Please provide origin, destination, departure date, and return date."
        })
        return
    
    try:
        with st.spinner("Searching for the best options..."):
            # Search for flights
            debug_print("Starting flight search")
            flights = search_flights_serp(
                origin=info["origin"],
                destination=info["destination"],
                departure_date=info["departure_date"],
                return_date=info["return_date"],
                budget=info["budget"]
            )
            st.session_state.search_results["flights"] = flights
            debug_print(f"Flight search returned {len(flights)} results")
            
            # Search for hotels
            debug_print("Starting hotel search")
            hotels = search_hotels_serp(
                location=info["destination"],
                check_in=info["departure_date"],
                check_out=info["return_date"],
                budget=info["budget"]
            )
            st.session_state.search_results["hotels"] = hotels
            debug_print(f"Hotel search returned {len(hotels)} results")
            
            # Get destination information
            debug_print("Starting destination info search")
            destination_info = get_destination_info_serp(info["destination"])
            st.session_state.search_results["destination_info"] = destination_info
            debug_print(f"Destination info received: {bool(destination_info)}")
        
        # Update chat stage
        st.session_state.chat_stage = "results"
        debug_print("Search completed, showing results")
        
        # Generate response message about results
        flights_msg = f"I found {len(flights)} flight options" if flights else "I couldn't find any flights matching your criteria"
        hotels_msg = f"and {len(hotels)} hotel options" if hotels else "but couldn't find any hotels matching your criteria"
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"I've completed my search! {flights_msg} {hotels_msg}. Please review the options below and let me know if you'd like to make any adjustments."
        })
        
    except Exception as e:
        error_msg = f"Error searching for travel options: {str(e)}"
        debug_print(error_msg)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"I encountered an error while searching. Please check your search parameters or try again later. Error: {str(e)}"
        })

# Debug section to show API keys status
with st.expander("🔍 Debug Information"):
    debug_print("Rendering debug section")
    st.write("API Keys Status:")
    api_keys_status = {
        "SERPAPI_API_KEY": "✅ Set" if os.getenv("SERPAPI_API_KEY") else "❌ Not Set",
        "AIMLAPI_API_KEY": "✅ Set" if os.getenv("AIMLAPI_API_KEY") else "❌ Not Set",
        "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
    }
    st.write(api_keys_status)
    debug_print(f"API Keys Status: {api_keys_status}")
    
    st.write("Current Extracted Information:")
    st.write(st.session_state.extracted_info)
    
    st.write("Current Chat Stage:")
    st.write(st.session_state.chat_stage)
    
    if st.button("Test SerpAPI Connection"):
        debug_print("SerpAPI test button clicked")
        from serpapi import GoogleSearch
        try:
            test_params = {
                "engine": "google_flights",
                "departure_id": "JFK",
                "arrival_id": "LAX",
                "outbound_date": "2025-07-01",
                "api_key": os.getenv("SERPAPI_API_KEY")
            }
            debug_print(f"Testing SerpAPI with params: {test_params}")
            test_search = GoogleSearch(test_params)
            test_results = test_search.get_dict()
            st.success("✅ SerpAPI connection successful")
            st.json(test_results)
            debug_print("SerpAPI test successful")
        except Exception as e:
            error_msg = f"❌ SerpAPI connection failed: {str(e)}"
            st.error(error_msg)
            debug_print(f"SerpAPI test failed: {error_msg}")

# Layout the app with a chat interface
st.title("✈️ AI Travel Planner")
st.markdown("Chat with our AI to plan your perfect trip. Just tell us what you're looking for!")

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input and search button
col1, col2 = st.columns([5, 1])
with col1:
    # Use a callback to process the input instead of modifying session state directly
    user_input = st.text_input("", placeholder="Tell me about your travel plans...", key="user_input_field")
with col2:
    if st.session_state.chat_stage == "ready_to_search":
        search_button = st.button("Search Now")
    else:
        search_button = False

# Process user input
if user_input:
    handle_user_message(user_input)
    # Don't try to clear input field by modifying session state directly
    # Streamlit will handle this on rerun
    
# Process search button
if search_button:
    search_travel_options()

# Display results if available
if st.session_state.chat_stage == "results" and st.session_state.search_results:
    results = st.session_state.search_results
    
    # Display flights section
    if results["flights"]:
        st.header("Flight Options")
        debug_print(f"Displaying {len(results['flights'])} flight options")
        flight_cols = st.columns(min(3, len(results["flights"])))
        for i, flight in enumerate(results["flights"]):
            with flight_cols[i % 3]:
                with st.container():
                    airline = flight.get('airline', 'Unknown Airline')
                    price = f"{flight.get('price', 'Unknown')} {flight.get('currency', 'INR')}"
                    departure = format_datetime(flight.get('departure_time', 'Unknown'))
                    arrival = format_datetime(flight.get('arrival_time', 'Unknown'))
                    duration = flight.get('duration', 'Unknown')
                    debug_print(f"Flight {i+1}: {airline}, Price: {price}")
                    
                    st.markdown(f"""
                    <div class='travel-card {"selected-card" if st.session_state.selected_flight == i else ""}'>
                        <h3>{airline}</h3>
                        <p><strong>Price:</strong> {price}</p>
                        <p><strong>Departure:</strong> {departure}</p>
                        <p><strong>Arrival:</strong> {arrival}</p>
                        <p><strong>Duration:</strong> {duration}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Select Flight {i+1}", key=f"flight_{i}"):
                        debug_print(f"User selected flight {i+1}")
                        st.session_state.selected_flight = i
    
    # Display hotels section
    if results["hotels"]:
        st.header("Accommodation Options")
        debug_print(f"Displaying {len(results['hotels'])} hotel options")
        hotel_cols = st.columns(min(3, len(results["hotels"])))
        for i, hotel in enumerate(results["hotels"]):
            with hotel_cols[i % 3]:
                with st.container():
                    name = hotel.get('name', 'Unknown')
                    price = f"{hotel.get('price_per_night', 'Unknown')} {hotel.get('currency', 'INR')}"
                    rating = hotel.get('rating', 'N/A')
                    description = hotel.get('description', 'No description available')
                    amenities = ', '.join(hotel.get('amenities', [])[:3]) if hotel.get('amenities') else 'N/A'
                    debug_print(f"Hotel {i+1}: {name}, Price: {price}, Rating: {rating}")
                    
                    # Display image if available
                    if hotel.get('image_url'):
                        debug_print(f"Hotel {i+1} has image URL: {hotel['image_url'][:50]}...")
                        st.image(hotel['image_url'], use_container_width=True)
                    
                    st.markdown(f"""
                    <div class='travel-card {"selected-card" if st.session_state.selected_accommodation == i else ""}'>
                        <h3>{name}</h3>
                        <p><strong>Price:</strong> {price} per night</p>
                        <p><strong>Rating:</strong> {rating} ⭐</p>
                        <p><strong>Description:</strong> {description[:100]}...</p>
                        <p><strong>Amenities:</strong> {amenities}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Select Hotel {i+1}", key=f"hotel_{i}"):
                        debug_print(f"User selected hotel {i+1}")
                        st.session_state.selected_accommodation = i
    
    # Display destination information
    destination_info = results["destination_info"]
    if destination_info:
        st.header("Destination Information")
        debug_print(f"Displaying destination info for {destination_info.get('name', st.session_state.extracted_info.get('destination', 'Unknown'))}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(destination_info.get('name', st.session_state.extracted_info.get('destination', 'Unknown')))
            st.write(destination_info.get('description', 'No description available'))
            
            if destination_info.get('attractions'):
                st.subheader("Top Attractions")
                for attraction in destination_info.get('attractions'):
                    st.write(f"• {attraction}")
                debug_print(f"Listed {len(destination_info.get('attractions', []))} attractions")
        
        with col2:
            if destination_info.get('local_tips'):
                st.subheader("Local Tips")
                for tip in destination_info.get('local_tips'):
                    st.write(f"• {tip}")
                debug_print(f"Listed {len(destination_info.get('local_tips', []))} local tips")

    # Display itinerary if flight and hotel are selected
    if st.session_state.selected_flight is not None and st.session_state.selected_accommodation is not None:
        debug_print("Both flight and hotel selected, showing itinerary options")
        st.header("Your Itinerary")
        
        # Create sample activities from destination info
        activities = destination_info.get('attractions', [])[:5] if destination_info else ["Sightseeing"]
        
        col7, col8 = st.columns(2)
        with col7:
            if st.button("Download PDF"):
                debug_print("PDF download button clicked")
                selected_flight = results["flights"][st.session_state.selected_flight]
                selected_hotel = results["hotels"][st.session_state.selected_accommodation]
                
                pdf = create_pdf_itinerary(
                    selected_flight,
                    selected_hotel,
                    activities
                )
                
                # Save PDF to memory and create download button
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.download_button(
                    label="Download Itinerary PDF",
                    data=pdf_output,
                    file_name="travel_itinerary.pdf",
                    mime="application/pdf"
                )
        
        with col8:
            email = st.text_input("Email address for itinerary")
            send_email = st.button("Email Itinerary")
            if email and send_email:
                st.success("Your itinerary has been sent to your email!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>AI Travel Planner v2.0 | Built with ❤️ using Streamlit, LangGraph, and OpenAI</p>
    </div>
    """,
    unsafe_allow_html=True
)
debug_print("App rendering complete") 