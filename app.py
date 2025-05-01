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
import importlib
from booking_agent import (
    search_flights_serp,
    search_hotels_serp,
    get_destination_info_serp,
    get_human_selection,
    TravelState,
    create_travel_graph,
    debug_print
)
import time

# Load config
from config import LLM_PROVIDER, OLLAMA_MODEL

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
    
    /* Processing message animation */
    .processing-message {
        background-color: #5a5a5a;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    
    /* Typing indicator dots */
    .typing-indicator {
        display: inline-block;
    }
    
    .typing-indicator span {
        height: 8px;
        width: 8px;
        background-color: white;
        display: inline-block;
        border-radius: 50%;
        margin: 0 2px;
        opacity: 0.7;
        animation: bounce 1.3s linear infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.15s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.3s;
    }
    
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-4px); }
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

# Initialize processing state
if 'pending_message' not in st.session_state:
    st.session_state.pending_message = None
    debug_print("Initialized pending_message in session state")
    
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
    debug_print("Initialized is_processing in session state")

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

if 'selected_flight_data' not in st.session_state:
    st.session_state.selected_flight_data = None
    debug_print("Initialized selected_flight_data in session state")

if 'selected_accommodation' not in st.session_state:
    st.session_state.selected_accommodation = None
    debug_print("Initialized selected_accommodation in session state")

if 'selected_hotel_data' not in st.session_state:
    st.session_state.selected_hotel_data = None
    debug_print("Initialized selected_hotel_data in session state")

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
    
    # Add a temporary processing indicator message with spinner
    processing_id = str(len(st.session_state.messages))
    with st.spinner("Processing your request..."):
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "⏳ Analyzing your travel request...", 
            "is_processing": True, 
            "id": processing_id
        })
    
    # Store the processing status
    st.session_state.is_processing = True
    
    # Force refresh to show the processing indicator
    st.rerun()

# This function will be called after the rerun
def process_message(user_input):
    """Process the user message after showing the processing indicator."""
    debug_print(f"Now processing user message: {user_input}")
    
    # Check if graph is initialized
    if st.session_state.graph is None:
        debug_print("Graph is not initialized")
        # Remove the processing message
        st.session_state.messages = [msg for msg in st.session_state.messages if not msg.get('is_processing', False)]
        st.session_state.is_processing = False
        
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
        # Remove the processing message
        st.session_state.messages = [msg for msg in st.session_state.messages if not msg.get('is_processing', False)]
        
        events = []
        reached_end = False
        
        # Use a timeout to prevent infinite processing
        start_time = time.time()
        max_process_time = 60  # Maximum processing time in seconds
        
        for event in st.session_state.graph.stream(st.session_state.travel_state):
            events.append(event)
            
            # Check for timeout
            if time.time() - start_time > max_process_time:
                debug_print("Processing timeout reached, stopping graph execution")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I've been processing for too long and had to stop. Please try a more specific request or check your input."
                })
                break
                
            # Check if we've reached the end of the graph
            if event.get("type") == "end":
                debug_print("Reached end of graph execution")
                reached_end = True
                break
            
            # Display agent outputs
            if event.get("type") == "agent":
                agent_name = event.get("agent", "Assistant")
                debug_print(f"Processing event from agent: {agent_name}")
                state = event.get("state")
                if state and state.messages and len(state.messages) > 0:
                    last_message = state.messages[-1]
                    if last_message["role"] == "assistant":
                        # Add a visual indicator for which agent is responding
                        agent_label = ""
                        if agent_name == "supervisor":
                            agent_label = "🧠 "
                        elif agent_name == "flight_agent":
                            agent_label = "✈️ "
                        elif agent_name == "hotel_agent":
                            agent_label = "🏨 "
                        elif agent_name == "itinerary_agent":
                            agent_label = "📋 "
                        
                        message_content = f"{agent_label}{last_message['content']}"
                        st.session_state.messages.append({"role": "assistant", "content": message_content})
                        st.session_state.travel_state = state
                        
                        # Extract and structure travel information if it looks like a structured request
                        extract_travel_info(last_message["content"])
                        
                        # Force a rerun to display the message immediately
                        st.rerun()
        
        # If we didn't get any agent responses
        if not any(e.get("type") == "agent" for e in events) and not reached_end:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'm processing your request. Can you please provide more details about your travel plans?"
            })
        
        # If we reached the end, add a completion message
        if reached_end:
            debug_print("Conversation completed, adding completion indicator")
            # Check if the last message already indicates completion
            last_msg = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
            if not any(phrase in last_msg.lower() for phrase in ["anything else", "final itinerary", "thank you", "goodbye"]):
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "✅ I've completed processing your request. Is there anything else you'd like to know?"
                })
        
        # Clear processing state
        st.session_state.is_processing = False
        debug_print("Processing complete, cleared is_processing state")
        
    except Exception as e:
        error_msg = f"Error in agent processing: {str(e)}"
        debug_print(error_msg)
        # Clear processing state and indicator
        st.session_state.is_processing = False
        st.session_state.messages = [msg for msg in st.session_state.messages if not msg.get('is_processing', False)]
        
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
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Search for flights
        status_text.text("Searching for flights...")
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
        progress_bar.progress(33)
        
        # Search for hotels
        status_text.text("Finding the best accommodations...")
        debug_print("Starting hotel search")
        hotels = search_hotels_serp(
            location=info["destination"],
            check_in=info["departure_date"],
            check_out=info["return_date"],
            budget=info["budget"]
        )
        st.session_state.search_results["hotels"] = hotels
        debug_print(f"Hotel search returned {len(hotels)} results")
        progress_bar.progress(66)
        
        # Get destination information
        status_text.text("Gathering destination information...")
        debug_print("Starting destination info search")
        destination_info = get_destination_info_serp(info["destination"])
        st.session_state.search_results["destination_info"] = destination_info
        debug_print(f"Destination info received: {bool(destination_info)}")
        progress_bar.progress(100)
        
        # Clear the progress indicators
        status_text.empty()
        
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
    
    # Debug settings (only visible when DEBUG is True)
    if DEBUG and st.sidebar.checkbox("Show Debug Controls", value=False):
        st.sidebar.header("Debug Controls")
        
        # LLM Provider selection
        current_provider = os.getenv("LLM_PROVIDER", "openai")
        llm_provider = st.sidebar.radio(
            "LLM Provider",
            options=["openai", "ollama"],
            index=0 if current_provider == "openai" else 1
        )
        
        # Apply LLM provider change
        if llm_provider != current_provider:
            os.environ["LLM_PROVIDER"] = llm_provider
            if llm_provider == "openai":
                st.success("LLM provider updated to OpenAI. App will use OpenAI for all LLM calls.")
                st.rerun()
            else:
                st.success(f"LLM provider updated to Ollama ({os.getenv('OLLAMA_MODEL', 'llama3.2')}). App will use Ollama for all LLM calls.")
                st.rerun()
    
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

# Display processing status if active
if st.session_state.is_processing:
    st.info("🔄 The AI is processing your request. This may take a moment...", icon="ℹ️")

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        # Check if this is a processing indicator message
        is_processing = message.get('is_processing', False)
        content_class = "ai-message processing-message" if is_processing else "ai-message"
        st.markdown(f'<div class="{content_class}">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input and search button
col1, col2 = st.columns([5, 1])
with col1:
    # Add guidance text above the input field
    st.markdown("""
    <div style="background-color: #4b4b4b; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;">
        <strong>For best results, please include:</strong>
        <ul style="margin: 5px 0 5px 20px; padding: 0;">
            <li>Origin and destination cities</li>
            <li>Travel dates (YYYY-MM-DD format)</li>
            <li>Number of travelers</li>
            <li>Budget (if applicable)</li>
            <li>Special preferences (e.g., direct flights, hotel amenities)</li>
        </ul>
        <em>Example: "I want to travel from Mumbai to Delhi from 2025-06-15 to 2025-06-20 for 2 adults with a budget of 50,000 INR"</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Use a callback to process the input instead of modifying session state directly
    user_input = st.text_input("", placeholder="Tell me about your travel plans...", key="user_input_field")
with col2:
    if st.session_state.chat_stage == "ready_to_search":
        search_button = st.button("Search Now")
    else:
        search_button = False

# Check if we need to process a previous message
if 'pending_message' in st.session_state and st.session_state.pending_message:
    process_message(st.session_state.pending_message)
    st.session_state.pending_message = None
    
# Process user input
if user_input:
    # Store the message to be processed after the rerun
    st.session_state.pending_message = user_input
    handle_user_message(user_input)
    
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
        
        # Create a user-friendly flight comparison view
        flight_data = []
        for i, flight in enumerate(results["flights"]):
            flight_data.append({
                "Option": f"Flight {i+1}",
                "Airline": flight.get('airline', 'Unknown Airline'),
                "Price": f"{flight.get('price', 'Unknown')} {flight.get('currency', 'INR')}",
                "Departure": format_datetime(flight.get('departure_time', 'Unknown')),
                "Arrival": format_datetime(flight.get('arrival_time', 'Unknown')),
                "Duration": flight.get('duration', 'Unknown'),
                "id": i
            })
        
        # Create a DataFrame for better comparison
        flight_df = pd.DataFrame(flight_data)
        st.dataframe(flight_df.set_index("Option").drop("id", axis=1), use_container_width=True)
        
        # Display individual flight cards for selection
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
                        # Store actual flight data in session state
                        st.session_state.selected_flight_data = flight
    
    # Display hotels section
    if results["hotels"]:
        st.header("Accommodation Options")
        debug_print(f"Displaying {len(results['hotels'])} hotel options")
        
        # Create a user-friendly hotel comparison view
        hotel_data = []
        for i, hotel in enumerate(results["hotels"]):
            hotel_data.append({
                "Option": f"Hotel {i+1}",
                "Name": hotel.get('name', 'Unknown'),
                "Price/Night": f"{hotel.get('price_per_night', 'Unknown')} {hotel.get('currency', 'INR')}",
                "Rating": f"{hotel.get('rating', 'N/A')} ⭐",
                "Amenities": ', '.join(hotel.get('amenities', [])[:3]) if hotel.get('amenities') else 'N/A',
                "id": i
            })
        
        # Create a DataFrame for better comparison
        hotel_df = pd.DataFrame(hotel_data)
        st.dataframe(hotel_df.set_index("Option").drop("id", axis=1), use_container_width=True)
        
        # Display individual hotel cards for selection
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
                        try:
                            st.image(hotel['image_url'], use_container_width=True)
                        except Exception as e:
                            debug_print(f"Error loading hotel image: {str(e)}")
                    
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
                        # Store actual hotel data in session state
                        st.session_state.selected_hotel_data = hotel
    
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
        
        selected_flight = st.session_state.selected_flight_data or results["flights"][st.session_state.selected_flight]
        selected_hotel = st.session_state.selected_hotel_data or results["hotels"][st.session_state.selected_accommodation]
        
        # Create summary of selected options
        st.subheader("Your Selections")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Selected Flight**")
            st.markdown(f"""
            **Airline**: {selected_flight.get('airline', 'Unknown')}  
            **Departure**: {format_datetime(selected_flight.get('departure_time', 'Unknown'))}  
            **Arrival**: {format_datetime(selected_flight.get('arrival_time', 'Unknown'))}  
            **Duration**: {selected_flight.get('duration', 'Unknown')}  
            **Price**: {selected_flight.get('price', 'Unknown')} {selected_flight.get('currency', 'INR')}
            """)
        
        with col2:
            st.markdown("**Selected Accommodation**")
            st.markdown(f"""
            **Hotel**: {selected_hotel.get('name', 'Unknown')}  
            **Price/Night**: {selected_hotel.get('price_per_night', 'Unknown')} {selected_hotel.get('currency', 'INR')}  
            **Rating**: {selected_hotel.get('rating', 'N/A')} ⭐  
            **Amenities**: {', '.join(selected_hotel.get('amenities', [])[:3]) if selected_hotel.get('amenities') else 'N/A'}
            """)
        
        # Create sample activities from destination info
        destination_info = results["destination_info"]
        activities = destination_info.get('attractions', [])[:5] if destination_info else ["Sightseeing"]
        
        # Create a suggested itinerary based on the collected data
        st.subheader("Suggested Activities")
        for i, activity in enumerate(activities):
            st.markdown(f"**Day {i+1}:** {activity}")
        
        col7, col8 = st.columns(2)
        with col7:
            if st.button("Download PDF"):
                debug_print("PDF download button clicked")
                
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
                try:
                    # Generate the PDF
                    pdf = create_pdf_itinerary(
                        selected_flight,
                        selected_hotel,
                        activities
                    )
                    pdf_file = "temp_itinerary.pdf"
                    pdf.output(pdf_file)
                    
                    # Set up email
                    msg = MIMEMultipart()
                    msg['From'] = "travel.assistant@example.com"
                    msg['To'] = email
                    msg['Subject'] = "Your Travel Itinerary"
                    
                    # Email body
                    body = "Please find your travel itinerary attached. Thank you for using our service!"
                    msg.attach(MIMEText(body, 'plain'))
                    
                    # Attach PDF
                    with open(pdf_file, "rb") as f:
                        attach = MIMEApplication(f.read(), _subtype="pdf")
                    attach.add_header('Content-Disposition', 'attachment', filename="travel_itinerary.pdf")
                    msg.attach(attach)
                    
                    # Show success message (in a real app, you would send the email here)
                    st.success(f"Your itinerary has been sent to {email}!")
                    
                    # Clean up
                    if os.path.exists(pdf_file):
                        os.remove(pdf_file)
                    
                except Exception as e:
                    st.error(f"Error sending email: {str(e)}")
                    debug_print(f"Email error: {str(e)}")

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