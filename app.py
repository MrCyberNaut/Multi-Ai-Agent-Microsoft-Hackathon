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
from booking_agent import (
    search_flights_serp,
    search_hotels_serp,
    get_destination_info_serp,
    get_human_selection
)

# Configure Streamlit page
st.set_page_config(
    page_title="Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize session state
if 'selected_flight' not in st.session_state:
    st.session_state.selected_flight = None
if 'selected_accommodation' not in st.session_state:
    st.session_state.selected_accommodation = None

# Debug section to show API keys status
with st.expander("🔍 Debug Information"):
    st.write("API Keys Status:")
    st.write({
        "SERPAPI_API_KEY": "✅ Set" if os.getenv("SERPAPI_API_KEY") else "❌ Not Set",
        "AIMLAPI_API_KEY": "✅ Set" if os.getenv("AIMLAPI_API_KEY") else "❌ Not Set"
    })
    
    if st.button("Test SerpAPI Connection"):
        from serpapi import GoogleSearch
        try:
            test_params = {
                "engine": "google_flights",
                "departure_id": "JFK",
                "arrival_id": "LAX",
                "outbound_date": "2025-07-01",
                "api_key": os.getenv("SERPAPI_API_KEY")
            }
            test_search = GoogleSearch(test_params)
            test_results = test_search.get_dict()
            st.success("✅ SerpAPI connection successful")
            st.json(test_results)
        except Exception as e:
            st.error(f"❌ SerpAPI connection failed: {str(e)}")

def create_pdf_itinerary(flight_data, accommodation_data, activities):
    """Create a PDF itinerary from selected options."""
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
    
    return pdf

def send_email(email_address, pdf_bytes, itinerary_data):
    """Send itinerary PDF via email (mock implementation)."""
    try:
        # In a real implementation, you would:
        # 1. Configure SMTP server
        # 2. Create and send the email with attachment
        # For now, we'll just show a success message
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def create_timeline(departure_date, return_date, activities):
    """Create a Plotly timeline visualization."""
    df = []
    
    # Add flight events
    df.append(dict(Task="Travel", Start=departure_date, Finish=departure_date, Resource="Flight"))
    df.append(dict(Task="Travel", Start=return_date, Finish=return_date, Resource="Flight"))
    
    # Add activities
    current_date = departure_date
    while current_date <= return_date:
        for activity in activities:
            df.append(dict(
                Task="Activity",
                Start=current_date,
                Finish=current_date + timedelta(hours=2),
                Resource=activity
            ))
        current_date += timedelta(days=1)
    
    # Create the timeline
    fig = ff.create_gantt(
        df,
        colors={
            'Flight': '#ff6b35',
            'Activity': '#4b4b4b'
        },
        index_col='Resource',
        show_colorbar=True,
        group_tasks=True,
        showgrid_x=True,
        showgrid_y=True
    )
    
    # Update layout for dark theme
    fig.update_layout(
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#2b2b2b',
        font_color='#ffffff'
    )
    
    return fig

def format_datetime(iso_string):
    """Format datetime string for display."""
    try:
        # Handle different datetime formats
        if "T" in iso_string:
            dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(iso_string, "%Y-%m-%d %H:%M")
        return dt.strftime("%b %d, %Y | %I:%M %p")  # Example: Mar 06, 2025 | 6:20 PM
    except Exception as e:
        return iso_string  # Return original if parsing fails

# Sidebar form
with st.sidebar:
    st.title("✈️ Travel Planner")
    st.markdown("---")
    
    # User inputs
    prompt = st.text_area("What kind of trip are you dreaming of?",
                         placeholder="E.g., I want a relaxing beach vacation...")
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origin", placeholder="City or airport code")
    with col2:
        destination = st.text_input("Destination", placeholder="City or airport code")
    
    col3, col4 = st.columns(2)
    with col3:
        departure_date = st.date_input("Departure")
    with col4:
        return_date = st.date_input("Return")
    
    travel_mode = st.selectbox(
        "Mode of Travel",
        ["flight", "train", "bus", "mixed"]
    )
    
    accommodation_type = st.selectbox(
        "Accommodation Type",
        ["hotel", "hostel", "apartment"]
    )
    
    # Auto-calculate length of stay
    default_stay = (return_date - departure_date).days
    if default_stay < 0:
        default_stay = 1
        st.warning("Return date must be after departure date.")
    
    # Set length of stay input based on dates
    if departure_date and return_date:
        # Ensure that length_of_stay is at least 1 day
        length_of_stay = max(default_stay, 1)  # This makes sure length_of_stay is never below 1
        length_of_stay = st.number_input("Length of Stay (days)", 
                                        value=length_of_stay, min_value=1)

    if (return_date - departure_date).days <= 0:
        st.warning("Return date must be after departure date.")

    
    col5, col6 = st.columns(2)
    with col5:
        budget = st.number_input("Budget ($)", min_value=0, step=100)
    with col6:
        travelers = st.number_input("Number of Travelers", min_value=1, value=1)
    
    search_button = st.button("Search")

# Main content area
if search_button:
    if not origin or not destination or not departure_date or not return_date:
        st.error("Please fill in all required fields: origin, destination, departure date, and return date.")
    else:
        try:
            with st.spinner("Searching for the best options..."):
                # Search for flights
                flights = search_flights_serp(
                    origin=origin,
                    destination=destination,
                    departure_date=departure_date.strftime("%Y-%m-%d"),
                    return_date=return_date.strftime("%Y-%m-%d"),
                    budget=str(budget) if budget else None
                )
                
                # Search for accommodations
                hotels = search_hotels_serp(
                    location=destination,
                    check_in=departure_date.strftime("%Y-%m-%d"),
                    check_out=return_date.strftime("%Y-%m-%d"),
                    budget=str(budget) if budget else None
                )
                
                # Get destination information
                destination_info = get_destination_info_serp(destination)
            
            # Display flights section
            st.header("Flight Options")
            if flights:
                flight_cols = st.columns(min(3, len(flights)))
                for i, flight in enumerate(flights):
                    with flight_cols[i % 3]:
                        with st.container():
                            airline = flight.get('airline', 'Unknown Airline')
                            price = flight.get('price', 'Unknown')
                            departure = format_datetime(flight.get('departure', 'Unknown'))
                            arrival = format_datetime(flight.get('arrival', 'Unknown'))
                            duration = flight.get('duration', 'Unknown')
                            
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
                                st.session_state.selected_flight = i
            else:
                st.warning("No flight options found. Please check your search parameters or try a different route.")
                # Show cache content for debugging
                if os.path.exists(f"cache/debug_flights_{origin}_{destination}_{departure_date.strftime('%Y-%m-%d')}.json"):
                    with st.expander("View API Debug Information"):
                        with open(f"cache/debug_flights_{origin}_{destination}_{departure_date.strftime('%Y-%m-%d')}.json") as f:
                            st.json(json.load(f))
            
            # Display hotels section
            st.header("Accommodation Options")
            if hotels:
                hotel_cols = st.columns(min(3, len(hotels)))
                for i, hotel in enumerate(hotels):
                    with hotel_cols[i % 3]:
                        with st.container():
                            name = hotel.get('name', 'Unknown')
                            price = hotel.get('price', 'Unknown')
                            rating = hotel.get('rating', 'N/A')
                            address = hotel.get('address', 'Unknown')
                            amenities = ', '.join(hotel.get('amenities', [])[:3]) if hotel.get('amenities') else 'N/A'
                            
                            st.markdown(f"""
                            <div class='travel-card {"selected-card" if st.session_state.selected_accommodation == i else ""}'>
                                <h3>{name}</h3>
                                <p><strong>Price:</strong> {price}</p>
                                <p><strong>Rating:</strong> {rating} ⭐</p>
                                <p><strong>Address:</strong> {address}</p>
                                <p><strong>Amenities:</strong> {amenities}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"Select Hotel {i+1}", key=f"hotel_{i}"):
                                st.session_state.selected_accommodation = i
            else:
                st.warning("No accommodation options found. Please check your search parameters or try a different location.")
                # Show cache content for debugging
                if os.path.exists(f"cache/debug_hotels_{destination}_{departure_date.strftime('%Y-%m-%d')}_{return_date.strftime('%Y-%m-%d')}.json"):
                    with st.expander("View API Debug Information"):
                        with open(f"cache/debug_hotels_{destination}_{departure_date.strftime('%Y-%m-%d')}_{return_date.strftime('%Y-%m-%d')}.json") as f:
                            st.json(json.load(f))
            
            # Display destination information
            st.header("Destination Information")
            if destination_info:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader(destination_info.get('name', destination))
                    st.write(destination_info.get('description', 'No description available'))
                    
                    if destination_info.get('attractions'):
                        st.subheader("Top Attractions")
                        for attraction in destination_info.get('attractions'):
                            st.write(f"• {attraction}")
                
                with col2:
                    if destination_info.get('local_tips'):
                        st.subheader("Local Tips")
                        for tip in destination_info.get('local_tips'):
                            st.write(f"• {tip}")
            else:
                st.info(f"No detailed information available for {destination}")
        
            # Display timeline if both flight and accommodation are selected
            if st.session_state.selected_flight is not None and st.session_state.selected_accommodation is not None:
                st.header("Your Itinerary Timeline")
                
                # Create sample activities from destination info
                activities = destination_info.get('attractions', [])[:5] if destination_info else ["Sightseeing"]
                
                # Create and display timeline
                timeline = create_timeline(departure_date, return_date, activities)
                st.plotly_chart(timeline, use_container_width=True)
                
                # Export options
                col7, col8 = st.columns(2)
                with col7:
                    if st.button("Download PDF"):
                        selected_flight = flights[st.session_state.selected_flight]
                        selected_hotel = hotels[st.session_state.selected_accommodation]
                        
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
                    if email and st.button("Email Itinerary"):
                        selected_flight = flights[st.session_state.selected_flight]
                        selected_hotel = hotels[st.session_state.selected_accommodation]
                        
                        pdf = create_pdf_itinerary(
                            selected_flight,
                            selected_hotel,
                            activities
                        )
                        
                        if send_email(email, pdf.output(dest='S').encode('latin-1'), {
                            'flight': selected_flight,
                            'hotel': selected_hotel,
                            'activities': activities
                        }):
                            st.success("Itinerary sent successfully!")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Check the debug section to make sure your API keys are properly set up.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>Travel Planner v1.0 | Built with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
) 