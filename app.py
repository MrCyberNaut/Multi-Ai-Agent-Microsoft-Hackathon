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

# Sidebar form
with st.sidebar:
    st.title("✈️ Travel Planner")
    st.markdown("---")
    
    # User inputs
    prompt = st.text_area("What kind of trip are you dreaming of?",
                         placeholder="E.g., I want a relaxing beach vacation...")
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origin", placeholder="City or airport")
    with col2:
        destination = st.text_input("Destination", placeholder="City or airport")
    
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
    if departure_date and return_date:
        length_of_stay = (return_date - departure_date).days
        length_of_stay = st.number_input("Length of Stay (days)", 
                                       value=length_of_stay, min_value=1)
    
    col5, col6 = st.columns(2)
    with col5:
        budget = st.number_input("Budget ($)", min_value=0, step=100)
    with col6:
        travelers = st.number_input("Number of Travelers", min_value=1, value=1)
    
    search_button = st.button("Search")

# Main content area
if search_button:
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
        
        # Display results in cards
        st.header("Flight Options")
        flight_cols = st.columns(3)
        for i, flight in enumerate(flights):
            with flight_cols[i % 3]:
                with st.container():
                    st.markdown(f"""
                    <div class='travel-card {"selected-card" if st.session_state.selected_flight == i else ""}'>
                        <h3>{flight['airline']}</h3>
                        <p>Flight: {flight['flight_number']}</p>
                        <p>Price: {flight['price']}</p>
                        <p>Departure: {flight['departure']}</p>
                        <p>Arrival: {flight['arrival']}</p>
                        <p>Duration: {flight.get('duration', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Select Flight {i+1}", key=f"flight_{i}"):
                        st.session_state.selected_flight = i
        
        st.header("Accommodation Options")
        hotel_cols = st.columns(3)
        for i, hotel in enumerate(hotels):
            with hotel_cols[i % 3]:
                with st.container():
                    st.markdown(f"""
                    <div class='travel-card {"selected-card" if st.session_state.selected_accommodation == i else ""}'>
                        <h3>{hotel['name']}</h3>
                        <p>Price: {hotel['price']}</p>
                        <p>Rating: {hotel['rating']} ⭐</p>
                        <p>Address: {hotel['address']}</p>
                        <p>Amenities: {', '.join(hotel['amenities'][:3])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Select Hotel {i+1}", key=f"hotel_{i}"):
                        st.session_state.selected_accommodation = i
        
        # Display timeline if both flight and accommodation are selected
        if st.session_state.selected_flight is not None and st.session_state.selected_accommodation is not None:
            st.header("Your Itinerary Timeline")
            
            # Create sample activities from destination info
            activities = destination_info.get('attractions', [])[:5]
            
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