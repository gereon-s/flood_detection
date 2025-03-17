import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import aiohttp
import re
import os
import json
import folium
from streamlit_folium import st_folium
import time
import openai
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime, timedelta
from aio_georss_gdacs import GdacsFeed

# Set page configuration
st.set_page_config(
    page_title="Environmental Catastrophe Detection", page_icon="🌍", layout="wide"
)

# Custom CSS - with navbar additions
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
    }
    .alert-high {
        background-color: rgba(244, 67, 54, 0.15);
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #F44336;
        color: #FFFFFF;
        font-weight: 500;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .alert-high h3 {
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #FFCDD2;
        font-weight: 600;
    }
    .alert-high p {
        margin: 0.4rem 0;
        line-height: 1.4;
        font-size: 0.95rem;
    }
    .alert-high strong {
        color: #FFCDD2;
    }
    .alert-medium {
        background-color: rgba(255, 193, 7, 0.15);
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FFC107;
        color: #FFFFFF;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .alert-medium h3 {
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #FFF9C4;
        font-weight: 600;
    }
    .alert-medium p {
        margin: 0.4rem 0;
        line-height: 1.4;
        font-size: 0.95rem;
    }
    .alert-low {
        background-color: rgba(76, 175, 80, 0.15);
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
        color: #FFFFFF;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .alert-low h3 {
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #C8E6C9;
        font-weight: 600;
    }
    .metric-card {
        background-color: #1E4620;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-weight: 400;
        color: #AAFFAA;
    }
    .metric-card h2 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
        color: white;
    }
    .stApp {
        background-color: #121212;
    }
    .css-18e3th9, .css-1d391kg {
        background-color: #121212;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: white;
    }
    .css-1544g2n {
        padding: 1rem;
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .refresh-btn {
        background-color: #388E3C;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
        cursor: pointer;
        text-align: center;
    }
    .refresh-btn:hover {
        background-color: #2E7D32;
    }
    .last-updated {
        font-size: 0.8rem;
        color: #AAAAAA;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Improved chat styling */
    .chatbox {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        min-height: 100px;  /* smaller minimum height */
        max-height: 400px;  /* keep maximum height */
        height: auto;       /* auto height based on content */
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        border: 1px solid #333;
    }
    .user-message {
        background-color: #388E3C;
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 1rem 1rem 0 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
        align-self: flex-end;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .bot-message {
        background-color: #333333;
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 1rem 1rem 1rem 0;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-right: auto;
        align-self: flex-start;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .chat-input {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .main-column-left {
        padding-right: 1rem;
    }
    .main-column-right {
        padding-left: 1rem;
        display: flex;
        flex-direction: column;
    }
    .stTextInput > div > div > input {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #444;
    }
    .assistant-section {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .chat-container {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .clear-button {
        background-color: #444;
        color: white;
        border: none;
        padding: 0.3rem 0.7rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        cursor: pointer;
    }
    .clear-button:hover {
        background-color: #555;
    }
    /* Make empty chatbox smaller */
    .chatbox:empty {
        height: 0;
        min-height: 0;
        padding: 0;
        margin: 0;
        border: none;
    }
    
    /* Navigation Bar Styling */
    .navbar {
        display: flex;
        justify-content: center;
        padding: 0.5rem;
        background-color: #1E1E1E;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .nav-item {
        background-color: #333;
        color: white;
        padding: 0.7rem 1.5rem;
        margin: 0 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
        font-weight: 500;
        text-decoration: none;
    }
    .nav-item:hover {
        background-color: #444;
        transform: translateY(-2px);
    }
    .nav-item-active {
        background-color: #388E3C;
        color: white;
        padding: 0.7rem 1.5rem;
        margin: 0 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: 600;
    }
    .info-box {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .date-filter {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .pagination {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }
    .pagination-button {
        background-color: #333;
        color: white;
        border: none;
        padding: 0.3rem 0.7rem;
        margin: 0 0.2rem;
        border-radius: 0.3rem;
        cursor: pointer;
    }
    .pagination-button:hover {
        background-color: #444;
    }
    .pagination-active {
        background-color: #388E3C;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "gdacs"


def navigation():
    # Create columns for the navigation items
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "GDACS Alerts",
            key="nav_gdacs",
            help="View real-time GDACS alerts",
            use_container_width=True,
            type="primary" if st.session_state.page == "gdacs" else "secondary",
        ):
            st.session_state.page = "gdacs"
            st.rerun()

    with col2:
        if st.button(
            "Flood Predictions",
            key="nav_flood",
            help="View AI-based flood predictions",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == "flood_predictions"
                else "secondary"
            ),
        ):
            st.session_state.page = "flood_predictions"
            st.rerun()


# Initialize OpenAI API if available
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def initialize_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # For development/testing, you can set the API key in the Streamlit secrets
        if "openai_api_key" in st.secrets:
            api_key = st.secrets["openai_api_key"]

    if api_key:
        openai.api_key = api_key
        return True
    return False


# Helper function to extract numeric value from various formats
def extract_numeric_value(value, default=50.0):
    """Extract numeric value from a string, dict, or return the value if already numeric."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, dict) and "value" in value:
        try:
            return float(value["value"])
        except (ValueError, TypeError):
            return default
    elif isinstance(value, str):
        # Try to extract numeric part from strings like "Magnitude 5.8M, Depth:10km"
        match = re.search(r"(\d+\.?\d*)", value)
        if match:
            return float(match.group(1))
    return default


# Function to fetch GDACS data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_gdacs_data():
    # Run the async function to fetch data
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    entries = loop.run_until_complete(fetch_gdacs_alerts())
    loop.close()

    # Process the data for our app
    return process_gdacs_entries(entries)


async def fetch_gdacs_alerts():
    """Fetch only ongoing GDACS alerts using aio-georss-gdacs library."""
    async with aiohttp.ClientSession() as session:
        # Use a neutral point (0, 0) for coordinates
        feed = GdacsFeed(session, (0, 0))

        status, entries = await feed.update()

        if status == "OK":
            # Only include entries that are currently active
            # Define stricter rules for current events
            current_entries = []
            for entry in entries:
                # Check if explicitly marked as current
                is_current = getattr(entry, "is_current", None)
                # Check for "to_date" being in the future or not set
                to_date = getattr(entry, "to_date", None)
                is_ongoing = True

                if to_date:
                    try:
                        end_date = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
                        if end_date < datetime.now():
                            is_ongoing = False
                    except (ValueError, TypeError):
                        # If date parsing fails, assume it's ongoing
                        pass

                # Only include if it's marked as current or has no end date or end date is in future
                if (is_current is None or is_current) and is_ongoing:
                    current_entries.append(entry)

            return current_entries
        else:
            st.error(f"Error fetching GDACS alerts: {status}")
            return []


def process_gdacs_entries(entries):
    """Process GDACS entries into a pandas DataFrame."""
    data = []

    for entry in entries:
        # Map alert level to severity
        if entry.alert_level == "Red":
            severity = "High"
        elif entry.alert_level == "Orange":
            severity = "Medium"
        else:  # Green or unknown
            severity = "Low"

        # Map event type to our categories
        event_type_map = {
            "EQ": "Earthquake",
            "TC": "Tropical Cyclone",
            "FL": "Flooding",
            "VO": "Volcano",
            "DR": "Drought",
            "WF": "Forest Fire",
            "TS": "Tsunami",
        }

        event_type = event_type_map.get(
            entry.event_type_short, entry.event_type or "Other"
        )

        # Extract coordinates
        lat, lon = None, None
        if entry.coordinates:
            lat, lon = entry.coordinates[0], entry.coordinates[1]

        # Extract affected area - handle different formats
        affected_area = 10.0  # Default value
        if hasattr(entry, "affected_area_km2"):
            affected_area = extract_numeric_value(entry.affected_area_km2, 10.0)

        # Extract risk score or severity as a numeric value
        risk_score = extract_numeric_value(getattr(entry, "severity", 50.0))

        # Create a row for our DataFrame
        data.append(
            {
                "id": entry.external_id,
                "event_type": event_type,
                "latitude": lat,
                "longitude": lon,
                "severity": severity,
                "confidence": extract_numeric_value(
                    getattr(entry, "confidence", 0.9), 0.9
                ),
                "detected_at": entry.from_date
                or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "affected_area_km2": affected_area,
                "status": "Active",  # All events are active since we filtered for current ones
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "detection_method": "Satellite Imagery",
                "risk_score": risk_score,
                "description": entry.description or "",
                "country": entry.country or "Unknown",
                "alert_level": entry.alert_level or "Green",
                "original_entry": entry,  # Keep the original entry for reference
            }
        )

    # Create DataFrame
    if data:
        df = pd.DataFrame(data)

        # Assign colors based on severity for the map
        color_map = {
            "High": [255, 0, 0, 160],
            "Medium": [255, 165, 0, 140],
            "Low": [0, 255, 0, 120],
        }
        df["color"] = df["severity"].map(color_map)

        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "id",
                "event_type",
                "latitude",
                "longitude",
                "severity",
                "confidence",
                "detected_at",
                "affected_area_km2",
                "status",
                "last_updated",
                "detection_method",
                "risk_score",
                "description",
                "country",
                "alert_level",
                "color",
            ]
        )


def create_folium_map(df):
    """Create a Folium map with the alerts."""
    # Create a map centered on a neutral location if no data
    if df.empty or not any(pd.notna(df["latitude"]) & pd.notna(df["longitude"])):
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        folium.TileLayer("cartodbdark_matter").add_to(m)
        return m

    # Filter for entries with valid coordinates
    valid_df = df[pd.notna(df["latitude"]) & pd.notna(df["longitude"])]

    if valid_df.empty:
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        folium.TileLayer("cartodbdark_matter").add_to(m)
        return m

    # Create a map centered on data
    m = folium.Map(
        location=[valid_df["latitude"].mean(), valid_df["longitude"].mean()],
        zoom_start=2,
        control_scale=True,
    )

    # Add a dark mode tile layer
    folium.TileLayer("cartodbdark_matter").add_to(m)

    # Create feature groups for each event type
    event_types = valid_df["event_type"].unique()
    feature_groups = {
        event_type: folium.FeatureGroup(name=event_type) for event_type in event_types
    }

    # Add markers to appropriate feature groups
    for _, row in valid_df.iterrows():
        # Determine marker color based on alert level
        if row["alert_level"] == "Red":
            color = "red"
        elif row["alert_level"] == "Orange":
            color = "orange"
        else:  # Green or unknown
            color = "green"

        # Determine icon based on event type
        icon_map = {
            "Earthquake": "bolt",
            "Tropical Cyclone": "cloud",
            "Flooding": "tint",
            "Volcano": "fire",
            "Drought": "sun",
            "Forest Fire": "fire",
            "Tsunami": "water",
        }
        icon = icon_map.get(row["event_type"], "info-sign")

        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4 style="color: #333;">{row['event_type']} - {row['country']}</h4>
            <strong>Alert Level:</strong> {row['alert_level']}<br>
            <strong>Status:</strong> {row['status']}<br>
            <strong>Detected:</strong> {row['detected_at']}<br>
            <strong>Description:</strong> {row['description'][:150]}...
        </div>
        """

        # Add marker to map
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            tooltip=f"{row['event_type']} ({row['alert_level']})",
        ).add_to(feature_groups[row["event_type"]])

    # Add feature groups to map
    for group in feature_groups.values():
        group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


# Enhanced location map function to include nearby disasters
def create_user_location_map(lat, lon, location_name, nearby_disasters=None):
    """Create a focused map for the user's location with nearby disasters."""
    if lat is None or lon is None:
        return None

    # Create map centered on user location
    m = folium.Map(
        location=[lat, lon],
        zoom_start=4,  # Adjusted zoom for regions/countries
        control_scale=True,
    )

    # Add a dark mode tile layer
    folium.TileLayer("cartodbdark_matter").add_to(m)

    # Add a marker for the user's location with a popup
    folium.Marker(
        location=[lat, lon],
        popup=f"Your location: {location_name}",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        tooltip=f"Your location: {location_name}",
    ).add_to(m)

    # Add a circle to show the search radius (500km)
    folium.Circle(
        location=[lat, lon],
        radius=500000,  # 500km in meters
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.1,
        popup="Search radius: 500km",
    ).add_to(m)

    # Add nearby disasters to the map if available
    if nearby_disasters is not None and not nearby_disasters.empty:
        # Create a feature group for nearby disasters
        disaster_group = folium.FeatureGroup(name="Nearby Disasters")

        for _, disaster in nearby_disasters.iterrows():
            if pd.notna(disaster["latitude"]) and pd.notna(disaster["longitude"]):
                # Determine marker color based on alert level
                if disaster["alert_level"] == "Red":
                    color = "red"
                elif disaster["alert_level"] == "Orange":
                    color = "orange"
                else:  # Green or unknown
                    color = "green"

                # Determine icon based on event type
                icon_map = {
                    "Earthquake": "bolt",
                    "Tropical Cyclone": "cloud",
                    "Flooding": "tint",
                    "Volcano": "fire",
                    "Drought": "sun",
                    "Forest Fire": "fire",
                    "Tsunami": "water",
                }
                icon = icon_map.get(disaster["event_type"], "info-sign")

                # Create popup content
                popup_html = f"""
                <div style="font-family: Arial; max-width: 300px;">
                    <h4 style="color: #333;">{disaster['event_type']} - {disaster['country']}</h4>
                    <strong>Alert Level:</strong> {disaster['alert_level']}<br>
                    <strong>Status:</strong> {disaster['status']}<br>
                    <strong>Detected:</strong> {disaster['detected_at']}<br>
                    <strong>Distance:</strong> {disaster['distance_km']:.1f} km from your location<br>
                    <strong>Description:</strong> {disaster['description'][:150]}...
                </div>
                """

                # Add marker to map
                folium.Marker(
                    location=[disaster["latitude"], disaster["longitude"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                    tooltip=f"{disaster['event_type']} ({disaster['alert_level']})",
                ).add_to(disaster_group)

        # Add disaster group to map
        disaster_group.add_to(m)

    return m


# Validate if the input is a recognized country or location
def validate_country(location_name):
    """Validate if the input is a recognized country name or location."""
    try:
        # Use a list of common country names, regions, and ISO codes
        common_locations = {
            # Countries
            "usa": "United States",
            "us": "United States",
            "united states": "United States",
            "america": "United States",
            "uk": "United Kingdom",
            "england": "United Kingdom",
            "britain": "United Kingdom",
            "canada": "Canada",
            "australia": "Australia",
            "india": "India",
            "china": "China",
            "japan": "Japan",
            "germany": "Germany",
            "france": "France",
            "italy": "Italy",
            "spain": "Spain",
            "brazil": "Brazil",
            "mexico": "Mexico",
            "russia": "Russia",
            # Regions
            "europe": "Europe",
            "north america": "North America",
            "south america": "South America",
            "asia": "Asia",
            "africa": "Africa",
            "australia": "Australia",  # Both country and continent
            "oceania": "Oceania",
            # US Regions
            "west coast": "West Coast, United States",
            "east coast": "East Coast, United States",
            "midwest": "Midwest, United States",
            "pacific northwest": "Pacific Northwest, United States",
            "southeast": "Southeast United States",
            "northeast": "Northeast United States",
            "southwest": "Southwest United States",
            # European Regions
            "western europe": "Western Europe",
            "eastern europe": "Eastern Europe",
            "northern europe": "Northern Europe",
            "southern europe": "Southern Europe",
            "scandinavia": "Scandinavia",
            # Add more as needed
        }

        # Check if input is in our common locations list (case insensitive)
        normalized_input = location_name.lower().strip()
        if normalized_input in common_locations:
            return common_locations[normalized_input]

        # If not found in simple lookup, return the original
        return location_name

    except Exception as e:
        return location_name  # Return original if any error occurs


# Geocoding function to get coordinates from location name with improved error handling
def get_coordinates(location_name):
    """Get coordinates from a location name, handling broad regions and countries."""
    try:
        # Process input for countries first
        processed_location = validate_country(location_name)

        # Special case handling for broad regions that might not geocode well
        region_coordinates = {
            "Europe": (48.8566, 19.3522),  # Somewhere central in Europe
            "North America": (39.8283, -98.5795),  # Central US as proxy
            "South America": (-23.5505, -58.4371),  # Somewhere in center of S. America
            "Asia": (34.0479, 100.6197),  # Central Asia
            "Africa": (8.7832, 25.5085),  # Central Africa
            "Australia": (-25.2744, 133.7751),  # Center of Australia
            "Oceania": (-8.7832, 143.4317),  # Central Pacific
            "West Coast, United States": (37.7749, -122.4194),  # San Francisco
            "East Coast, United States": (40.7128, -74.0060),  # New York
            "Midwest, United States": (41.8781, -87.6298),  # Chicago
            # Add more region coordinates as needed
        }

        # Check if processed location is a known broad region
        if processed_location in region_coordinates:
            return region_coordinates[processed_location]

        # For more specific locations, use geocoding
        geolocator = Nominatim(user_agent="gdacs_app")
        location = geolocator.geocode(processed_location, timeout=10)

        if location:
            return location.latitude, location.longitude
        else:
            # Try with broader search
            broader_location = geolocator.geocode(
                processed_location, exactly_one=False, limit=5
            )
            if broader_location and len(broader_location) > 0:
                # Use the first result
                return broader_location[0].latitude, broader_location[0].longitude

            # If still not found, check if it's a common misspelling or alternate name
            print(f"Location '{location_name}' could not be found.")
            return None, None
    except Exception as e:
        print(f"Error geocoding location: {str(e)}")
        return None, None


# Find disasters near a location (continuation)
def find_disasters_near_location(df, lat, lon, radius_km=500):
    if lat is None or lon is None:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Create a new DataFrame with distance column
    near_df = df.copy()

    # Calculate distance to the specified location
    distances = []
    for idx, row in near_df.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            distance = geodesic(
                (lat, lon), (row["latitude"], row["longitude"])
            ).kilometers
            distances.append(distance)
        else:
            distances.append(float("inf"))  # For entries with no coordinates

    near_df["distance_km"] = distances

    # Filter by radius - use a larger radius for broad regions like "Europe" or "USA"
    if lat in [48.8566, 39.8283, -23.5505, 34.0479, 8.7832, -25.2744, -8.7832]:
        # This is a broad region, use larger radius
        result = near_df[near_df["distance_km"] <= 1000].sort_values("distance_km")
    else:
        # Normal location, use standard radius
        result = near_df[near_df["distance_km"] <= radius_km].sort_values("distance_km")

    return result


# AI Chat function with improved error handling
def get_ai_response(user_query, user_location, nearby_disasters):
    has_openai = initialize_openai()

    if not has_openai:
        return "Sorry, AI features are currently unavailable. Please set your OpenAI API key in the environment variables or Streamlit secrets."

    try:
        # Format disaster information for the AI
        disasters_info = ""
        if not nearby_disasters.empty:
            disasters_info = "Here are the disasters near the user's location:\n"
            for _, disaster in nearby_disasters.iterrows():
                disasters_info += f"- {disaster['event_type']} in {disaster['country']}, {disaster['distance_km']:.0f}km away. Alert level: {disaster['alert_level']}. {disaster['description'][:100]}...\n"
        else:
            disasters_info = "There are no disasters detected near the user's location."

        # Create the prompt for OpenAI
        prompt = f"""You are a disaster information assistant. The user is asking about disasters near {user_location}.

User query: {user_query}

{disasters_info}

Please provide a helpful, conversational response addressing the user's query based on this GDACS disaster data. 
Keep your response brief and focused on the disaster information relevant to their location.
If there are no disasters nearby, reassure them but mention that they should still stay informed about global events.
"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful disaster information assistant based on GDACS data.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        return f"Sorry, I encountered an error while processing your question. Please try again later."


# FLOOD PREDICTION PAGE FUNCTIONS


# Optimized flood data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_flood_data():
    """Load flood prediction results from JSON file with improved performance."""
    try:
        with open("flood_detection_results.json", "r") as f:
            # Read just the first 500 items instead of the entire file
            # This is a simple approach - for production, you might want a more sophisticated solution
            data = json.load(f)[:1000]  # Limit to 500 items

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Ensure date is in datetime format for filtering
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df
    except Exception as e:
        st.error(f"Error loading flood detection data: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "location_id",
                "location_folder",
                "date",
                "filename",
                "latitude",
                "longitude",
                "true_label",
                "predicted_label",
                "confidence",
                "coordinates",
                "environmental_impact",
                "economic_impact",
                "social_impact",
                "affected_area_km2",
                "total_impact",
                "status",
                "risk_score",
            ]
        )


def create_flood_map(df):
    """Create a Folium map with flood prediction markers."""
    # Create a map centered on the mean of the data points
    if df.empty or not any(pd.notna(df["latitude"]) & pd.notna(df["longitude"])):
        # Default to a neutral location if no data
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        folium.TileLayer("cartodbdark_matter").add_to(m)
        return m

    # Filter for entries with valid coordinates
    valid_df = df[pd.notna(df["latitude"]) & pd.notna(df["longitude"])].copy()

    if valid_df.empty:
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
        folium.TileLayer("cartodbdark_matter").add_to(m)
        return m

    # Create a map centered on data
    m = folium.Map(
        location=[valid_df["latitude"].mean(), valid_df["longitude"].mean()],
        zoom_start=3,
        control_scale=True,
    )

    # Add a dark mode tile layer
    folium.TileLayer("cartodbdark_matter").add_to(m)

    # Create feature groups for active floods and non-floods
    active_floods = folium.FeatureGroup(name="Active Floods")
    no_floods = folium.FeatureGroup(name="No Flood Detected")

    # Add markers to appropriate feature groups
    for _, row in valid_df.iterrows():
        # Determine marker color and icon based on prediction
        if row["predicted_label"] == 1:  # Active flood
            color = "blue"
            icon = "tint"
            feature_group = active_floods
        else:  # No flood
            color = "green"
            icon = "check"
            feature_group = no_floods

        # Scale marker size based on confidence/risk score
        radius = 6 + (row["confidence"] * 6) if "confidence" in row else 8

        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4 style="color: #333;">Flood Prediction - {row.get('location_id', 'Unknown')}</h4>
            <strong>Status:</strong> {row.get('status', 'Unknown')}<br>
            <strong>Date:</strong> {row.get('date').strftime('%Y-%m-%d') if isinstance(row.get('date'), pd.Timestamp) else row.get('date', 'Unknown')}<br>
            <strong>Confidence:</strong> {row.get('confidence', 0)*100:.1f}%<br>
            <strong>Risk Score:</strong> {row.get('risk_score', 0):.1f}<br>
            <strong>Affected Area:</strong> {row.get('affected_area_km2', 0):.2f} km²<br>
            <strong>Total Impact:</strong> {row.get('total_impact', 0):.2f}/10<br>
        </div>
        """

        # Plot the flood polygon if coordinates are available
        if "coordinates" in row and row["coordinates"] and len(row["coordinates"]) > 2:
            # Convert coordinates list to the format folium expects
            coords = [[coord[1], coord[0]] for coord in row["coordinates"]]
            # Get fill color based on risk score
            fill_color = get_color_from_risk_score(row.get("risk_score", 0))

            folium.Polygon(
                locations=coords,
                popup=folium.Popup(popup_html, max_width=300),
                color="#3186cc",
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.4,
                weight=2,
                tooltip=f"Flood Prediction ({row.get('confidence', 0)*100:.0f}% confidence)",
            ).add_to(feature_group)

        # Add marker at the centroid
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"Flood Prediction ({row.get('confidence', 0)*100:.0f}% confidence)",
        ).add_to(feature_group)

    # Add feature groups to map
    active_floods.add_to(m)
    no_floods.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def get_color_from_risk_score(risk_score):
    """Return a color based on the risk score."""
    if risk_score >= 75:
        return "#ff0000"  # Red
    elif risk_score >= 50:
        return "#ff9900"  # Orange
    elif risk_score >= 25:
        return "#ffcc00"  # Yellow
    else:
        return "#00cc00"  # Green


def create_time_series_chart(df):
    """Create a time series chart of flood events."""
    if df.empty or "date" not in df.columns:
        return None

    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Group by date and count flood events
    flood_counts = (
        df[df["predicted_label"] == 1]
        .groupby(df["date"].dt.to_period("M"))
        .size()
        .reset_index()
    )
    flood_counts.columns = ["month", "count"]
    flood_counts["month"] = flood_counts["month"].dt.to_timestamp()

    # Group by date and average confidence
    flood_confidence = (
        df[df["predicted_label"] == 1]
        .groupby(df["date"].dt.to_period("M"))["confidence"]
        .mean()
        .reset_index()
    )
    flood_confidence.columns = ["month", "confidence"]
    flood_confidence["month"] = flood_confidence["month"].dt.to_timestamp()

    # Create figure with two y-axes
    fig = go.Figure()

    # Add count line
    fig.add_trace(
        go.Scatter(
            x=flood_counts["month"],
            y=flood_counts["count"],
            name="Flood Events",
            line=dict(color="#2E7D32", width=3),
            mode="lines+markers",
        )
    )

    # Add confidence line (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=flood_confidence["month"],
            y=flood_confidence["confidence"],
            name="Avg. Confidence",
            line=dict(color="#1976D2", width=3, dash="dot"),
            mode="lines+markers",
            yaxis="y2",
        )
    )

    # Update layout - FIXED property names
    fig.update_layout(
        title="Flood Events Over Time",
        xaxis=dict(
            title="Month",
            tickfont=dict(color="white"),  # Use tickfont instead of titlefont
            gridcolor="#333333",
            type="date",  # Explicitly set the axis type
        ),
        yaxis=dict(
            title="Number of Flood Events",
            titlefont=dict(color="#2E7D32"),  # This is correct
            tickfont=dict(color="white"),  # Use tickfont for the ticks
            gridcolor="#333333",
        ),
        yaxis2=dict(
            title="Average Confidence",
            titlefont=dict(color="#1976D2"),  # This is correct
            tickfont=dict(color="white"),  # Use tickfont for the ticks
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        legend=dict(font=dict(color="white"), bgcolor="#1E1E1E", bordercolor="#333333"),
        font=dict(color="white"),
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        height=400,
    )

    return fig


# Fix for the impact heatmap
def create_impact_heatmap(df):
    """Create a heatmap of the environmental, economic, and social impacts with fixed configuration."""
    if df.empty:
        return None

    # Filter active floods and select relevant columns
    active_floods = (
        df[df["predicted_label"] == 1]
        .sort_values("total_impact", ascending=False)
        .head(10)
    )

    if active_floods.empty:
        return None

    # Prepare data for heatmap
    heatmap_df = active_floods[
        ["location_id", "environmental_impact", "economic_impact", "social_impact"]
    ].copy()

    # Melt the dataframe to get it in the right format for Plotly
    melted_df = pd.melt(
        heatmap_df,
        id_vars=["location_id"],
        value_vars=["environmental_impact", "economic_impact", "social_impact"],
        var_name="Impact Type",
        value_name="Impact Score",
    )

    # Make labels more readable
    melted_df["Impact Type"] = melted_df["Impact Type"].map(
        {
            "environmental_impact": "Environmental",
            "economic_impact": "Economic",
            "social_impact": "Social",
        }
    )

    # Create heatmap with explicit configuration
    fig = px.imshow(
        melted_df.pivot(
            index="location_id", columns="Impact Type", values="Impact Score"
        ),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        labels=dict(x="Impact Type", y="Location", color="Impact Score"),
        title="Impact Analysis by Location",
    )

    # Update layout with explicit configuration
    fig.update_layout(
        height=400,
        coloraxis_colorbar=dict(title="Impact Score (0-10)"),
        xaxis_title="Impact Category",
        yaxis_title="Location ID",
        font=dict(color="white"),
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        title_font=dict(size=20, color="white"),
        xaxis=dict(type="category", automargin=True),  # Use category type for x-axis
        yaxis=dict(type="category", automargin=True),  # Use category type for y-axis
    )

    # Update axes with explicit configuration
    fig.update_xaxes(
        side="top",
        title_font=dict(size=16, color="white"),
        tickfont=dict(color="white"),
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(size=16, color="white"),
        tickfont=dict(color="white"),
        automargin=True,
    )

    return fig


# GDACS PAGE IMPLEMENTATION
def show_gdacs_page():
    # Initialize chat session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome to the Disaster Assistant! Please enter your location to get started.",
            }
        ]

    if "user_location" not in st.session_state:
        st.session_state.user_location = None

    if "user_coordinates" not in st.session_state:
        st.session_state.user_coordinates = (None, None)

    # Header
    st.markdown(
        "<h1 class='main-header'>Environmental Catastrophe Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("### Real-time monitoring and detection of environmental incidents")

    # Add refresh button with last updated time
    col1, col2 = st.columns([3, 1])
    with col1:
        if "last_updated" not in st.session_state:
            st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.markdown(
            f'<p class="last-updated">Last updated: {st.session_state.last_updated}</p>',
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("🔄 Refresh Data", help="Fetch the latest GDACS alerts"):
            with st.spinner("Fetching latest data..."):
                # Clear the cache to force a refresh
                fetch_gdacs_data.clear()
                # Update timestamp
                st.session_state.last_updated = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.rerun()

    try:
        # Fetch GDACS data
        with st.spinner("Loading live GDACS alerts..."):
            df = fetch_gdacs_data()

        # Create a two-column layout with map on left, chat on right
        col_map, col_chat = st.columns([3, 2], gap="large")

        with col_map:
            # Add a class for styling
            st.markdown('<div class="main-column-left">', unsafe_allow_html=True)

            # Summary metrics
            st.markdown(
                "<h2 class='sub-header'>Overall Statistics</h2>", unsafe_allow_html=True
            )

            # Create metrics in a row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                active_count = len(df) if not df.empty else 0
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <h3>Live Events</h3>
                    <h2>{active_count}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                high_severity_count = (
                    len(df[df["severity"] == "High"]) if not df.empty else 0
                )
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <h3>High Severity</h3>
                    <h2>{high_severity_count}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                countries_count = len(df["country"].unique()) if not df.empty else 0
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <h3>Countries Affected</h3>
                    <h2>{countries_count}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                event_types_count = (
                    len(df["event_type"].unique()) if not df.empty else 0
                )
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <h3>Event Types</h3>
                    <h2>{event_types_count}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Map section
            st.markdown(
                "<h2 class='sub-header'>GDACS Live Incident Map</h2>",
                unsafe_allow_html=True,
            )

            if df.empty:
                st.warning("No live GDACS alerts available. Please try refreshing.")
            else:
                # Filters for the map
                col1, col2 = st.columns(2)

                with col1:
                    selected_severity = st.multiselect(
                        "Filter by Severity",
                        options=df["severity"].unique(),
                        default=list(df["severity"].unique()),
                    )

                with col2:
                    selected_event_types = st.multiselect(
                        "Filter by Event Type",
                        options=df["event_type"].unique(),
                        default=list(df["event_type"].unique()),
                    )

                # Apply filters
                filtered_df = df[
                    (df["severity"].isin(selected_severity))
                    & (df["event_type"].isin(selected_event_types))
                ]

                # Create and display the Folium map
                m = create_folium_map(filtered_df)
                # Use st_folium instead of folium_static
                st_folium(m, width=800, height=450)

                # Recent alerts section
                st.markdown(
                    "<h2 class='sub-header'>High-Priority Live Alerts</h2>",
                    unsafe_allow_html=True,
                )

                # Get high severity events
                high_severity = (
                    df[df["severity"] == "High"]
                    .sort_values("detected_at", ascending=False)
                    .head(3)
                )

                if not high_severity.empty:
                    for _, event in high_severity.iterrows():
                        # Add icon based on event type
                        icon = ""
                        if "Oil" in event["event_type"]:
                            icon = "🛢️ "
                        elif "Fire" in event["event_type"]:
                            icon = "🔥 "
                        elif "Flood" in event["event_type"]:
                            icon = "💧 "
                        elif (
                            "Cyclone" in event["event_type"]
                            or "Storm" in event["event_type"]
                        ):
                            icon = "🌀 "
                        elif "Earthquake" in event["event_type"]:
                            icon = "⚡ "
                        elif "Volcano" in event["event_type"]:
                            icon = "🌋 "
                        elif "Tsunami" in event["event_type"]:
                            icon = "🌊 "
                        elif "Drought" in event["event_type"]:
                            icon = "☀️ "

                        st.markdown(
                            f"""
                        <div class='alert-high'>
                            <h3>{icon}{event['event_type']} detected in {event['country']}</h3>
                            <p><strong>Detected:</strong> {event['detected_at']} | <strong>Alert Level:</strong> {event['alert_level']}</p>
                            <p><strong>Description:</strong> {event['description'][:150]}...</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No high severity alerts currently")

                # Close the div for left column
                st.markdown("</div>", unsafe_allow_html=True)

        with col_chat:
            # Add a class for styling
            st.markdown(
                '<div class="main-column-right"><div class="assistant-section">',
                unsafe_allow_html=True,
            )

            # Chatbot implementation
            st.markdown(
                "<h2 class='sub-header'>Disaster Assistant</h2>", unsafe_allow_html=True
            )

            # Check if user has provided a location
            has_location = (
                st.session_state.get("user_location") is not None
                and st.session_state.get("user_coordinates", (None, None))[0]
                is not None
            )

            # Location input with better error handling
            location_input = st.text_input(
                "Enter your location:",
                key="location_input",
                help="Enter a continent, country, or region (e.g., 'USA', 'Europe', 'Asia')",
            )

            # Check location input and update coordinates with improved error handling
            if location_input and location_input != st.session_state.get(
                "user_location", ""
            ):
                with st.spinner(f"Finding location: {location_input}"):
                    # Process input for countries
                    processed_location = validate_country(location_input)

                    # Get coordinates for the location
                    lat, lon = get_coordinates(processed_location)

                    if lat is not None and lon is not None:
                        st.session_state.user_location = (
                            location_input  # Keep original input for display
                        )
                        st.session_state.user_coordinates = (lat, lon)

                        # Find nearby disasters
                        nearby_disasters = find_disasters_near_location(df, lat, lon)

                        # Add system message about location
                        if not nearby_disasters.empty:
                            num_disasters = len(nearby_disasters)
                            disaster_types = ", ".join(
                                nearby_disasters["event_type"].unique()
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"I've found {num_disasters} active {'disaster' if num_disasters == 1 else 'disasters'} near {location_input}, including {disaster_types}. Ask me for more details!",
                                }
                            )
                        else:
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Good news! I don't see any active disasters near {location_input}. Feel free to ask me about any other locations or general disaster information.",
                                }
                            )

                        st.rerun()
                    else:
                        st.error(
                            f"I couldn't find the location '{location_input}'. Please try a different location like a continent, country, or region name."
                        )

            # Display content based on whether location is set
            if not has_location:
                # Display welcome message when no location is set yet
                st.info(
                    "Welcome to the Disaster Assistant! Please enter your location above to get started."
                )
            else:
                # Get user location details
                lat, lon = st.session_state.user_coordinates
                location_name = st.session_state.user_location

                # Find nearby disasters
                nearby_disasters = find_disasters_near_location(df, lat, lon)

                # Create location map
                location_map = create_user_location_map(
                    lat, lon, location_name, nearby_disasters
                )

                # Create tabs for map and chat
                map_tab, chat_tab = st.tabs(["Your Location", "Chat History"])

                with map_tab:
                    # Display the user's location map
                    if location_map:
                        st_folium(location_map, width=700, height=400)

                        # Add informative caption
                        if not nearby_disasters.empty:
                            disaster_count = len(nearby_disasters)
                            st.caption(
                                f"Map showing your location ({location_name}) with {disaster_count} nearby disaster{'s' if disaster_count != 1 else ''}"
                            )
                        else:
                            st.caption(
                                f"Map showing your location ({location_name}). No disasters detected within 500km."
                            )

                with chat_tab:
                    # Create container for chat messages
                    if st.session_state.messages:
                        st.markdown(
                            "<div class='chat-container'>", unsafe_allow_html=True
                        )
                        st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

                        for message in st.session_state.messages:
                            if message["role"] == "user":
                                st.markdown(
                                    f"<div class='user-message'>{message['content']}</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div class='bot-message'>{message['content']}</div>",
                                    unsafe_allow_html=True,
                                )

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            # Chat input - using a form for immediate submission
            with st.form(key="chat_form", clear_on_submit=True):
                user_query = st.text_input(
                    "Ask about disasters in your area:", key="chat_input"
                )
                submit_button = st.form_submit_button("Send")

                if submit_button and user_query:
                    # Add user message to chat
                    st.session_state.messages.append(
                        {"role": "user", "content": user_query}
                    )

                    # Only process if we have a location
                    if (
                        st.session_state.get("user_location")
                        and st.session_state.get("user_coordinates", (None, None))[0]
                        is not None
                    ):
                        with st.spinner("Processing your question..."):
                            lat, lon = st.session_state.user_coordinates

                            # Find nearby disasters
                            nearby_disasters = find_disasters_near_location(
                                df, lat, lon
                            )

                            # Get AI response
                            ai_response = get_ai_response(
                                user_query,
                                st.session_state.user_location,
                                nearby_disasters,
                            )

                            # Add AI response to chat
                            st.session_state.messages.append(
                                {"role": "assistant", "content": ai_response}
                            )
                    else:
                        # Prompt user to enter location first
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "Please enter your location first, so I can provide information about disasters in your area.",
                            }
                        )

                    st.rerun()

            # Add options to clear chat
            if st.button(
                "Clear Chat History", key="clear_chat", help="Clear all chat messages"
            ):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Welcome to the Disaster Assistant! Please enter your location to get started.",
                    }
                ]
                st.rerun()

            # Add info about the assistant
            with st.expander("About the Disaster Assistant"):
                st.markdown(
                    """
                This AI-powered assistant provides information about active disasters near your location from the Global Disaster Alert and Coordination System (GDACS).
                
                You can ask about:
                - Specific disasters in your area
                - Safety recommendations
                - Details about disaster severity and impact
                - Historical disaster information
                
                The assistant uses real-time GDACS data and has a search radius of 500km from your specified location.
                """
                )

            # Close the divs for right column
            st.markdown("</div></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(
            "Please try refreshing the data or check the console for detailed error information."
        )


# Optimized flood data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_flood_data():
    """Load flood prediction results from JSON file with improved performance."""
    try:
        with open("flood_detection_results.json", "r") as f:
            # Read just the first 500 items instead of the entire file
            # This is a simple approach - for production, you might want a more sophisticated solution
            data = json.load(f)[:500]  # Limit to 500 items

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Ensure date is in datetime format for filtering
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df
    except Exception as e:
        st.error(f"Error loading flood detection data: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "location_id",
                "location_folder",
                "date",
                "filename",
                "latitude",
                "longitude",
                "true_label",
                "predicted_label",
                "confidence",
                "coordinates",
                "environmental_impact",
                "economic_impact",
                "social_impact",
                "affected_area_km2",
                "total_impact",
                "status",
                "risk_score",
            ]
        )


# FLOOD PREDICTION PAGE (streamlined version)
def show_flood_prediction_page():
    # Header
    st.markdown(
        "<h1 class='main-header'>Flood Prediction Analysis</h1>", unsafe_allow_html=True
    )
    st.markdown("### AI-based flood detection from satellite imagery")

    # Load data - limit to 500 most recent floods for performance
    with st.spinner("Loading flood prediction data..."):
        df = load_flood_data()
        if not df.empty and "date" in df.columns:
            # Sort by date (most recent first) and limit to 500 entries
            df = df.sort_values("date", ascending=False).head(500)

    if df.empty:
        st.error(
            "No flood prediction data available. Please run the flood detection model first."
        )
        return

    # Add timestamp for last predictions
    if "date" in df.columns and not df.empty:
        latest_date = df["date"].max()
        st.markdown(
            f'<p class="last-updated">Latest prediction: {latest_date.strftime("%Y-%m-%d") if isinstance(latest_date, pd.Timestamp) else latest_date}</p>',
            unsafe_allow_html=True,
        )

    # Date range filter only - removed confidence threshold
    min_date = (
        df["date"].min() if "date" in df.columns and not df.empty else datetime.now()
    )
    max_date = (
        df["date"].max() if "date" in df.columns and not df.empty else datetime.now()
    )

    if isinstance(min_date, pd.Timestamp) and isinstance(max_date, pd.Timestamp):
        date_range = st.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        # Convert to datetime for filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[
                (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
            ]
        else:
            filtered_df = df
    else:
        filtered_df = df

    # Show only floods toggle
    show_only_floods = st.checkbox("Show Only Detected Floods", value=True)
    if show_only_floods:
        filtered_df = filtered_df[filtered_df["predicted_label"] == 1]

    # Summary metrics
    active_floods = len(filtered_df[filtered_df["predicted_label"] == 1])
    avg_confidence = (
        filtered_df[filtered_df["predicted_label"] == 1]["confidence"].mean()
        if active_floods > 0
        else 0
    )
    total_area = (
        filtered_df[filtered_df["predicted_label"] == 1]["affected_area_km2"].sum()
        if active_floods > 0
        else 0
    )
    avg_impact = (
        filtered_df[filtered_df["predicted_label"] == 1]["total_impact"].mean()
        if active_floods > 0
        else 0
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>Active Floods</h3>
                <h2>{active_floods}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>Avg. Confidence</h3>
                <h2>{avg_confidence:.1%}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>Total Area</h3>
                <h2>{total_area:.1f} km²</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>Avg. Impact</h3>
                <h2>{avg_impact:.1f}/10</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Create and display map
    st.markdown("### Flood Prediction Map")

    # Limit map data for performance
    map_df = filtered_df.head(100)  # Just show top 100 entries on map
    flood_map = create_flood_map(map_df)
    st_folium(flood_map, width=1200, height=600)

    # Just show high impact floods section and nothing below
    st.markdown("### High Impact Flood Events")
    high_impact_floods = (
        filtered_df[
            (filtered_df["predicted_label"] == 1) & (filtered_df["total_impact"] >= 7.0)
        ]
        .sort_values("total_impact", ascending=False)
        .head(3)
    )

    if not high_impact_floods.empty:
        for _, event in high_impact_floods.iterrows():
            # Format date for display
            event_date = event["date"]
            if isinstance(event_date, pd.Timestamp):
                formatted_date = event_date.strftime("%Y-%m-%d")
            else:
                formatted_date = str(event_date)

            st.markdown(
                f"""
                <div class='alert-high'>
                    <h3>🌊 Flood detected in Location {event['location_id']}</h3>
                    <p><strong>Date:</strong> {formatted_date} | <strong>Confidence:</strong> {event['confidence']:.1%} | <strong>Impact:</strong> {event['total_impact']:.1f}/10</p>
                    <p><strong>Affected Area:</strong> {event['affected_area_km2']:.2f} km² | <strong>Environmental Impact:</strong> {event['environmental_impact']:.1f}/10 | <strong>Economic Impact:</strong> {event['economic_impact']:.1f}/10</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No high impact flood events in the selected date range")

    # Add download options for the data
    st.markdown("### Download Filtered Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="flood_predictions_export.csv",
            mime="text/csv",
        )

    with col2:
        # Export as JSON
        json_data = filtered_df.to_json(orient="records", date_format="iso")
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="flood_predictions_export.json",
            mime="application/json",
        )


# MAIN APPLICATION
def main():
    # Display navigation bar
    navigation()

    # Display appropriate page based on session state
    if st.session_state.page == "gdacs":
        show_gdacs_page()
    elif st.session_state.page == "flood_predictions":
        show_flood_prediction_page()


if __name__ == "__main__":
    main()
