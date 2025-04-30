"""Configuration settings for the travel assistant."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "your_serpapi_api_key")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY", "your_amadeus_api_key")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET", "your_amadeus_api_secret")
HOTEL_API_KEY = os.getenv("HOTEL_API_KEY", "your_hotel_api_key")

# OpenAI model settings
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# SerpAPI settings
SERPAPI_TIMEOUT = 30  # seconds
SERPAPI_CACHE_DURATION = 3600  # 1 hour in seconds

# Human-in-the-loop settings
HITL_ENABLED = True
HITL_TIMEOUT = 300  # 5 minutes in seconds

# Other configuration
DEBUG_MODE = True
LOG_LEVEL = "INFO"
CACHE_DIR = "cache"
