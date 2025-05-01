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

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Options: "openai", "ollama"

# OpenAI model settings
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# Ollama model settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# SerpAPI settings
SERPAPI_TIMEOUT = 30  # seconds
SERPAPI_CACHE_DURATION = 3600  # 1 hour in seconds

# Human-in-the-loop settings
HITL_ENABLED = False  # Set to True to enable human-in-the-loop, False to disable
HITL_MODE = "streamlit"  # "cli" or "streamlit"
HITL_TIMEOUT = 300  # 5 minutes in seconds

# Other configuration
DEBUG_MODE = True
LOG_LEVEL = "INFO"
CACHE_DIR = "cache"
