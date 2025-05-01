# AI Travel Planner

A streamlined multi-agent system for intelligent travel planning, built with LangGraph and Streamlit.

## Project Overview

AI Travel Planner is a sophisticated multi-agent system designed to help users plan their trips. The system employs specialized agents that collaborate to provide comprehensive travel assistance:

- **Supervisor Agent**: Coordinates between specialist agents and handles initial request analysis
- **Flight Agent**: Searches for and recommends flight options
- **Hotel Agent**: Finds suitable accommodations based on user preferences
- **Itinerary Agent**: Creates complete travel schedules and suggests activities

The system uses SerpAPI for real-time travel data and can be configured to use either OpenAI's API or a local Ollama instance (Llama 3.2) for language processing.

## Features

- **Natural Language Interface**: Simply describe your travel plans in plain language
- **Multi-Agent Coordination**: Specialized agents work together to handle different aspects of trip planning
- **Real-Time Flight and Hotel Search**: Uses SerpAPI to find current travel options
- **Flexible LLM Support**: Choose between OpenAI's API or local Ollama for processing
- **Beautiful Streamlit UI**: Clean, intuitive interface for interacting with the system
- **Visual Agent Identification**: Each agent is visually indicated in the chat interface
- **Itinerary Creation**: Generate comprehensive travel plans with flights, accommodations, and activities
- **PDF Export**: Save and download your complete travel itinerary

## Setup Instructions

### Prerequisites

- Python 3.9+
- SerpAPI key (for search functionality)
- OpenAI API key (optional, if using OpenAI as the LLM provider)
- Ollama with Llama 3.2 installed (optional, if using Ollama as the LLM provider)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-travel-planner.git
   cd ai-travel-planner
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file or directly in `config.py`:
   ```
   SERPAPI_API_KEY=your_serpapi_api_key
   OPENAI_API_KEY=your_openai_api_key (if using OpenAI)
   LLM_PROVIDER=openai (or "ollama" if using Ollama)
   OLLAMA_MODEL=llama3.2 (if using Ollama)
   OLLAMA_URL=http://localhost:11434 (if using Ollama)
   ```

### Running the Application

Simply run the provided script:

```bash
python run_app.py
```

Or run Streamlit directly:

```bash
streamlit run app.py
```

## Usage

1. Start the application using either method above
2. In the text input field, describe your travel plans, including:
   - Origin and destination cities
   - Travel dates (YYYY-MM-DD format)
   - Number of travelers
   - Budget (optional)
   - Any special preferences

Example query:

```
I want to travel from New York to Miami from 2025-06-15 to 2025-06-20 with 2 adults and a budget of $3000.
```

3. The system will process your request and provide flight and hotel options
4. Select your preferred options to generate a complete itinerary
5. Download or email your itinerary as needed

## Switching LLM Providers

The system supports both OpenAI and Ollama as LLM providers:

- For OpenAI: Ensure you have set your `OPENAI_API_KEY` in the environment variables or config.py
- For Ollama: Install Ollama locally, run the Ollama server, and set `LLM_PROVIDER=ollama` in your environment variables or config.py

You can toggle between providers in the debug panel of the application.

## Project Structure

The project consists of four main files:

- `app.py`: The Streamlit user interface and main application logic
- `booking_agent.py`: Implementation of the multi-agent system using LangGraph
- `config.py`: Configuration settings for API keys, LLM settings, and other parameters
- `run_app.py`: Helper script to run the application with proper Python module imports

## Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/), [Streamlit](https://streamlit.io/), and [SerpAPI](https://serpapi.com/)
- Created as part of the Multi-Agent System Hackathon
