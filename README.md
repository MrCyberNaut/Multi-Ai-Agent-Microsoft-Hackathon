# Multi-AI-Agent Microsoft Hackathon

A sophisticated multi-agent system for travel planning, leveraging OpenAI's API to provide intelligent travel assistance through specialized agents.

## Features

- **Triage Agent**: Initial contact point that analyzes user requests and routes them to specialized agents
- **Flight Agent**: Handles flight searches and bookings
- **Hotel Agent**: Manages hotel searches and reservations
- **Itinerary Agent**: Creates complete travel itineraries

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Multi-Ai-Agent-Microsoft-Hackathon.git
cd Multi-Ai-Agent-Microsoft-Hackathon
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY='your_api_key_here'
```

## Project Structure

```
agents/
├── supervisor.py    # Main supervisor agent implementation
├── flight_agent.py  # Flight search and booking agent
├── hotel_agent.py   # Hotel search and booking agent
└── itinerary_agent.py  # Travel itinerary creation agent
```

## Configuration

Update `config.py` with your OpenAI API key and other settings.

## Usage

[Usage instructions to be added]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
