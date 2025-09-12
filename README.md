# MindBridge AI

MindBridge AI is a mental health chatbot that uses advanced multi-agent orchestration with LangGraph and LangChain. It provides personalized, crisis-aware therapeutic support, including goal setting, progress tracking, and crisis intervention.

## Features

- Multi-step conversation flow with persistent memory
- Crisis detection and intervention
- Goal setting and progress tracking
- Session summaries and user profile management
- Powered by Groq API and DeepSeek model

## Setup

1. Clone this repository.
2. Install dependencies:
	```
	pip install -r requirements.txt
	```
3. Create a `.env` file with your Groq API key:
	```
	GROQ_API_KEY=your_groq_api_key_here
	```
4. Run the chatbot:
	```
	python main.py
	```

## Usage

- Type your message to start a session.
- Use commands: `goals`, `progress`, `summary`, or `exit` to interact with the assistant.

## Tech Stack

- Python 3.9+
- LangChain
- LangGraph
- Groq API
- DeepSeek Model

## Requirements

- Python 3.9+
- Groq API key