# MindBridge AI

MindBridge AI is a context-aware mental health chatbot built with LangGraph and LangChain. It provides supportive, goal-oriented, and crisis-aware conversations, using persistent memory to remember your session and goals.

## Features

- Multi-agent workflow: routes messages to crisis, support, goal, or info agents
- Context-aware: Remembers recent conversation and user goals
- Persistent memory: Saves your session and goals between runs
- Crisis detection and safety resources
- Goal setting and progress tracking
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
- The bot will remember your conversation and goals.
- Type `quit` or `exit` to end the session.

## Tech Stack

- Python 3.9+
- LangChain
- LangGraph
- Groq API
- DeepSeek Model

## Requirements

- Python 3.9+
- Groq API key