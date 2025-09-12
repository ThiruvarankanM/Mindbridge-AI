
import os
import json
from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()

# Simple Models
class MessageType(BaseModel):
    category: Literal["crisis", "support", "goal", "info"] = "support"
    urgency: Literal["low", "medium", "high"] = "medium"

# State
class State(TypedDict):
    messages: list[dict]   # keep messages as simple dicts
    message_type: str
    user_goals: list[str]

# Initialize LLM
llm = init_chat_model("groq:deepseek-r1-distill-llama-70b")

def analyze_message(state: State):
    """Analyze incoming message to determine routing"""
    last_message = state["messages"][-1]["content"]
    
    if any(word in last_message.lower() for word in ["suicide", "kill myself", "end it", "harm"]):
        message_type = "crisis"
    elif any(word in last_message.lower() for word in ["goal", "want to", "plan", "achieve"]):
        message_type = "goal"  
    elif any(word in last_message.lower() for word in ["how", "what", "why", "explain"]):
        message_type = "info"
    else:
        message_type = "support"
    
    return {"message_type": message_type}

def get_history_context(messages, max_turns=3):
    # Get the last max_turns user/assistant messages for context
    history = messages[-max_turns*2:] if len(messages) > 1 else []
    context = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" for m in history if m.get('role') in ['user', 'assistant']
    ])
    return context

def crisis_agent(state: State):
    """Handle crisis situations with safety resources"""
    last_message = state["messages"][-1]["content"]
    context = get_history_context(state["messages"])
    
    response = llm.invoke([
        {
            "role": "system", 
            "content": f"""You are a crisis counselor. Provide immediate support and safety resources.\n\nRecent conversation:\n{context}\n\nAlways include: National Suicide Prevention Lifeline: 988\nKeep response under 3 sentences and focus on immediate safety."""
        },
        {"role": "user", "content": last_message}
    ])
    
    crisis_resources = "\n\n🚨 IMMEDIATE HELP:\n• Call 988 (Suicide Prevention)\n• Text HOME to 741741\n• Call 911 if in danger"
    
    return {"messages": [{"role": "assistant", "content": str(response.content) + crisis_resources}]}

def support_agent(state: State):
    """Provide therapeutic support"""
    last_message = state["messages"][-1]["content"]
    goals_context = f"User's goals: {state.get('user_goals', [])}" if state.get('user_goals') else ""
    context = get_history_context(state["messages"])
    
    response = llm.invoke([
        {
            "role": "system",
            "content": f"""You are an empathetic therapist. Provide supportive, caring responses.\n\nRecent conversation:\n{context}\n\nKeep responses to 2-3 sentences. Be warm and encouraging.\n{goals_context}"""
        },
        {"role": "user", "content": last_message}
    ])
    
    return {"messages": [{"role": "assistant", "content": str(response.content)}]}

def goal_agent(state: State):
    """Help with goal setting and tracking"""
    last_message = state["messages"][-1]["content"]
    current_goals = state.get("user_goals", [])
    context = get_history_context(state["messages"])
    
    response = llm.invoke([
        {
            "role": "system",
            "content": f"""You are helping with goal setting.\n\nRecent conversation:\n{context}\n\nCurrent goals: {current_goals}\nHelp create specific, achievable goals. Keep response to 2-3 sentences."""
        },
        {"role": "user", "content": last_message}
    ])
    
    new_goals = current_goals.copy()
    if "goal" in last_message.lower() and len(last_message) > 20:
        goal_text = last_message[:100] + "..." if len(last_message) > 100 else last_message
        if goal_text not in new_goals:
            new_goals.append(goal_text)
    
    return {
        "messages": [{"role": "assistant", "content": str(response.content)}],
        "user_goals": new_goals
    }

def info_agent(state: State):
    """Handle informational questions"""
    last_message = state["messages"][-1]["content"]
    context = get_history_context(state["messages"])
    
    response = llm.invoke([
        {
            "role": "system",
            "content": f"""Provide clear, helpful information with a therapeutic context.\n\nRecent conversation:\n{context}\n\nKeep responses educational but warm. 2-3 sentences maximum."""
        },
        {"role": "user", "content": last_message}
    ])
    
    return {"messages": [{"role": "assistant", "content": str(response.content)}]}

def route_message(state: State):
    """Route to appropriate agent based on message type"""
    return state["message_type"]

def create_graph():
    """Build the LangGraph workflow"""
    graph = StateGraph(State)
    
    graph.add_node("analyze", analyze_message)
    graph.add_node("crisis", crisis_agent)
    graph.add_node("support", support_agent)
    graph.add_node("goal", goal_agent)
    graph.add_node("info", info_agent)
    
    graph.add_edge(START, "analyze")
    
    graph.add_conditional_edges(
        "analyze",
        route_message,
        {
            "crisis": "crisis",
            "support": "support", 
            "goal": "goal",
            "info": "info"
        }
    )
    
    graph.add_edge("crisis", END)
    graph.add_edge("support", END)
    graph.add_edge("goal", END)
    graph.add_edge("info", END)
    
    return graph.compile(checkpointer=MemorySaver())


def load_state(filename="chat_memory.json"):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "messages": [],
        "message_type": "support",
        "user_goals": []
    }

def save_state(state, filename="chat_memory.json"):
    try:
        with open(filename, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

def run_chatbot():
    """Simple chatbot interface with persistent memory"""
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Please set GROQ_API_KEY in your .env file")
        return

    graph = create_graph()

    print("🤖 Simple Therapy Assistant (LangGraph Demo)")
    print("Type 'quit' to exit\n")

    state = load_state()
    config = {"configurable": {"thread_id": "demo"}}

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\n👋 Take care!")
            save_state(state)
            break

        state["messages"].append({"role": "user", "content": user_input})

        try:
            result = graph.invoke(state, config=config)
            state.update(result)
            last_response = result["messages"][-1]

            print(f"\nBot: {last_response['content']}\n")

            if result.get("user_goals") and len(result["user_goals"]) > len(state.get("user_goals", [])):
                print("✅ Goal noted!")

            save_state(state)

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Bot: I'm here to help. Can you tell me more?\n")

if __name__ == "__main__":
    run_chatbot()
