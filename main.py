import os
from dotenv import load_dotenv
from typing import Annotated, Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import json
from datetime import datetime

load_dotenv()

llm = init_chat_model("groq:deepseek-r1-distill-llama-70b")

# Enhanced Models
class MessageAnalysis(BaseModel):
    message_type: Literal["emotional_support", "crisis", "logical_inquiry", "follow_up", "goal_setting"] = Field(
        description="Classify the message type for appropriate routing"
    )
    urgency_level: Literal["low", "medium", "high", "crisis"] = Field(
        description="Urgency level of the user's situation"
    )
    emotional_state: str = Field(description="User's apparent emotional state")
    main_concerns: List[str] = Field(description="Key concerns or topics mentioned")

class CrisisAssessment(BaseModel):
    is_crisis: bool = Field(description="Whether this requires immediate intervention")
    risk_level: Literal["none", "low", "moderate", "high"] = Field(description="Risk assessment level")
    recommended_action: str = Field(description="Recommended immediate action")

class TherapyGoal(BaseModel):
    goal_text: str = Field(description="The therapy goal")
    target_date: str = Field(description="When to achieve this goal")
    action_steps: List[str] = Field(description="Concrete steps to achieve the goal")

class SessionSummary(BaseModel):
    key_insights: List[str] = Field(description="Main insights from the session")
    emotional_progress: str = Field(description="How the user's emotional state evolved")
    next_session_focus: str = Field(description="Suggested focus for next session")

# Enhanced State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_analysis: Dict[str, Any] | None
    crisis_assessment: Dict[str, Any] | None
    user_profile: Dict[str, Any]
    session_goals: List[Dict[str, Any]]
    conversation_memory: List[Dict[str, Any]]
    current_flow: str | None
    session_summary: Dict[str, Any] | None

def get_message_content(message):
    """Helper function to safely extract content from different message types"""
    if hasattr(message, 'content'):
        return message.content
    elif isinstance(message, dict):
        return message.get('content', '')
    else:
        return str(message)

def get_message_role(message):
    """Helper function to safely extract role from different message types"""
    if hasattr(message, 'type'):
        return 'user' if message.type == 'human' else 'assistant'
    elif isinstance(message, dict):
        return message.get('role', 'user')
    else:
        return 'user'

def analyze_message(state: State):
    """Advanced message analysis with multiple dimensions"""
    last_message = state["messages"][-1]
    last_content = get_message_content(last_message)
    
    # Include conversation context for better analysis
    context = ""
    if len(state["messages"]) > 1:
        recent_messages = state["messages"][-3:]  # Last 3 messages for context
        context = "Recent conversation context:\n" + "\n".join([
            f"{get_message_role(msg)}: {get_message_content(msg)}" 
            for msg in recent_messages[:-1]
        ])
    
    try:
        analyzer_llm = llm.with_structured_output(MessageAnalysis)
        result = analyzer_llm.invoke([
            {
                "role": "system",
                "content": f"""Analyze the user message considering:
                1. Message type: emotional support, crisis, logical inquiry, follow-up, or goal setting
                2. Urgency level: how immediate the response needs to be
                3. Emotional state: current emotional condition
                4. Main concerns: key topics to address
                
                {context}"""
            },
            {"role": "user", "content": last_content}
        ])
        
        return {
            "message_analysis": {
                "message_type": result.message_type,
                "urgency_level": result.urgency_level,
                "emotional_state": result.emotional_state,
                "main_concerns": result.main_concerns,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"Analysis failed: {e}")
        # Fallback analysis
        return {
            "message_analysis": {
                "message_type": "emotional_support",
                "urgency_level": "medium",
                "emotional_state": "seeking support",
                "main_concerns": ["general wellbeing"],
                "timestamp": datetime.now().isoformat()
            }
        }

def crisis_detection(state: State):
    """Dedicated crisis detection and risk assessment"""
    if not state.get("message_analysis"):
        return {"crisis_assessment": None}
    
    if state["message_analysis"]["urgency_level"] in ["high", "crisis"]:
        last_message = state["messages"][-1]
        last_content = get_message_content(last_message)
        
        try:
            crisis_llm = llm.with_structured_output(CrisisAssessment)
            result = crisis_llm.invoke([
                {
                    "role": "system",
                    "content": """Assess if this message indicates a mental health crisis requiring immediate intervention.
                    Look for: suicidal ideation, self-harm plans, psychotic episodes, severe depression/anxiety.
                    Be conservative - err on the side of caution for safety."""
                },
                {"role": "user", "content": last_content}
            ])
            
            return {
                "crisis_assessment": {
                    "is_crisis": result.is_crisis,
                    "risk_level": result.risk_level,
                    "recommended_action": result.recommended_action
                }
            }
        except Exception as e:
            print(f"Crisis detection failed: {e}")
            return {"crisis_assessment": None}
    
    return {"crisis_assessment": None}

def router(state: State):
    """Intelligent routing based on analysis"""
    analysis = state.get("message_analysis", {})
    crisis = state.get("crisis_assessment")
    
    # Crisis takes highest priority
    if crisis and crisis.get("is_crisis"):
        return {"current_flow": "crisis_intervention"}
    
    message_type = analysis.get("message_type", "emotional_support")
    
    if message_type == "goal_setting":
        return {"current_flow": "goal_setting"}
    elif message_type == "follow_up":
        return {"current_flow": "progress_tracking"}
    elif message_type == "logical_inquiry":
        return {"current_flow": "logical_response"}
    else:
        return {"current_flow": "therapeutic_support"}

def crisis_intervention_agent(state: State):
    """Specialized crisis intervention with safety protocols"""
    last_message = state["messages"][-1]
    last_content = get_message_content(last_message)
    crisis_info = state.get("crisis_assessment", {})
    
    messages = [
        {
            "role": "system",
            "content": f"""You are a crisis intervention specialist. The user is in potential crisis.
            
            Crisis Assessment: {crisis_info}
            
            IMMEDIATE PRIORITIES:
            1. Ensure immediate safety
            2. Provide crisis hotline numbers
            3. Encourage professional help
            4. Stay calm and supportive
            5. Do not dismiss their feelings
            
            Crisis Resources:
            - National Suicide Prevention Lifeline: 988
            - Crisis Text Line: Text HOME to 741741
            - Emergency Services: 911
            
            Keep responses short, direct, and focused on immediate safety."""
        },
        {"role": "user", "content": last_content}
    ]
    
    reply = llm.invoke(messages)
    reply_content = get_message_content(reply)
    
    # Log crisis intervention
    crisis_log = {
        "timestamp": datetime.now().isoformat(),
        "risk_level": crisis_info.get("risk_level"),
        "intervention_provided": True
    }
    
    return {
        "messages": [{
            "role": "assistant", 
            "content": f"🚨 CRISIS SUPPORT 🚨\n\n{reply_content}\n\n" + 
                      "Remember: If you're in immediate danger, call 911 or go to your nearest emergency room."
        }],
        "user_profile": {
            **state.get("user_profile", {}),
            "crisis_history": state.get("user_profile", {}).get("crisis_history", []) + [crisis_log]
        }
    }

def therapeutic_agent(state: State):
    """Enhanced therapeutic support with memory and personalization"""
    last_message = state["messages"][-1]
    last_content = get_message_content(last_message)
    analysis = state.get("message_analysis", {})
    profile = state.get("user_profile", {})
    memory = state.get("conversation_memory", [])
    
    # Build context from user profile and memory
    context_info = ""
    if profile:
        context_info += f"User Profile: {json.dumps(profile, indent=2)}\n"
    if memory:
        recent_memory = memory[-3:]  # Last 3 conversation memories
        context_info += f"Recent Session Notes: {json.dumps(recent_memory, indent=2)}\n"
    
    messages = [
        {
            "role": "system",
            "content": f"""You are an experienced, empathetic therapist. 
            
            Current Analysis: {analysis}
            
            {context_info}
            
            Approach:
            1. Validate their emotions briefly
            2. Ask one thoughtful question
            3. Offer one practical suggestion
            4. Keep responses short and conversational (2-3 sentences max)
            5. Do NOT include <think> tags or reasoning - just provide direct therapeutic response
            
            Be warm and caring but concise."""
        },
        {"role": "user", "content": last_content}
    ]
    
    reply = llm.invoke(messages)
    reply_content = get_message_content(reply)
    
    # Clean up any thinking tags that might appear
    if "<think>" in reply_content:
        reply_content = reply_content.split("</think>")[-1].strip()
    
    # Update conversation memory
    new_memory = {
        "timestamp": datetime.now().isoformat(),
        "user_emotional_state": analysis.get("emotional_state"),
        "main_concerns": analysis.get("main_concerns"),
        "therapy_approach_used": "supportive_therapy",
        "key_insights": []
    }
    
    return {
        "messages": [{"role": "assistant", "content": reply_content}],
        "conversation_memory": state.get("conversation_memory", []) + [new_memory]
    }

def goal_setting_agent(state: State):
    """Help users set and track therapeutic goals"""
    last_message = state["messages"][-1]
    last_content = get_message_content(last_message)
    existing_goals = state.get("session_goals", [])
    
    messages = [
        {
            "role": "system",
            "content": f"""You are a therapist helping with goal setting. Current goals: {existing_goals}
            
            Guide them to create SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound).
            Focus on mental health and emotional wellbeing objectives.
            Keep responses conversational and supportive (2-3 sentences)."""
        },
        {"role": "user", "content": last_content}
    ]
    
    therapy_response = llm.invoke(messages)
    reply_content = get_message_content(therapy_response)
    
    # Clean up any thinking tags
    if "<think>" in reply_content:
        reply_content = reply_content.split("</think>")[-1].strip()
    
    # Try to extract a goal if one was mentioned - simplified approach
    try:
        if any(word in last_content.lower() for word in ["goal", "want to", "plan to", "hope to"]):
            new_goal = {
                "id": len(existing_goals) + 1,
                "goal_text": f"User expressed: {last_content[:100]}...",
                "target_date": "To be determined",
                "action_steps": ["Discuss specific steps in next session"],
                "created_date": datetime.now().isoformat(),
                "status": "active"
            }
            updated_goals = existing_goals + [new_goal]
        else:
            updated_goals = existing_goals
    except:
        updated_goals = existing_goals
    
    return {
        "messages": [{"role": "assistant", "content": reply_content}],
        "session_goals": updated_goals
    }

def progress_tracking_agent(state: State):
    """Track progress on goals and overall wellbeing"""
    goals = state.get("session_goals", [])
    memory = state.get("conversation_memory", [])
    last_message = state["messages"][-1]
    last_content = get_message_content(last_message)
    
    progress_context = f"Current Goals: {json.dumps(goals, indent=2)}\n"
    progress_context += f"Recent Sessions: {json.dumps(memory[-3:], indent=2) if memory else 'None'}"
    
    messages = [
        {
            "role": "system",
            "content": f"""You are reviewing progress with the user. 
            
            {progress_context}
            
            Help them:
            1. Reflect on their progress toward goals
            2. Identify what's working and what isn't
            3. Adjust goals if needed
            4. Celebrate achievements
            5. Address obstacles
            
            Be encouraging and help them see their growth. Keep responses 2-3 sentences."""
        },
        {"role": "user", "content": last_content}
    ]
    
    reply = llm.invoke(messages)
    reply_content = get_message_content(reply)
    
    # Clean up any thinking tags
    if "<think>" in reply_content:
        reply_content = reply_content.split("</think>")[-1].strip()
    
    return {"messages": [{"role": "assistant", "content": reply_content}]}

def logical_agent(state: State):
    """Handle logical/informational queries"""
    last_message = state["messages"][-1]
    last_content = get_message_content(last_message)
    
    messages = [
        {
            "role": "system",
            "content": """Provide clear, evidence-based information. Be helpful and informative 
            while maintaining a warm tone since this is in a therapeutic context.
            Keep responses concise (2-3 sentences)."""
        },
        {"role": "user", "content": last_content}
    ]
    
    reply = llm.invoke(messages)
    reply_content = get_message_content(reply)
    
    # Clean up any thinking tags
    if "<think>" in reply_content:
        reply_content = reply_content.split("</think>")[-1].strip()
    
    return {"messages": [{"role": "assistant", "content": reply_content}]}

def session_summarizer(state: State):
    """Generate session summary and insights"""
    if len(state["messages"]) < 6:  # Skip summary for very short conversations
        return {"session_summary": None}
    
    conversation_text = "\n".join([
        f"{get_message_role(msg)}: {get_message_content(msg)}" 
        for msg in state["messages"][-10:]  # Last 10 messages
    ])
    
    try:
        summary_llm = llm.with_structured_output(SessionSummary)
        result = summary_llm.invoke([
            {
                "role": "system",
                "content": """Create a therapeutic session summary focusing on:
                1. Key insights that emerged
                2. How the user's emotional state evolved
                3. What to focus on in the next session"""
            },
            {"role": "user", "content": f"Summarize this therapy session:\n{conversation_text}"}
        ])
        
        return {
            "session_summary": {
                "key_insights": result.key_insights,
                "emotional_progress": result.emotional_progress,
                "next_session_focus": result.next_session_focus,
                "timestamp": datetime.now().isoformat()
            }
        }
    except:
        return {"session_summary": None}

# Build Enhanced Graph
def create_therapy_graph():
    graph_builder = StateGraph(State)
    
    # Add all nodes
    graph_builder.add_node("analyze", analyze_message)
    graph_builder.add_node("crisis_check", crisis_detection)
    graph_builder.add_node("route", router)
    graph_builder.add_node("crisis_intervention", crisis_intervention_agent)
    graph_builder.add_node("therapy", therapeutic_agent)
    graph_builder.add_node("goal_setting", goal_setting_agent)
    graph_builder.add_node("progress_tracking", progress_tracking_agent)
    graph_builder.add_node("logical", logical_agent)
    graph_builder.add_node("summarize", session_summarizer)
    
    # Define flow
    graph_builder.add_edge(START, "analyze")
    graph_builder.add_edge("analyze", "crisis_check")
    graph_builder.add_edge("crisis_check", "route")
    
    # Conditional routing
    graph_builder.add_conditional_edges(
        "route",
        lambda state: state.get("current_flow"),
        {
            "crisis_intervention": "crisis_intervention",
            "therapeutic_support": "therapy",
            "goal_setting": "goal_setting", 
            "progress_tracking": "progress_tracking",
            "logical_response": "logical"
        }
    )
    
    # All paths lead to summarizer, then end
    for node in ["crisis_intervention", "therapy", "goal_setting", "progress_tracking", "logical"]:
        graph_builder.add_edge(node, "summarize")
    
    graph_builder.add_edge("summarize", END)
    
    return graph_builder.compile(checkpointer=MemorySaver())

def run_enhanced_chatbot():
    """Enhanced chatbot with persistent memory and advanced features"""
    # Check if GROQ API key is available
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in environment variables!")
        print("Please create a .env file with: GROQ_API_KEY=your_api_key_here")
        return
    
    graph = create_therapy_graph()
    
    # Initialize persistent state
    state = {
        "messages": [],
        "user_profile": {
            "first_session": datetime.now().isoformat(),
            "session_count": 0,
            "crisis_history": []
        },
        "session_goals": [],
        "conversation_memory": [],
        "message_analysis": None,
        "crisis_assessment": None,
        "current_flow": None,
        "session_summary": None
    }
    
    print("🌟 MindBridge AI - Enhanced Therapy Assistant")
    print("🔧 Powered by GROQ API | Built with LangGraph")
    print("\nCommands:")
    print("  'goals' - Work on therapeutic goals")
    print("  'progress' - Review your progress")  
    print("  'summary' - See session summary")
    print("  'exit' - End session\n")
    
    thread_config = {"configurable": {"thread_id": "therapy_session_1"}}
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                # Generate final summary
                if state.get("conversation_memory"):
                    try:
                        final_state = graph.invoke(
                            {**state, "messages": state["messages"] + [{"role": "user", "content": "Please provide a session summary"}]},
                            config=thread_config
                        )
                        if final_state.get("session_summary"):
                            print("\n📋 Session Summary:")
                            summary = final_state["session_summary"]
                            print(f"Key Insights: {', '.join(summary.get('key_insights', []))}")
                            print(f"Emotional Progress: {summary.get('emotional_progress', 'N/A')}")
                            print(f"Next Focus: {summary.get('next_session_focus', 'N/A')}")
                    except Exception as e:
                        print(f"Summary generation failed: {e}")
                
                print("\n💙 Take care of yourself. Remember, healing is a journey.")
                break
            
            if user_input.lower() == "goals":
                user_input = "I'd like to work on setting some therapeutic goals for myself."
            elif user_input.lower() == "progress":
                user_input = "Can you help me review my progress on my goals and overall wellbeing?"
            elif user_input.lower() == "summary":
                if state.get("session_summary"):
                    print(f"\n📋 Latest Session Summary: {json.dumps(state['session_summary'], indent=2)}")
                else:
                    print("No session summary available yet. Continue our conversation!")
                continue
            
            # Add user message and process
            new_state = {
                **state,
                "messages": state["messages"] + [{"role": "user", "content": user_input}]
            }
            
            # Run through the graph with error handling
            try:
                result_state = graph.invoke(new_state, config=thread_config)
                
                # Update persistent state
                state = result_state
                state["user_profile"]["session_count"] += 1
                
                # Display response
                if result_state.get("messages"):
                    last_response = result_state["messages"][-1]
                    if isinstance(last_response, dict) and last_response.get("role") == "assistant":
                        print(f"\nTherapist: {last_response['content']}\n")
                    else:
                        # Handle LangChain message objects
                        content = get_message_content(last_response)
                        role = get_message_role(last_response)
                        if role == "assistant":
                            print(f"\nTherapist: {content}\n")
                
                # Show any alerts or updates
                crisis_assessment = result_state.get("crisis_assessment")
                if crisis_assessment and crisis_assessment.get("is_crisis"):
                    print("⚠️  Crisis intervention protocols activated.")
                
                current_goals = result_state.get("session_goals", [])
                previous_goals = state.get("session_goals", [])
                if len(current_goals) > len(previous_goals):
                    print("✅ New therapeutic goal added!")
                    
            except Exception as e:
                print(f"❌ Error processing request: {e}")
                print("Let me try a different approach...")
                
                # Fallback simple response
                try:
                    simple_response = llm.invoke([
                        {"role": "system", "content": "You are a supportive therapist. Provide a brief, caring response (1-2 sentences only). Do not include <think> tags or reasoning."},
                        {"role": "user", "content": user_input}
                    ])
                    
                    content = get_message_content(simple_response)
                    
                    # Clean up any thinking tags
                    if "<think>" in content:
                        content = content.split("</think>")[-1].strip()
                    
                    print(f"\nTherapist: {content}\n")
                    
                    # Update state manually for fallback
                    state["messages"].append({"role": "user", "content": user_input})
                    state["messages"].append({"role": "assistant", "content": content})
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
                    print("\nTherapist: I'm here to listen and support you. Can you tell me more about what you're experiencing?\n")
                
        except KeyboardInterrupt:
            print("\n\n💙 Session interrupted. Take care!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

if __name__ == "__main__":
    run_enhanced_chatbot()