from typing import Annotated, Literal, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

#--- Define Schema
class AgentState(TypedDict):
    onboarding_done: bool
    user_age: int
    age_rating_filter: list[str]
    preferred_genres: list[str]
    onboarding_answer: str
    answer: str
    tool_calls: Annotated[list, lambda x, y: y]
    messages: Annotated[list, add_messages]
    session_id: Optional[str]
    route: Literal['retrieval', 'airing', 'chatterbox']
    next_agent: str
    retrieval_mode: Literal['exact', 'similar', 'discover']
    retrieval_target: str
    target_type: str
    retrieval_result: Annotated[list, lambda x, y: y]
    retrieval_attempt: int
    retrieval_query: str
    retrieval_source: Annotated[list, lambda x,y:y]
    sentiment_tone: str
    sentiment_keywords: Annotated[list, lambda x, y: y]
    sentiment_modifier: str
    diversity_modifier: str
    divergence_level: int
    seen_titles: Annotated[list, lambda x, y: y]
    recommendations: Annotated[list, lambda x, y: y]
    recommendation_attempts: int
    airing_results: Annotated[list, lambda x, y: y]
    airing_attempt: int
    validator_approved: bool
    validator_target: str
    validator_issues: Annotated[list, lambda x, y: y]
    uploaded_file_context: str
    conversation_ended: bool
    chatterbox_response: str

#--- Functions ---------------------------------------------------------------------------------------------------------------------

def get_last_user_message(state: AgentState) -> str:
    messages = state.get('messages', [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def build_query_message(state: AgentState) -> list[BaseMessage]:
    prefs_genres      = state.get('preferred_genres', [])
    onboarding_answer = state.get('onboarding_answer', '')
    last_user_message = get_last_user_message(state)
    issues            = state.get('validator_issues', [])
    
    user_context = f"""Preferred genres / interests: {prefs_genres}
                    Onboarding answer: {onboarding_answer}
                    Last user message: {last_user_message}"""
    
    if issues:
        user_context += f"\nprevious retrieval issues to avoid: {'; '.join(issues)}"
    
    query_message = [{'role': 'system',
                      'content': [{'type': 'text',
                                   'text': 'You are a search query builder. You are a search query builder. Focus on recent user answer {user_context} and preferance{prefs_genres}. Return <12 words'}]},
                    {'role': 'user',
                     'content': [{'type': 'text', 'text': user_context}]}]
    
    return query_message

#--- IMPORTANT: All router functions must return strings, NOT DICTIONARIES!
def onboard_router(state: AgentState) -> str:
    """After onboarding, go to router or end"""
    messages = state.get('messages', [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            last_content = last_msg.content.lower().strip()
            if last_content == 'end':
                return 'end'
    
    if state.get('onboarding_done'):
        return 'router_agent'
    return 'end'

def router_retrieve_airing(state: AgentState) -> str:
    if state.get('conversation_ended') == True:
        return 'end'
    route = state.get('route', 'chatterbox')
    if route == 'retrieval':
        return 'retrieval_agent'
    elif route == 'airing':
        return 'airing_agent'
    else:
        return 'chatterbox_agent'

def chatterbox_to_wait(state:AgentState) -> str:
    return 'wait_node'

def validator_edge(state: AgentState, ) -> Literal['retrieval_agent', 'sentiment_agent', 'end']:
    if state.get("validator_approved", False):
        return "end"
    route = state.get('route', 'retrieval')
    if route == 'airing':
        if state.get('airing_attempt', 0) < 2:
            return 'airing_agent'
        return 'end'
    
    target = state.get("validator_target", "done")
 
    if target == "retrieval" and state.get("retrieval_attempts", 0) < 2:
        return "retrieval_agent"
 
    if target == "sentiment" and state.get("recommendation_attempts", 0) < 2:
        return "sentiment_agent"
 
    return "end"

# -------------------------------------------------------------------------------------------------------------------------

def graphy(onboarding_agent, router_agent, retrieval_agent, sentiment_agent, 
           recommendation_agent, airing_agent, supervisor_agent, chatterbox_agent, wait_node=None) -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("onboarding_agent", onboarding_agent)
    g.add_node("router_agent", router_agent)
    g.add_node("retrieval_agent", retrieval_agent)
    g.add_node("sentiment_agent", sentiment_agent)
    g.add_node("recommendation_agent", recommendation_agent)
    g.add_node("airing_agent", airing_agent)
    g.add_node("supervisor_agent", supervisor_agent)
    g.add_node("chatterbox_agent", chatterbox_agent)
    g.add_node('wait_node', wait_node)
    
    g.add_edge(START, "onboarding_agent")
    
    g.add_conditional_edges("onboarding_agent", onboard_router,
        {"router_agent": "router_agent",
            "end": END})
    
    g.add_conditional_edges("router_agent", router_retrieve_airing,
                            {"retrieval_agent":  "retrieval_agent",
                             "airing_agent":     "airing_agent",
                             "chatterbox_agent": "chatterbox_agent",
                             "end":              END})
    
    g.add_edge("retrieval_agent", "sentiment_agent")
    g.add_edge("sentiment_agent", "recommendation_agent")
    g.add_edge("recommendation_agent", "supervisor_agent")
    g.add_conditional_edges("supervisor_agent", validator_edge, {
    "retrieval_agent":  "retrieval_agent",
    "sentiment_agent":  "sentiment_agent",
    "airing_agent":     "airing_agent", 
    "end":              END})
    g.add_edge("airing_agent", "supervisor_agent")
    g.add_edge("chatterbox_agent", 'wait_node')
    g.add_edge('wait_node', END)
    
    return g.compile()

def run_graph(state: AgentState, movie_graph) -> dict:
    try:
        cb = CallbackHandler()
        return movie_graph.invoke(state, config={'callbacks': [cb]})
    except Exception as e:
        print(f"Error with callback: {e}")
        return movie_graph.invoke(state)

def initial_state(session_id: str = 'new') -> AgentState:
    return {"onboarding_done": False,
        "user_age": -1,
        "age_rating_filter": [],
        "preferred_genres": [],
        "onboarding_answer": "",
        "answer": "",
        "tool_calls": [],
        "messages": [],
        "session_id": session_id,
        "route": "",
        "next_agent": "onboarding_agent",
        "retrieval_mode": "",
        "retrieval_target": "",
        "target_type": "none",
        "retrieval_result": [],
        "retrieval_attempt": 0,
        "retrieval_query": "",
        "sentiment_tone": "",
        "sentiment_keywords": [],
        "sentiment_modifier": "",
        "diversity_modifier": "",
        "divergence_level": 0,
        "seen_titles": [],
        "recommendations": [],
        "recommendation_attempts": 0,
        "airing_results": [],
        "airing_attempt": 0,
        "validator_approved": False,
        "validator_target": "done",
        "validator_issues": [],
        'uploaded_file_context': "",
        'conversation_ended': False,
        'chatterbox_response': "",}