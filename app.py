import os
import streamlit as st
import streamlit.components.v1 as components
from graph import graphy, initial_state
from agent import (onboarding_agent, router_agent, retrieval_agent, sentiment_agent, recommendation_agent, airing_agent, supervisor_agent, chatterbox_agent, wait_node)

from chatbox import render_main, render_bottom
from floating_agent import render_floating_agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

st.set_page_config(layout="wide")
#--- LOAD CSS
def load_css():
    with open("styles.css", "r") as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#--- FONT
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">', unsafe_allow_html=True,)

load_css()

#--- SESSION STATE 
defaults = {"startenter":    False,
            "panel_open":    False,
            "bottom_open":   False,
            "messages":      [],
            "show_tutorial": True,
            "agent_state":   None,
            "movie_graph":   None,
            "graph_ready":   False,
            "keys_entered":  False,   # Langfuse + OpenAI keys collected
}
# Initialize value in session state
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value #bisa jg tulis st.session_state.key = 'value'

# START SCREEN 
if not st.session_state.startenter:
    st.markdown("""
    <div class="start-screen">
        <div class="pixel-row">▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓</div>
        <div class="start-logo">MOVIE ASSISTANT</div>
        <div class="pixel-row">░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░</div>
        <div class="start-prompt">▶ PRESS ENTER TO START ◀</div>
        <div style="font-family:'Press Start 2P',monospace;font-size:8px;color:#444;margin-top:8px;">
            © 2026 · INSERT COIN · LV.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    def start_app():
        st.session_state.startenter = True
    
    _, mid, _ = st.columns([1.5, 1, 1.5]) #3 columns w/ width 1.5,1,1.5
    with mid:
        st.button("[ ENTER ]", on_click=start_app, shortcut='ENTER', use_container_width=True)
    st.stop()

# LANGGRAPH 
if not st.session_state.graph_ready:
    st.session_state.movie_graph = graphy(
        onboarding_agent=onboarding_agent,
        router_agent=router_agent,
        retrieval_agent=retrieval_agent,
        sentiment_agent=sentiment_agent,
        recommendation_agent=recommendation_agent,
        airing_agent=airing_agent,
        supervisor_agent=supervisor_agent,
        chatterbox_agent=chatterbox_agent,
        wait_node=wait_node,)
    
    if st.session_state.agent_state is None:
        st.session_state.agent_state  = initial_state(
            session_id=st.session_state.get("session_id", "new"))
    st.session_state.graph_ready = True

# LAYOUT   
main_col, side_col = st.columns([5, 1])

# LOAD MAIN CHATBOT 
with main_col:
    render_main()

render_bottom()

# LOAD FLOATING AI AGENT TUTORIAL
render_floating_agent()