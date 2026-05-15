import streamlit as st

def render_floating_agent():
    st.markdown('''
    <div class="ai-agent" id="ai-agent-btn">
        <div class="ai-agent-portrait">🤖</div>
                
        🎬 MOVIE ASSISTANT 🎬
            I recommend movies based on
            your age & preferences,
            help you find where to watch
            them legally in Indonesia or
            just for a chat.
            Start typing in the chatbox!
            If you want to end just type
            'end' or just close the window
    </div>
    ''', unsafe_allow_html=True)