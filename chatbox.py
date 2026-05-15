import streamlit as st
from langchain_core.messages import HumanMessage
from graph import run_graph

# RPG Bubble Function
def rpg_bubble(role: str, content: str):
    """Render an RPG-style pixel chat bubble."""
    is_user = role == "user"
    wrap_class      = "rpg-bubble-wrap user" if is_user else "rpg-bubble-wrap assistant"
    bubble_class    = "rpg-bubble user" if is_user else "rpg-bubble assistant"
    portrait        = "🧑" if is_user else "🤖"
    name            = "YOU" if is_user else "Movi."
    name_class      = "rpg-name user" if is_user else "rpg-name assistant"
    st.markdown(f"""
    <div class="{wrap_class}">
        <div class="rpg-portrait">{portrait}</div>
        <div class="{bubble_class}">
            <div class="{name_class}">{name}</div>
            <div class="rpg-text">{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)     

# ----------------------------------------------------------------------------------
def _handle_input(user_input:str, uploaded_file=None):
    # Reset all per-request fields so old results never bleed into a new query
    st.session_state.agent_state.update({
        'answer':                  '',
        'recommendations':         [],
        'retrieval_result':        [],
        'airing_results':          [],
        'tool_calls':              [],
        'validator_approved':      False,
        'validator_target':        'done',
        'validator_issues':        [],
        'retrieval_attempt':       0,
        'recommendation_attempts': 0,
        'airing_attempt':          0,
        'route':                   'retrieval',
        'chatterbox_response':     '',
        'uploaded_file_context':   '',})

    if uploaded_file is not None:
        from agent import scan_uploaded_file
        file_ctx = scan_uploaded_file(uploaded_file)
        if file_ctx:
            st.session_state.agent_state['uploaded_file_context'] = file_ctx

    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))

    # LLM LOADING MESSAGE
    loading_messages = [
        "🎬 LOADING CINEMATIC DATA...",
        "🍿 GRABBING POPCORN FROM THE FIELD...",
        "📡 SCANNING THE MULTIVERSE...",
        "🎮 LEVELING UP RECOMMENDATIONS...",
        "🔍 CONSULTING THE MOVIE GODS...",]

    import random
    loading_msg = random.choice(loading_messages)

    with st.spinner(loading_msg):
        result = run_graph(
            st.session_state.agent_state, #return schema datas
            st.session_state.movie_graph,) #return callable objects

    # Update the full agent state
    st.session_state.agent_state.update(result)

    # Priority 1: recommendation results
    answer = ''
    recs = result.get('recommendations', [])
    if recs:
        # '\n\n'.join gives double enter after reason
        answer = '\n\n'.join(f'''Rank: {r['rank']}
                                Title: {r['title']}'''.strip() for r in recs)

    # Priority 2: airing results
    if not answer and result.get("airing_results"):
        airing = result["airing_results"]
        lines  = [f'''Platform: {r['platform']}
                    Availability: {r['availability']}
                    [Link]({r['url']})'''.strip() for r in airing]
        answer = f'''Whither thou shalt direct thine eyes to {result.get("retrieval_target", "")} in Indonesia:
                    {"\n\n".join(lines)}'''

    # Priority 3: chatterbox response
    if not answer and not result.get('airing_results') and not result.get('recommendations'):
        answer = result.get('chatterbox_response', '')

    # Priority 4: plain text
    if not answer:
        answer = result.get('answer', '')

    # Priority 5: fallback
    if not answer:
        answer = "Apologies, Mine eyes have scanned the parchment and found nothing. Speak thy desire in a different fashion."
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})

def render_main():
# MAIN CHATBOT 
    st.markdown('<div class="main-padding">', unsafe_allow_html=True)

    # Pixel deco strip + title
    st.markdown('<div class="pixel-border-top"></div>', unsafe_allow_html=True)
    st.markdown('<div class="rpg-title">⚔ MOVIE ASSISTANT ⚔</div>', unsafe_allow_html=True)
    st.markdown('<div class="rpg-subtitle">READY MOVIE GOERS<span class="cursor-blink">_</span></div>',
            unsafe_allow_html=True,)
    st.markdown('<div class="pixel-border-top"></div>', unsafe_allow_html=True)

    # Chat
    st.markdown('<div id="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        rpg_bubble(msg['role'], msg['content'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input('TYPE YOUR MESSAGE HERE', accept_file=True, file_type=['jpg', 'jpeg', 'png', 'pdf', 'docx', 'doc'],)
    if user_input:
        text = user_input.text if hasattr(user_input, 'text') else str(user_input)
        _handle_input(text)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# BOTTOM PANEL WITH BIGGER FONT 
def render_bottom():
    def toggle_bottom(): st.session_state.bottom_open = not st.session_state.bottom_open #Kalau false knp error ya?

    _, tog_col = st.columns([10, 1])
    with tog_col:
        st.button("≡", on_click=toggle_bottom, key="bottom_toggle")

    if not st.session_state.bottom_open:
        return
 
    agent_state = st.session_state.get("agent_state") or {}
    
    if st.session_state.bottom_open:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        tabs = st.tabs(["HISTORY", "TOOLS", "STATE", "PARSES"])
        with tabs[0]: 
            import re
            display_msgs = st.session_state.get("messages", [])
            recs = agent_state.get('recommendations', [])
            if recs:
                recs_text = '\n'.join(f'{r['rank']}. {r['title']}' for r in recs)
                display_msgs.append({'role': 'assistant', 'content': recs_text})
            
            chat_msgs = [m for m in display_msgs
                if not (m["role"] == "assistant"and re.match(r'\s*\*\*\d+\.', m["content"]))]
            last_6 = chat_msgs[-6:] if len(chat_msgs) >=6 else chat_msgs
            if not last_6:
                st.caption("Silence holds the road.")
            else:
                for m in last_6:
                    colour = "#7dd3fc" if m["role"] == "user" else "#86efac"
                    label  = "YOU" if m["role"] == "user" else "Movi."
                    st.markdown(
                        f'<p style="font-family:\'Press Start 2P\';font-size:9px;color:{colour}">'
                        f'[{label}] {str(m["content"])[:300]}</p>',
                        unsafe_allow_html=True,)

        with tabs[1]: 
            tool_calls = agent_state.get("tool_calls", [])
            if not tool_calls:
                st.caption("Silence holds the road.")
            else:
                for tc in tool_calls[-5:]:
                    st.markdown(
                        f'<p style="font-family:\'Press Start 2P\';font-size:9px;color:#fbbf24">'
                        f'🔧 {tc.get("tool","?")}</p>',
                        unsafe_allow_html=True,)
                    st.json({"inputs": tc.get("inputs", {}), "output": tc.get("output", "")})
            
        with tabs[2]: 
            if not agent_state:
                st.caption("Silence holds the road.")
            else:
                display_state = {
                    "route":                  agent_state.get("route",                ""),
                    "next_agent":             agent_state.get("next_agent",           ""),
                    "retrieval_mode":         agent_state.get("retrieval_mode",       ""),
                    "retrieval_target":       agent_state.get("retrieval_target",      ""),
                    "retrieval_attempt":      agent_state.get("retrieval_attempt",    0),
                    'retrieval_source':       agent_state.get('retrieval_source', None),
                    "recommendation_attempts":agent_state.get("recommendation_attempts", 0),
                    "airing_attempt":         agent_state.get("airing_attempt",       0),
                    "divergence_level":       ["SAFE","STRETCH","WILD"][
                                                agent_state.get("divergence_level", 0)],
                    "seen_titles_count":      len(agent_state.get("seen_titles",      [])),
                    "validator_approved":     agent_state.get("validator_approved",   False),
                    "validator_target":       agent_state.get("validator_target",     ""),
                    "validator_issues":       agent_state.get("validator_issues",     []),}
                st.json(display_state)

        with tabs[3]: 
            recs   = agent_state.get("recommendations",  [])
            airing = agent_state.get("airing_results",   [])
 
            if recs:
                st.markdown(
                    '<p style="font-family:\'Press Start 2P\';font-size:9px;color:#c084fc">'
                    '📝 RECOMMENDATIONS</p>',
                    unsafe_allow_html=True,)
                st.json(recs)
    
            elif airing:
                st.markdown(
                    '<p style="font-family:\'Press Start 2P\';font-size:9px;color:#c084fc">'
                    '📡 AIRING RESULTS</p>',
                    unsafe_allow_html=True,)
                st.json(airing)
    
            else:
                st.caption("Silence holds the road.")
 
    st.markdown('</div>', unsafe_allow_html=True)