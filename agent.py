import os
import json
import re
import time
import base64
import pandas as pd
import mysql.connector
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Optional
from langfuse.langchain import CallbackHandler
from ddgs import DDGS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from graph import AgentState

# LLM Libraries
llm = ChatOpenAI(model='gpt-4o-mini', temperature = 1)

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
client = QdrantClient(url=os.environ["QDRANT_URL"],api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=120)

colections_response = client.get_collections()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant = QdrantVectorStore.from_existing_collection(embedding=embeddings, collection_name='movies',
                                                    url=os.environ["QDRANT_URL"],
                                                    api_key=os.environ['QDRANT_API_KEY'])
# ---Text-to-SQL --------------------------------------------------------------------------------------
sql_conn: Optional[mysql.connector.connection.MySQLConnection] = None
 
def get_sql_connection():
    global sql_conn
    if sql_conn is not None:
        try:
            sql_conn.ping(reconnect=True, attempts=2, delay=1)
            return sql_conn
        except Exception:
            sql_conn = None
    try:
        sql_conn    = mysql.connector.connect(
            host    ='localhost',
            user    ='root',
            password= os.environ.get("MYSQL_PASSWORD"),
            database= 'movierecom',)
        return sql_conn
    except Exception as e:
        print(f"[SQL] Connection failed: {e}")
        return None
 
def read_table_preferences(conn) -> pd.DataFrame:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_preference")
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    cursor.close()
    return pd.DataFrame(rows, columns=cols)
 
def read_table_watch(conn) -> pd.DataFrame:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM watch_history")
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    cursor.close()
    return pd.DataFrame(rows, columns=cols)
 
def execute_query(conn, query: str, params=None):
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    cursor.close()
 
def save_user_preference_from_state(state: 'AgentState') -> pd.DataFrame:
    conn = get_sql_connection()
    if conn is None:
        print("[SQL] Skipping user_preference save — no DB connection.")
        return pd.DataFrame()
 
    user_age: int = state.get('user_age', -1)
    genres_list: list = state.get('preferred_genres', [])
    preferred_genres: str = ', '.join(str(g) for g in genres_list).lower()
    rating_list: list = state.get('age_rating_filter', [])
    age_rating: str = str(rating_list[-1]).lower() if rating_list else 'all ages'
 
    df      = read_table_preferences(conn)
    new_id  = 1 if df.empty else int(df['id'].max()) + 1
    created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
 
    execute_query(conn, 'INSERT INTO user_preference VALUES (%s, %s, %s, %s, %s)',
        (new_id, user_age, preferred_genres, age_rating, created))
    print(f"[SQL] user_preference saved — age={user_age}, genres='{preferred_genres}', rating='{age_rating}'")
    return read_table_preferences(conn)
 
def save_watch_history_from_state(state: 'AgentState') -> pd.DataFrame:
    conn = get_sql_connection()
    if conn is None:
        print("[SQL] Skipping watch_history save — no DB connection.")
        return pd.DataFrame()
 
    airing_results: list = state.get('airing_results', [])
    if not airing_results:
        print("[SQL] Skipping watch_history save — no airing results.")
        return pd.DataFrame()
 
    movie_title: str = str(airing_results[0].get('title', '')).strip().lower()
    if not movie_title:
        print("[SQL] Skipping watch_history save — title is empty.")
        return pd.DataFrame()
 
    rating_list: list = state.get('age_rating_filter', [])
    age_rating: str = str(rating_list[-1]).lower() if rating_list else 'all ages'
 
    df         = read_table_watch(conn)
    new_id     = 1 if df.empty else int(df['id'].max()) + 1
    watch_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
 
    execute_query(conn,'INSERT INTO watch_history VALUES (%s, %s, %s, %s)',
        (new_id, movie_title, age_rating, watch_date))
    print(f"[SQL] watch_history saved — title='{movie_title}', rating='{age_rating}'")
    return read_table_watch(conn)

# --- LOG TOOLS & FUNCTION -------------------------------------------------------------------------------------------
def _langfuse_cb(state):
    return CallbackHandler() #Logging LLM calls for Langfuse

# Add new tool_call in a list
def _log_tool(state: AgentState, tool_name:str, inputs: dict, output: Any) -> list:
    existing = state.get('tool_calls', [])
    return existing + [{'tool': tool_name, 'inputs': inputs, 'output': str(output)[:300]}]    

def age_ratings(age:int) -> str:
    '''
    Map a user's age according to the most permissive content rating they could see
    Ratings follows:
    - Adults (17+): Can see everything including adult content and not rated contents
    - Age 16: Can see everything, but not adults only
    - Age 14-15: Can see everything, but not 16+ or adults only
    - Age 8-13: Parental guidance and all ages content only
    - Age 0-7: Only All ages content
    '''
    if age == -1:
        return age_group == ['all ages']
    if age >= 17:
        age_group = ['all ages', 'parental guidance', '16 years old and above','14 years old and above', '13 years old and above', 'not rated', 'adults only']
    elif age ==16:
        age_group = ['all ages', 'parental guidance', '14 years old and above', '13 years old and above', '16 years old and above']
    elif 14<= age <=15:
        age_group = ['all ages', 'parental guidance', '13 years old and above', '14 years old and above']
    elif 8<= age <=13:
        age_group = ['all ages', 'parental guidance', '13 years old and above']
    elif 5< age <=8:
        age_group = ['all ages', 'parental guidance']
    else:
        age_group = ['all ages']

    return age_group

def scan_uploaded_file(file_obj) -> str: #pdf, doc, img
    if file_obj is None:
        return ''
    fname = file_obj.name.lower()
    raw   = file_obj.read()
    b64   = base64.b64encode(raw).decode('utf-8')

    if  any(fname.endswith(e) for e in ('.jpg', '.jpeg', '.png')):
        mime = 'image/jpeg' if fname.endswith (('.jpg', '.jpeg')) else 'image/png'
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": (f'''Identify any movie titles, posters, actors, directors, genres,
                            or themes visible. Return ≤40 words as a movie search hint.''')},]}]
        return llm.invoke(msg).content.strip()
    
    if fname.endswith('.pdf'):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages[:3])
        except Exception:
            text =''
        if not text.strip():
            return ''
        respond = llm.invoke([SystemMessage(content="Extract movie titles, genres, actors, directors, or themes. Return ≤40 words."),
                           HumanMessage(content=text[:2000]),])
        return respond.content.strip()
    
    if any(fname.endswtih(e) for e in ('.docx', '.doc')):
        try:
            import docx as _docx, io
            doc  = _docx.Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            text = ''
        if not text.strip():
            return ''
        respond = llm.invoke([SystemMessage(content="Extract movie titles, genres, actors, directors, or themes. Return ≤40 words."),
                            HumanMessage(content=text[:2000]),])
        return respond.content.strip()
    return ''

def _is_end(text:str) -> bool:
        return text.strip().upper() == 'END'
def _latest_human(state: AgentState) -> str:
    for msg in reversed(state.get('messages', [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ''

farewell = 'Fare thee well, noble traveller. May thine watchlist be ever plentiful.'
# -- Onboarding agent: ask age and movies that user likes ---------------------------------------------------------------------------------------------
def onboarding_agent(state: AgentState) -> dict:
    '''
    Collects age and movie preferences
    Flow:
    - Returning user (has age + prefs + onboarding_done) -> skip to router_agent
    - No human message -> show greeting -> END (wait)
    - Both age + prefs given -> confirm -> router_agent (continue immediately)
    - Age only -> save age, ask for prefs -> END (wait)
    - Pref only -> save prefs, ask for age -> END (wait)
    - NEITHER -> ask again -> END (wait)
    Partial data (age or prefs) is saved to state between invokes so the
    next human message can complete onboarding without asking again.
    '''
    cb = _langfuse_cb(state)

    # IF END
    latest = _latest_human(state)
    if _is_end(latest):
        return{'conversation_ended': True, 'answer': farewell, 'messages': [AIMessage(content=farewell)], 'next_agent': 'END'}

    # RETURNING USER
    is_returning = (state.get('user_age', -1) != -1 and bool(state.get('preferred_genres')) and state.get('onboarding_done'))
    if is_returning:
        return {'onboarding_done': True, 'next_agent': 'router_agent',}

    # NO HUMAN MESSAGE YET -> show greeting
    if not latest:
        prompt = ChatPromptTemplate.from_messages([SystemMessage(content='''You are a movie recommendation assistant.
                                                                        You speak in a medieval vibe, friendly assistant.
                                                                        Ask the user their age and what kind of movies they enjoy
                                                                        (genres, titles, actors, directors).
                                                                        Keep under 60 words. Be engaging and fun.'''),
                HumanMessage(content='Start onboarding conversation'),])
        response = llm.invoke(prompt.format_messages(), config={'callbacks': [cb]})
        return {'onboarding_done': False,
                'answer': response.content,
                'messages': [AIMessage(content=response.content)],
                'next_agent': 'END',}

    # PARSE LATEST HUMAN MESSAGE
    extract_prompt = ChatPromptTemplate.from_messages([SystemMessage(content='''Extract structured data from the user message.
                                                                Return only valid JSON with exactly these keys:
                                                                age: integer (user age, or -1 if not mentioned)
                                                                preferences: array of strings (genres, titles, actors, directors mentioned)
                                                                Example: "I am 22 and love action and the matrix" -> {"age": 22, "preferences": ["action", "the matrix"]}
                                                                Example: "I am 18" -> {"age": 18, "preferences": []}
                                                                Example: "I like action movies" -> {"age": -1, "preferences": ["action"]}'''),
        HumanMessage(content=latest),])
    
    raw       = llm.invoke(extract_prompt.format_messages(), config={'callbacks': [cb]})
    match     = re.search(r'\{.*\}', raw.content, re.DOTALL) #extract the first JSON object from a string
    extracted = json.loads(match.group()) if match else {'age': -1, 'preferences': []} #load the json's in match

    new_age   = int(extracted.get('age', -1))
    new_prefs = [str(p) for p in extracted.get('preferences', [])]

    # Carry forward partial data from previous invokes
    age   = new_age   if new_age   != -1 else state.get('user_age', -1)
    prefs = new_prefs if new_prefs        else state.get('preferred_genres', [])

    # BOTH age and prefs -> complete onboarding
    if age != -1 and prefs:
        age_rating_ceiling = age_ratings(age)
        confirm_prompt = ChatPromptTemplate.from_messages([SystemMessage(content='''Acknowledge the user age and preferences and ask what do they need in under 30 words. Be enthusiastic and medieval vibe.'''),
                        HumanMessage(content=f'User is {age} and likes {prefs}'),])
        confirm    = llm.invoke(confirm_prompt.format_messages(), config={'callbacks': [cb]})
        tool_calls = _log_tool(state, 'onboarding_extract',
                               {'reply': latest},
                               {'age': age, 'age_rating_ceiling': age_rating_ceiling,
                                'preferences': prefs})
        save_user_preference_from_state({**state,
                                         'user_age': age,
                                         'age_rating_filter': age_rating_ceiling,
                                         'preferred_genres': prefs})
        return {'onboarding_done':   True,
                'user_age':          age,
                'age_rating_filter': age_rating_ceiling,
                'preferred_genres':  prefs,
                'onboarding_answer': confirm.content,
                'answer':            confirm.content,
                'messages':          [AIMessage(content=confirm.content)],
                'tool_calls':        tool_calls,
                'next_agent':        'router_agent',}

    # AGE ONLY -> save age, ask for prefs
    if age != -1 and not prefs:
        fu_prompt = ChatPromptTemplate.from_messages([SystemMessage(content='''The user gave their age but not their movie preferences.
                                                                            Speak in a medieval vibe.
                                                                            Acknowledge their age and ask what kind of movies they enjoy
                                                                            (genres, titles, actors, directors) in <40 words.'''),
            HumanMessage(content=latest),])
        fu = llm.invoke(fu_prompt.format_messages(), config={'callbacks': [cb]})
        return {'onboarding_done': False,
                'user_age':        age,
                'answer':          fu.content,
                'messages':        [AIMessage(content=fu.content)],
                'next_agent':      'onboarding_agent',}

    # PREFS ONLY -> save prefs, ask for age
    if age == -1 and prefs:
        fu_prompt = ChatPromptTemplate.from_messages([SystemMessage(content='''The user gave movie preferences but not their age. Speak in a medieval vibe.
                                                                                Acknowledge their preferences and ask how old they are in <40 words.'''),
            HumanMessage(content=latest),])
        fu = llm.invoke(fu_prompt.format_messages(), config={'callbacks': [cb]})
        return {'onboarding_done':  False,
                'preferred_genres': prefs,
                'answer':           fu.content,
                'messages':         [AIMessage(content=fu.content)],
                'next_agent':       'onboarding_agent',}

    # NEITHER -> ask again
    fu_prompt = ChatPromptTemplate.from_messages([SystemMessage(content='''The user did not mention age or movie preferences. Speak in a medieval vibe.
                                                                         Ask again in a fun way and < 30 words.'''),
        HumanMessage(content=latest),])
    fu = llm.invoke(fu_prompt.format_messages(), config={'callbacks': [cb]})
    return {'onboarding_done': False,
            'answer':          fu.content,
            'messages':        [AIMessage(content=fu.content)],
            'next_agent':      'onboarding_agent',}

# --- Router Agent: Choose between Retrieval or Airing agent ----------------------------------------------------------------------------------------
def router_agent(state: AgentState) -> dict:
    cb = _langfuse_cb(state)
    
    # IF END
    latest = _latest_human(state)
    if _is_end(latest):
        return {'conversation_ended': True, 'answer': farewell,
                'messages': [AIMessage(content=farewell)],
                'next_agent': 'END', 'route': 'retrieval'}

    #user context from onboarding agent
    prefs_genres = ','.join(state.get('preferred_genres', [])) or 'not specified'
    age_filter   = ','.join(state.get('age_rating_filter', [])) or 'not specified'

    file_ctx     = state.get('uploaded_file_context', '')
    file_hint    = f"\nUploaded file context: {file_ctx}" if file_ctx else ""

    user_context = (f''' User age: {state.get('user_age', 'unknown')}
                            Allowed age ratings: {age_filter}
                            Preferred genres: {prefs_genres}
                            Onboarding answer: {state.get('onboarding_answer', '')}
                            Latest user request: {latest}
                            {file_hint}''')
    router_message = [SystemMessage(content=f'''You are a routing agent for a movie recommendation system. Speak in a medieval vibe.
                                                From user's onboarding profile, decide which agent to invoke next:
                                                - 'retrieval' = the user wants personalised movie recommendation
                                                - 'airing' = the user wants to know where to legally watch a specific title in INDONESIA
                                                - 'chatterbox' = user wants to casually discuss movie STORIES / PLOTS / THEMES
                                                        (not actors private lifes, not revenue, not box office)

                                                Rules (apply in order):
                                                1. Specific title + streaming / watch / where  ->  "airing"
                                                2. Wants suggestions / "what should I watch"   ->  "retrieval"
                                                3. File context present + asks for recs        ->  "retrieval"
                                                4. Chatting about plot / story / themes        ->  "chatterbox"
                                                5. Genuinely ambiguous                         ->  "chatterbox"

                                                NO NEED to apply age_rating_filter but pass it to the next agent
                                                Return ONLY valid JSON object with double quotes, no markdown, no explanation with exactly these keys:
                                                    {{"route": "retrieval" | "airing" | "chatterbox"}}'''),
                        HumanMessage(content=user_context),]
    response = llm.invoke(router_message, config={'callbacks': [cb]},)

    #The kind to retrieve
    retrieve_prompt = f'''Classify the user intent to only one of these:
    'exact': asking about a specific movie title, actor or director by name
        (ex.'films about or starred by Jackie Chan' -> {{"mode":"exact", "target":"Jackie Chan", "target_type":"actors"}}
            'the star war series movies for first timer' -> {{"mode": "exact", "target": "star war", "target_type":"MovieName"}}
            'recommend me other movies by Steven Spielberg' -> {{"mode": "exact", "target": "Steven Spielberg", "target_type": "Director"}})
    'similar': wants recommendation similar to something named
        (ex.'I liked disney movies' -> {{"mode": "similar", "target":"disney", "target_type":"MovieName"}}
            'cartoon like zootopia' -> {{"mode": "similar", "target":"Andrew Garfield", "target_type":"actors"}})
    'discover': wants recommendation based on genres/mood ONLY, no specific references
        (ex.'I like action movies' -> {{"mode":"discover", "target":"", "target_type":"none"}})
        Return ONLY valid JSON object with double quotes, no markdown, no explanation with exactly these keys:
        {{"mode": "exact" OR "similar" OR "discover", "target": "<name the specific name mentioned, else empty string>", "target_type": "title" OR "actor" OR "director" OR "none"}}'''

    # Use latest human message for intent classification; fall back to onboarding answer
    rr_resp = llm.invoke([SystemMessage(content=retrieve_prompt),
         HumanMessage(content=latest or state.get('onboarding_answer', ''))],config={'callbacks': [cb]})
    try:
        rr = json.loads(rr_resp.content.strip())
        retrieval_mode   = rr.get('mode', 'discover')
        retrieval_target = rr.get('target', '')
        target_type      = rr.get('target_type', 'none')
    except json.JSONDecodeError:
        retrieval_mode, retrieval_target, target_type = 'discover', '', 'none'

    #Log the user's reply and output (parsed data)
    tool_calls       = _log_tool(state, tool_name='router_llm',inputs={'messages':[m.content for m in router_message]},output=response.content,)
    
    #Parsed data
    raw_text: str           = response.content.strip()
    try:
        parsed: dict        = json.loads(raw_text)
        route: str          = parsed.get("route", "retrieval")
        reason: str         = parsed.get("reason", "")
    except json.JSONDecodeError:
        raw_lower = raw_text.lower()
        if "airing" in raw_lower:
            route = "airing"
        elif "chatterbox" in raw_lower or "story" in raw_lower or "plot" in raw_lower:
            route = "chatterbox"
        else:
            route = "retrieval"
        reason: str         = "Fallback keyword parse(JSON decode failed)"
        parsed: dict        = {"route": route, "reason": reason, "raw": raw_text}

    if route not in ('retrieval', 'airing', 'chatterbox'):
        route               = "chatterbox"
        reason              = 'Unknown route so = chatterbox'

    if route == 'retrieval':
        next_agent = 'retrieval_agent'
    elif route == 'airing':
        next_agent = 'airing_agent'
    else:
        next_agent = 'chatterbox_agent'
    
    return{'route': route,
           'messages': [AIMessage(content=f'{response.content} | {reason}')],
           'retrieval_mode': retrieval_mode,
           'retrieval_target': retrieval_target,
           'target_type': target_type,
           'tool_calls': tool_calls,
           'next_agent': next_agent}

# --- CHATTERBOX_AGENT ------------------------------------------------------------------------------------------------------------------------------------------------
def chatterbox_agent(state: AgentState) -> dict:
    ''' Casual movie-story discussion.
    Allowed: plots, themes, story arcs, world-building, endings
    Forbidden: actors personal life, box office numbers, production gossips
    Ends by routing to wait_node, which terminates this graph run so the next user message reenters router_agent
    '''
    cb = _langfuse_cb(state)
    latest = _latest_human(state)
    file_ctx = state.get('uploaded_file_context', '')
    file_hint    = f"\nUploaded file context: {file_ctx}" if file_ctx else ""
    user_message = f"{latest}{file_hint}"

    prompt = [SystemMessage(content=f'''You are Movi., a bard-like movie enthusiast who speaks in a warm medieval-flavoured style.
                                    You love discussing movie STORIES, PLOTS, THEMES, CHARACTER ARCS, WORLD-BUILDING, and ENDINGS.
                                    Rules:
                                    - ONLY discuss movie/series narrative content
                                    - Never discuss actors or directors private life, salaries, relationships or gossips
                                    - Never metion movis budgets or production money
                                    - If user steers towards forbidden topics, gently redirect to the story
                                    - What you talk about must be age appropriate
                                    - Response less than 100 words
                                    - Close with one open question to invite further discussion.'''),
                HumanMessage(content=user_message),]
    
    respond = llm.invoke(prompt, config={'callbacks': [cb]})
    tool_call   = _log_tool(state, 'chatterbox_llm', {'input': latest}, respond.content[:200])
 
    return {'chatterbox_response': respond.content,
            'answer': respond.content,
            'tool_calls': tool_call,
            'messages': state['messages'] + [AIMessage(content=respond.content, name='chatterbox')],
            'next_agent': 'wait_node'}

# --- WAIT NODE -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def wait_node(state: AgentState) -> dict:
    return {'next_agent': 'router_agent'}


# --- THE THREE COMBO: RETRIEVAL, SENTIMENT, RECOMMENDATION ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Divergence so that it does not recommend that movie only
divergence_threshold = {0: (0,4),
                        1: (5,14),
                        2: (15,9999)}

divergence_info = {0: ('SAFE - stay very close to the user preferences and genres. No surprises'),
                   1: ('STRETCH - include 1-2 picks from adjacent genres or different eras/countries sharing similar moods or themes that the user enjoys'),
                   2: ('WILD - atleast half of the recommendations are suprising: cross-genre picks, international cinema, cult classics or hidden gems that the user unlikely have seen. The user has seen many recommendations so avoid repetition 5 times.')}

def divergence_level(seen_count:int) -> int:
    ''' Return 0,1 or 2 based on how many titles the user has seen'''
    for level, (low, hi) in divergence_threshold.items():
        if low <= seen_count <= hi:
            return level
    return 2 #kalau nilainya gede bgt ex. seen_count nya minus atau 10000

# --- Retrieval agent: retrieve data from Qdrant vector db and Duckduckgo
def retrieval_agent(state: AgentState) -> dict:
    '''
    1. Builds an optimised search query from user's profile and validator feedback
    2. Search data from Qdrant (vector similarity) for semantically matching movies
    3. Searches DuckDuckGo for recent / trending semantically matches the user's query if the asked question is not in Qdrant
    4. Merges, deduplicate and age-filters results
    5. Increments retrieval_attempts and logs all tool calls
    '''
    cb = _langfuse_cb(state)
    attempt     = state.get('retrieval_attempt', 0) + 1
    issues      = state.get('validator_issues', [])
    prefs_genres= ', '.join(state.get('preferred_genres', [])) or 'not specified'
    latest_human_msg = _latest_human(state)
    file_ctx= state.get('uploaded_file_context', '')
    file_hint = f"\nFile context hint: {file_ctx}" if file_ctx else ""
 
    issues_str = '; '.join(issues) if issues else ''

    query_message = [SystemMessage(content=f'''You are a search query builder for a movie vector database.
                                   Given the user profile, write <12 words of search query that will surface the most relevant movies.
                                   Focus on movies titles / directors / actors
                                   Do not include age or rating words.
                                   Return ONLY query string - no quotes, no explanation.'''),
                    HumanMessage(content=f''' Latest user request (highest priority): {latest_human_msg} and {file_hint}
                                Preferred genres / interests (background context): {prefs_genres} 
    Onboarding answer (background context): {state.get('onboarding_answer', '')} {f'previous retrieval issues to avoid: {issues_str}' if issues_str else ""}''')]
    
    query_response  = llm.invoke(query_message, config ={'callbacks':[cb]})
    search_query    = query_response.content.strip().strip('"')

    if state.get('retrieval_mode') == 'exact':
        #FILTER BY METADATA
        qdrant_hits = []
        target      = state.get('retrieval_target', '')
        target_type = state.get('target_type', 'title')
        field_map   = {'title': 'title',
                     'Actors': 'Actors',
                     'Director': 'Director',}
        field = field_map.get(target_type, 'title')

        try:
            qdrant_docs = qdrant.similarity_search(target, k=5, filter={'must': [{'key': field, 'match': {'value': target}}]})
            for doc in qdrant_docs:
                qdrant_hits.append({'title': doc.metadata.get('MovieName', doc.page_content[:60]),
                                    'overview': doc.page_content[:30],
                                    'rating': doc.metadata.get('age_group', ''),
                                    'source': 'qdrant',})
        except Exception:
            print(f'Qdrant exact similarity failed')

    else:
        #FILTER THROUGH SIMILARITY OR DISCOVERY
        qdrant_hits = []
        try:
            qdrant_docs = qdrant.similarity_search(search_query, k=5)
            for doc in qdrant_docs:
                qdrant_hits.append({'title': doc.metadata.get('MovieName', doc.page_content[:60]),
                                    'overview': doc.page_content[:30],
                                    'rating': doc.metadata.get('age_group', ''),
                                    'source': 'qdrant',})
        except Exception as e:
            print(f'Qdrant similarity failed')
          
    # Retrieve from DuckDuckGo
    ddg_hits = []
    tool_calls = state.get('tool_calls', [])
    if len(qdrant_hits) <= 5:
        ddg_query = f'best {prefs_genres}'

        time.sleep(1) #Duckduckgo needs time to search
        with DDGS() as ddgs:
            ddg_results = []

            for ddg_attempt in range(3):
                try:
                    with DDGS() as ddgs:
                        ddg_results = list(ddgs.text(ddg_query, max_results=5, region="wt-wt", backend="lite"))
                    if ddg_results:
                        break
                except Exception as e:
                    print("DDG error:", e)

                time.sleep(1.5)  #Duckduckgo needs time to search

        for r in ddg_results:
            ddg_hits.append({'title': r.get('title', ''),
                            'overview': r.get('body', ''),
                            'rating': '',
                            'source': 'duckduckgo'})
        
        tool_calls = _log_tool({'tool_calls': state.get('tool_calls', [])},
                            'duckduckgo_search',
                            {'query': ddg_query},
                            f'{len(ddg_hits)} hits')
    
    #Merge + deduplicate + age-filter retrieval datas
    allowed_ratings     = set(r.upper() for r in state.get('age_rating_filter', []))
    seen: set[str]      = set()
    merged: list[dict]  = []

    for hit in qdrant_hits + ddg_hits:
        key = re.sub(r'[^a-z0-9]', '', hit['title'].lower())
        if key in seen:
            continue
        seen.add(key)

        hit_rating = hit.get('rating', '').upper()
        if hit_rating and allowed_ratings and hit_rating not in allowed_ratings:
            continue
        merged.append(hit)
    
    if not merged:
        msg = ("Mine eyes have searched far and wide but found no matching scrolls. "
               "Prithee rephrase thy request or try a different title, actor, or genre.")
        return {'retrieval_result': [], 'retrieval_query': search_query,
                'retrieval_attempt': attempt, 'tool_calls': tool_calls,
                'answer': msg,
                'messages': state['messages'] + [AIMessage(content=msg)],
                'next_agent': 'sentiment_agent'}
    
    source_used = list(set(hit.get('source', 'unknown') for hit in merged))
    source_info = ' + '.join(source_used).title() if source_used else 'None'

    new_calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
    tool_calls = state['tool_calls'] + [tc for tc in new_calls if tc not in state.get('tool_calls', [])]

    return{'retrieval_result': merged,
            'retrieval_query': search_query,
            'retrieval_attempt': attempt,
            'retrieval_source': source_info,
            'tool_calls': tool_calls,
            'messages': state['messages'] + [AIMessage(content=f'''
                [Retrieval] attempt {attempt}= {len(merged)} movies found via {search_query}''')],
            'next_agent': 'sentiment_agent',}
                    

#--- Sentiment agent
def sentiment_agent(state: AgentState) -> dict:
    '''Reads onboarding_answer + preferred_genres + seen_titles history.
    return sentiment_tone, sentiment_modifier, diversity_modifier
    sentiment_tone: one-word emotional tone
    sentiment_keywords: 3-6 salient taste phrases
    sentiment_modifier: preference directive for the recommendation prompt
    diversity_modifier: diversity directive saled by (SAFE, STRETCH or WILD) and divergence level 0,1 or 2
    '''

    cb = _langfuse_cb(state)
    issues                  = state.get('validator_issues', [])
    seen_titles: list[str]  = state.get('seen_titles', [])
    seen_count              = len(seen_titles)
    latest                  = _latest_human(state)

    if seen_count <=4:
        divergence_level = 0
        divergence_label = 'SAFE'
        divergence_base = ('stay very close to the user\'s stated preferences and genres. No surprises')
    elif seen_count<=14:
        divergence_level = 1
        divergence_label = 'STRETCH'
        divergence_base = ('include 1-2 picks from adjacent genres or different eras/countries  sharing similar moods or themes that the user enjoys.')
    else:
        divergence_level = 2
        divergence_label = 'WILD'
        divergence_base =('atleast half of the recommendations are suprising: cross-genre picks, international cinema, cult classics or hidden gems that the user unlikely have seen. The user has seen many recommendations so avoid repetition 5 times')

    diversity_modifier  = divergence_base
    issues_block        = (f'Previous recommendation issues to fix: {''.join('- ' + i for i in issues) if issues else ''}')

    sentiment_prompt    = [SystemMessage(content=(f'''You are a sentiment and preference analyser for a movie recommendation system
                                               Analyse the user onboarding answer and return ONLY JSON object with double quotes, no markdown, no explanation with exactly these keys:
                                                tone: string - one word describing emotional tone (ex. excited, nostalgic, adventurous, picky, annoying)
                                                keywords: array - 3-6 salient words/phrases that reveal taste (ex. "plot twists", "strong female lead", "pixar")
                                                modifier: string - one sentence preference directive for the recommendation agent (ex. "I like plot twists and avoid slow-burn dramas"
                                                On a retry, adjust the modifier specifically to fix the listed issues.''')),
                        HumanMessage(content=(f'''User age: ' {str(state.get('user_age'))} Current request (highest priority): {latest}.
                                              Preferred genres (background): {', '.join(state.get('preferred_genres', []))}
                                              Onboarding answer: {state.get('onboarding_answer', '') + issues_block}'''))]
    respond= llm.invoke(sentiment_prompt, config={'callbacks': [cb]})
    match   = re.search(r'\{.*\}', respond.content, re.DOTALL)
    parsed  = json.loads(match.group()) if match else {'tone': 'casual',
                                                      'keywords': state.get('preferred_genres', [])[:6],
                                                      'modifier': f'Recommend movies matching: {', '.join(state.get('preferred_genres', []))}.'}
    
    tool_calls = _log_tool(state, 'sentiment_analysis',{
        'onboarding_answer': state.get('onboarding_answer', ''),
        'divergence_level': divergence_label,
        'seen_count': seen_count,},
        {**parsed, 'diversity_modifier': diversity_modifier[:120]},)
    
    return{'sentiment_tone': parsed.get('tone', 'casual'),
           'sentiment_keywords': parsed.get('keywords', []),
           'diversity_modifier': diversity_modifier,
           'divergence_level': divergence_level,
           'messages': state['messages'] + [AIMessage(content=f'''[Sentiment] tone={parsed.get('tone')}
                                                        divergence={divergence_label} (seen {seen_count} titles){parsed.get('modifier')}
                                                        '''.strip(), name='sentiment')],
           'tool_calls': tool_calls,
           'next_agent': 'recommendation_agent'}

#--- Recommendation agent: convert the retrieved movies to a personalized ranking list from sentimental_agent output
def recommendation_agent(state: AgentState) -> dict:
    '''Convert retrieval_results into a ranked, personalised list.
    sentiment_modifier and diversity_modifier are appended to the system prompt via strin concatenation with no .format() used anywhere
    
    After the LLM responds:
    - Hard age-rating filter runs as a safety net
    - Filter seen_titles so that the next session divergence level is computed correctly.'''
    
    cb = _langfuse_cb(state)
    attempt             = state.get('recommendation_attempts', 0) + 1
    issues              = state.get('validator_issues', [])
    modifier            = state.get('sentiment_modifier', 'Match the user stated preferences')
    diversity_modifier  = state.get('diversity_modifier', 'Stay close to stated preferences')
    genres              = ', '.join(state.get('preferred_genres', [])) or 'not specified'
    age_ratings         = ', '.join(state.get('age_rating_filter', [])) or 'all ages'
    results             = state.get('retrieval_result', [])
    seen_titles         = state.get('seen_titles', [])
    filtered_results    = results
    movies_block        = '\n'.join(f"{i+1}. {r['title']} [{r.get('rating', '?')}] - {r.get('overview', '')[:150]}"
                            for i, r in enumerate(filtered_results))
    prompt = ''' You are a movie recommendation agent.
    Convert retrieved movies into persoalised, ranked list that best matches the user age, preferences, emotional tone and any validator feedback.
    Rules:
    - Return JSON array of objects with double quotes, no markdown, no explanation with exactly these keys:
        {{"rank": int, "title": str, "source": str}}
        - Rank 1 = best match. Includes 3-7 movies
        - Never include movies who age_rating is not in the age_rating category
        - Sometimes recommend any title listed under 'Already seen'AIMessage
        - Reason must be directly reference the user stated preferences or tone
    '''
    recommendation_prompt = (f'''{prompt} Preference directive(what the user wants): {modifier}
                                Diversity directive" {diversity_modifier}''')
    issues_block = (f'Validator feedback from previous attempt (fix these): {chr(10).join('- ' + i for i in issues)}' if issues else'')
    user_content = f'''User age: {state.get('user_age')}
                    Allowed age ratings: {age_ratings}
                    Preferred genres / interests: {genres}
                    Sentiment tone: {state.get('sentiment_tone', 'casual')}
                    Sentiment keywords: {', '.join(state.get('sentiment_keywords', []))}
                    Divergence level: {['SAFE', 'STRETCH', 'WILD'][state.get('divergence_level', 0)]}
                    {issues_block}
                    Retrieved movies to rank:
                    {movies_block}
                    '''
    respond      = llm.invoke([SystemMessage(content=recommendation_prompt),
                           HumanMessage(content=user_content)], config={'callbacks': [_langfuse_cb(state)]},)
    
    match = re.search(r'\[.*\]', respond.content, re.DOTALL)
    try:
        recs = json.loads(match.group()) if match else []
    except json.JSONDecodeError:
        recs = []

    allowed = set(r.upper() for r in state.get('age_rating_filter', []))
    if allowed:
        recs= [r for r in recs
                if not r.get('age_rating') or r['age_rating'].upper() in allowed]
        
    for i, rec in enumerate(recs):
        rec['rank'] = i + 1

    new_normalised      = [ re.sub(r'[^a-z0-9]', '', r['title'].lower())
                            for r in recs]
    
    updated_seen_titles = seen_titles + [t for t in new_normalised if t not in set(seen_titles)]

    tool_calls = _log_tool(state, 'recommendation_llm',
        {'attempt': attempt,
            'pool_size': len(filtered_results),
            'seen_titles_count': len(seen_titles),
            'divergence': ['SAFE', 'STRETCH', 'WILD'][state.get('divergence_level', 0)],},
        str(len(recs)) + ' recommendations generated',)
 
    rec_summary = ' | '.join(str(r['rank']) + '. ' + r['title'] for r in recs[:5])
    
    return {'recommendations': recs,
            'recommendation_attempts': attempt,
            'seen_titles': updated_seen_titles,
            'tool_calls': tool_calls,
            'next_agent': 'supervisor_agent',
            'messages': state['messages'] + [AIMessage(content=('[Recommendation] attempt ' + str(attempt) + ' | '
                    'divergence=' + ['SAFE', 'STRETCH', 'WILD'][state.get('divergence_level', 0)] + ' | '
                    + rec_summary), name='recommendation',)],}

LEGAL_PLATFORMS = { #mostlikely will return netflix because python 3.7+ preserves insertion order
        'netflix': 'Netflix',
        'disney+': 'Disney+',
        'disney plus': 'Disney+',
        'vidio': 'Vidio',
        'mola': 'Mola',
        'bioskop online': 'Bioskop Online',
        'amazon prime': 'Amazon Prime Video',
        'prime video': 'Amazon Prime Video',
        'apple tv': 'Apple TV+',
        'hbo go': 'HBO Go',
        'vidio': 'Vidio',
        'max': 'Max',
        'catchplay': 'CatchPlay+',
        'viu': 'Viu',
        'iflix': 'iFlix',
        'youtube': 'YouTube',
        'google play': 'Google Play Movies',
        'itunes': 'iTunes',}

#--- Airing agent:
def airing_agent(state: AgentState) -> dict:
    '''
    Searches for legal streaming site of a specific movie/actor/director in Indonesia only via DuckDuckGo.

    Flow:
    1. Build a targeted DDG query from retrieval target + target_type
    2. Search DDG with retry logic
    3. Filter to known legal Indonesian streaming site only
    4. Validate retry and validator issues to refine query
    5. Feeds to validator agent to check they are legal, age appropriate and atleast 1 movie is found
    '''

    cb = _langfuse_cb(state)
    attempt = state.get('airing_attempt', 0) + 1
    issues  = state.get('validator_issues', [])
    target  = state.get('retrieval_target', state.get('onboarding_answer', ''))
    target_type = state.get('target_type', 'title')

    base_query_map = {
    "title": f"{target} streaming in Indonesia which is legal to watch online",
    "actor": f"{target} movies streaming in Indonesia which is legal to watch online",
    "director": f"{target} films streaming in Indonesia which is legal to watch online",}
    base_query = base_query_map.get(target_type, target + ' streaming Indonesia')

    if issues:
        issue_text = ' '.join(issues).lower()
        if 'non-legal' in issue_text or 'unrecognised' in issue_text:
            base_query = target + ' Netflix OR Vidio OR "Disney+" OR "Amazon Prime" Indonesia'
        elif 'no airing results' in issue_text or 'too narrow' in issue_text:
            base_query = target + ' where to watch in Indonesias online streaming'

    ddg_results = []
    time.sleep(1)
    for ddg_attempt in range(3):
        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(
                    base_query,
                    max_results=8,
                    region='id-id',
                    backend='lite',))
            if ddg_results:
                break
        except Exception as e:
            print('DDG airing error:', e)
        time.sleep(1.5)
    
    tool_calls = _log_tool(state, 'airing_duckduckgo',{'query': base_query, 'attempt': attempt}, str(len(ddg_results)) + ' raw results',)
    
    airing_results: list[dict] = []
 
    for r in ddg_results:
        title_text = r.get('title', '')
        body_text  = r.get('body',  '')
        url        = r.get('href',  '')
        combined   = (title_text + ' ' + body_text).lower()
 
        detected_platform = None
        for keyword, platform_name in LEGAL_PLATFORMS.items():
            if keyword in combined:
                detected_platform = platform_name
                break
 
        if not detected_platform:
            continue
 
        if any(w in combined for w in ['available', 'watch now', 'stream now', 'now streaming']):
            availability = 'Available'
        elif any(w in combined for w in ['not available', 'unavailable', 'coming soon']):
            availability = 'Not available'
        else:
            availability = 'Check platform'
 
        airing_results.append({'title':        target,
                                'platform':     detected_platform,
                                'url':          url,
                                'availability': availability,
                                'snippet':      body_text[:200],})
 
    seen_platforms: set[str] = set()
    deduped: list[dict] = []
    for r in airing_results:
        if r['platform'] not in seen_platforms:
            seen_platforms.add(r['platform'])
            deduped.append(r)
 
    new_calls       = tool_calls if isinstance(tool_calls, list) else [tool_calls]
    tool_calls  = state.get('tool_calls', []) + [tc for tc in new_calls if tc not in state.get('tool_calls', [])]
 
    platform_summary= (', '.join(r['platform'] for r in deduped)
        if deduped else 'no legal platforms found')
 
    return {
        'airing_results': deduped,
        'airing_attempt': attempt,
        'tool_calls':     tool_calls,
        'next_agent':     'supervisor_agent',
        'messages': state['messages'] + [AIMessage(content=f'[Airing] attempt {attempt} | {len(deduped)} platforms | {platform_summary}', name='airing')]}


#--- Supervisor agent: validate movie list. Age appropriate content, validate answer is already according to the query
def supervisor_agent(state: AgentState) -> dict:
    f'''Validates output from either pipeline:
 
    Retrieval pipeline  -> validates recommendations from recommendation_agent
    Airing pipeline     -> validates platform results from airing_agent
 
    Checks performed:
      1. Completeness    — minimum number of results returned
      2. Age-rating      — no result exceeds age_rating_filter
      3. Seen-titles     — no already-seen title slipped through
      4. Relevance       — LLM checks results match the user request
                           (only runs if checks 1-3 all pass)
      5. Airing legality — platforms are legal Indonesian services only
 
    target escalation (first issue wins):
      retrieval problems  -> target = "retrieval"
      recommendation problems -> target = "sentiment"
    '''
    cb    = _langfuse_cb(state)
    route = state.get('route', 'retrieval')
 
    issues: list[str] = []
    target = 'done'
    recs = state.get('recommendations', [])
    allowed_ratings = set(r.upper() for r in state.get('age_rating_filter', []))
    parsed = {'approved': False, 'issues': []}
    
    # TOO little recommendation
    if len(recs) < 3:
        issues.append(f'''Too few recommendations returned ({str(len(recs))}. Retrieval likely returned poor results - broaden the search query.''')
        target = 'retrieval'

    #Age rating not the same
    violating = [r['title'] for r in recs if r.get('age_rating') and r.get('age_rating', '').upper() not in allowed_ratings]
    if violating:
        issues.append(f'''Age-rating violation: {', '.join(violating)} exceed allowed ratings '
                        ({', '.join(allowed_ratings)}). Fix in recommendation ranking.''')
        if target == 'done':
            target = 'sentiment'
 
    if not issues:
        retrieval_mode   = state.get('retrieval_mode',   'discover')
        retrieval_target = state.get('retrieval_target', '')
        prefs            = ', '.join(state.get('preferred_genres', []))
        rec_titles       = ', '.join(r.get('title', '') for r in recs)
        relevance_focus  = (f'''The user asked specifically about: {retrieval_target}.
                            ALL recommendations must relate directly to this entity.'''
                            if retrieval_mode == 'exact'
                            else f'The user prefers: {prefs}.')

        val_prompt       = [SystemMessage(content=f''' You are a movie recommendation validator.
                                    Check if the recommendations are relevant to the user's request.
                                    Return ONLY valid JSON: {{"approved": true|false, "issues": ["issue1", "issue2"]}}
                                    If approved, issues must be an empty array.
                                    Be strict about exact-mode requests - if the user asked for a specific 
                                    title/actor/director, All recommendations must relate directly to that entity.'''.strip()),
                    HumanMessage(content=f'''{relevance_focus} Recommended titles: {rec_titles} Sentiment modifier applied: {state.get('sentiment_modifier', '')}'''.strip()),]
        val_respond = llm.invoke(val_prompt, config={'callbacks': [cb]})

        match  = re.search(r'\{.*\}', val_respond.content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = {'approved': False, 'issues': ['Validator LLM failed']}

    if route == 'airing':
        airing_results: list[dict] = state.get('airing_results', [])
 
        if len(airing_results) < 1:
            issues.append(f'''No airing results found. DuckDuckGo search may be too narrow - retry with broader query.'''.strip())
            target = 'retrieval'
 
        illegal = [r.get('platform', '') for r in airing_results
            if r.get('platform') and not any(p in r['platform'].lower() for p in LEGAL_PLATFORMS)]
        if illegal:
            issues.append(f'''Non-legal or unrecognised platforms: {', '.join(illegal)}.
                            Only show verified legal Indonesian streaming platforms.''')
            if target == 'done':
                target = 'retrieval'

    approved = len(issues) < 3
    if approved:
        target = 'done'
        if route == 'airing':
            save_watch_history_from_state(state)
 
    tool_calls = _log_tool(state, 'validator',
        {'route':    route,
        'attempt':   state.get('retrieval_attempts', 0),
        'rec_count': len(state.get('recommendations', [])),},
        {'validator_approved': approved, 'issues': issues, 'target': target},)
 
    status = 'APPROVED' if approved else 'REJECTED -> retry ' + target
 
    return {'validator_approved': approved,
            'validator_target': target,
            'validator_issues': parsed.get('issues', []),
            'tool_calls': tool_calls if isinstance(tool_calls, list) else [tool_calls],
            'messages': state['messages'] + [AIMessage(
                 content=f'''[Validator] {status} | issues: {str(issues if issues else 'none')}''',name='validator')],}
