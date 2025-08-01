import streamlit as st
import os
import json
import itertools
from openai import OpenAI

# --- Streamlit App ---
st.set_page_config(page_title="Deep Research Clone", layout="wide")
st.title("🔎 Deep Research Clone with GPT-4")

# --- Retrieve OpenAI API Key from Streamlit Secrets ---
try:
    openai_api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("OpenAI API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml or set it in the Streamlit Cloud secrets UI.")
    st.stop()

os.environ['OPENAI_API_KEY'] = openai_api_key
client = OpenAI()

# --- Session State ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'topic' not in st.session_state:
    st.session_state.topic = ''
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'goal' not in st.session_state:
    st.session_state.goal = ''
if 'queries' not in st.session_state:
    st.session_state.queries = []
if 'collected' not in st.session_state:
    st.session_state.collected = []
if 'report' not in st.session_state:
    st.session_state.report = ''

MODEL = "gpt-4.1"
MODEL_MINI = "gpt-4.1-mini"
TOOLS = [{"type": "web_search"}]
developer_message = """
You are an expert Deep Researcher.
You provide complete and in depth research to the user.
"""

def ask_clarifying_questions(topic):
    prompt = f"""
Ask 5 numbered clarifying question to the user about the topic: {topic}.
The goal os the questions is to understand the intended purpose of the research.
Reply only with the questions
"""
    clarify = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        instructions=developer_message
    )
    questions = clarify.output[0].content[0].text.split("\n")
    return [q for q in questions if q.strip()]

def get_goal_and_queries(questions, answers, topic):
    prompt = f"""
Using the user answers {answers} to  questions {questions}, write a goal sentence and 5 web search queries for the research about {topic}
Output: A json list of the goal and the 5 web search queries that will reach it.
Format: {{\"goal\": \"...\", \"queries\": [\"q1\", ....]}}
"""
    goal_and_queries = client.responses.create(
        model=MODEL,
        input=prompt,
        instructions=developer_message
    )
    plan = json.loads(goal_and_queries.output[0].content[0].text)
    return plan['goal'], plan['queries']

def run_search(q):
    web_search = client.responses.create(
        model=MODEL,
        input=f"search: {q}",
        instructions=developer_message,
        tools=TOOLS
    )
    return {
        "query": q,
        "resp_id": web_search.output[1].id,
        "research_output": web_search.output[1].content[0].text
    }

def evaluate(collected, goal):
    review = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": f"Research goal: {goal}"},
            {"role": "assistant", "content": json.dumps(collected)},
            {"role": "user", "content": "Does this information will fully satisfy the goal? Answer Yes or No only."}
        ],
        instructions=developer_message
    )
    return "yes" in review.output[0].content[0].text.lower()

def get_more_queries(collected, goal, prev_id):
    more_searches = client.responses.create(
        model=MODEL,
        input=[
            {"role": "assistant", "content": f"Current data: {json.dumps(collected)}"},
            {"role": "user", "content": f"This has not met the goal: {goal}. Write 5 other web searchs to achieve the goal"}
        ],
        instructions=developer_message,
        previous_response_id=prev_id
    )
    text = more_searches.output[0].content[0].text
    if not text:
        st.error("OpenAI returned an empty response when generating more queries. Please check your API key, usage limits, or try again.")
        st.stop()
    try:
        return json.loads(text)
    except Exception as e:
        st.error(f"OpenAI returned invalid JSON: {text}\nError: {e}")
        st.stop()

def write_report(goal, collected):
    report = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": (f"Write a complete and detailed report about research goal: {goal} "
                                                "Cite Sources inline using [n] and append a reference "
                                                "list mapping [n] to url")},
            {"role": "assistant", "content": json.dumps(collected)}
        ],
        instructions=developer_message
    )
    return report.output[0].content[0].text

# --- Streamlit App Flow ---
if st.session_state.step == 0:
    st.subheader("Step 1: Enter Research Topic")
    with st.form("topic_form"):
        topic = st.text_input("What topic do you want to research?")
        submitted = st.form_submit_button("Next")
    if submitted and topic:
        st.session_state.topic = topic
        st.session_state.step = 1
        st.rerun()

elif st.session_state.step == 1:
    st.subheader("Step 2: Clarifying Questions")
    if not st.session_state.questions:
        st.session_state.questions = ask_clarifying_questions(st.session_state.topic)
    with st.form("clarify_form"):
        answers = []
        for i, q in enumerate(st.session_state.questions):
            answers.append(st.text_input(f"Q{i+1}: {q}"))
        submitted = st.form_submit_button("Next")
    if submitted and all(a.strip() for a in answers):
        st.session_state.answers = answers
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("Step 3: Generating Research Plan")
    if not st.session_state.goal or not st.session_state.queries:
        goal, queries = get_goal_and_queries(
            st.session_state.questions,
            st.session_state.answers,
            st.session_state.topic
        )
        st.session_state.goal = goal
        st.session_state.queries = queries
    st.write(f"**Research Goal:** {st.session_state.goal}")
    st.write("**Initial Web Search Queries:**")
    for q in st.session_state.queries:
        st.write(f"- {q}")
    if st.button("Start Research"):
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.subheader("Step 4: Research in Progress")
    collected = st.session_state.collected
    queries = st.session_state.queries
    goal = st.session_state.goal
    with st.spinner("Running web searches and evaluating results..."):
        for _ in itertools.count():
            for q in queries:
                collected.append(run_search(q))
            if evaluate(collected, goal):
                break
            # If not enough, get more queries
            queries = get_more_queries(collected, goal, None)  # previous_response_id not tracked here
        st.session_state.collected = collected
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.subheader("Step 5: Final Research Report")
    if not st.session_state.report:
        with st.spinner("Writing final report..."):
            st.session_state.report = write_report(
                st.session_state.goal,
                st.session_state.collected
            )
    st.markdown(st.session_state.report, unsafe_allow_html=True)
    st.success("Research complete!")
    if st.button("Start New Research"):
        for key in ["step", "topic", "questions", "answers", "goal", "queries", "collected", "report"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else ''
        st.session_state.step = 0
        st.rerun() 