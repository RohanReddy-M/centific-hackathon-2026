import streamlit as st
import requests

st.set_page_config(page_title="Aegis.ai", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("Aegis.ai Status")
    try:
        r = requests.get("http://api:8000/health", timeout=3)
        data = r.json()
        st.success("All Agents Ready")
        st.info(f"Knowledge base: {'Loaded' if data.get('knowledge_base') else 'Empty'}")
    except:
        st.error("Cannot connect to agents")

st.title("Aegis.ai — Multi-Agent Intelligence System")
tab1, tab2 = st.tabs(["Run Agent Task", "Analytics"])

with tab1:
    task = st.text_area("Describe your task:",
        placeholder="Example: Why is medicine expiry tracking important?",
        height=100)
    if st.button("Run 4-Agent Pipeline", type="primary"):
        if task:
            with st.spinner("Running agents... Planner → Executor → Judge → Reporter"):
                try:
                    r = requests.post("http://api:8000/run-agent",
                        json={"task": task}, timeout=120)
                    data = r.json()
                    score = data.get("score", 0)
                    retries = data.get("retries", 0)
                    latency = data.get("latency_ms", 0)
                    st.success(f"Done in {latency}ms | Quality: {score:.1f}/10 | Retries: {retries}")
                    st.subheader("Agent Report")
                    st.text_area("", value=data.get("report", ""), height=400, disabled=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Quality Score", f"{score:.1f}/10")
                    c2.metric("Retries", retries)
                    c3.metric("Response Time", f"{latency}ms")
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({"task": task, "score": score})
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a task")

with tab2:
    st.header("Agent Analytics")
    hist = st.session_state.get("history", [])
    if hist:
        scores = [h["score"] for h in hist]
        c1, c2, c3 = st.columns(3)
        c1.metric("Tasks Run", len(hist))
        c2.metric("Avg Quality", f"{sum(scores)/len(scores):.1f}/10")
        c3.metric("Below 7", sum(1 for s in scores if s < 7))
    else:
        st.info("Run agent tasks to see analytics")