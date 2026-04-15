import os, time, json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json

load_dotenv()
app = FastAPI(title="Aegis.ai Agent System")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
vectorstore = None

REQUESTS = Counter("agent_requests_total", "Total requests", ["status"])
LATENCY = Histogram("agent_latency_seconds", "Response time")
QUALITY = Histogram("agent_quality_score", "Quality scores")

class State(TypedDict):
    task: str
    plan: List[str]
    result: str
    score: float
    feedback: str
    retries: int
    report: str

def planner(state):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"Break this task into 3 steps. Numbered list only.\nTask: {state['task']}"}],
        temperature=0.1)
    state["plan"] = [resp.choices[0].message.content]
    return state

def executor(state):
    feedback = f"\nIMPROVE BASED ON: {state['feedback']}" if state["feedback"] else ""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"Complete this task.\nPlan: {state['plan']}\nTask: {state['task']}{feedback}"}],
        temperature=0.2)
    state["result"] = resp.choices[0].message.content
    return state

def judge(state):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"""Rate this answer. Return ONLY valid JSON.
{{"accuracy":0,"completeness":0,"clarity":0,"overall":0,"feedback":""}}
Task: {state['task']}
Answer: {state['result']}"""}],
        temperature=0.1)
    scores = json.loads(resp.choices[0].message.content)
    state["score"] = scores["overall"]
    state["feedback"] = scores["feedback"]
    return state

def reporter(state):
    state["report"] = f"""TASK: {state['task']}

AGENT TRACE:
1. PLANNER created the plan
2. EXECUTOR wrote the answer (attempts: {state['retries']+1})
3. JUDGE scored it {state['score']}/10
4. REPORTER formatted this output

FINAL ANSWER:
{state['result']}

QUALITY SCORE: {state['score']}/10"""
    return state

def should_retry(state):
    if state["score"] < 7 and state["retries"] < 2:
        state["retries"] += 1
        return "executor"
    return "reporter"

def build_graph():
    wf = StateGraph(State)
    wf.add_node("planner", planner)
    wf.add_node("executor", executor)
    wf.add_node("judge", judge)
    wf.add_node("reporter", reporter)
    wf.set_entry_point("planner")
    wf.add_edge("planner", "executor")
    wf.add_edge("executor", "judge")
    wf.add_conditional_edges("judge", should_retry)
    wf.add_edge("reporter", END)
    return wf.compile()

agent_app = build_graph()

@app.on_event("startup")
async def startup():
    global vectorstore
    if os.path.exists("documents/"):
        docs = []
        for f in os.listdir("documents/"):
            path = f"documents/{f}"
            if f.endswith(".txt"):
                d = TextLoader(path).load()
                docs.extend(d)
        if docs:
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50).split_documents(docs)
            vectorstore = FAISS.from_documents(
                chunks, FakeEmbeddings(size=384))
            print(f"Knowledge base ready: {len(chunks)} chunks")
    else:
        print("No documents folder")

@app.post("/run-agent")
async def run_agent(request: Request):
    start = time.time()
    body = await request.json()
    task = body.get("task", "")
    if not task:
        return JSONResponse(400, {"error": "task required"})
    try:
        result = agent_app.invoke({
            "task": task, "plan": [], "result": "",
            "score": 0.0, "feedback": "", "retries": 0, "report": ""
        })
        latency = time.time() - start
        LATENCY.observe(latency)
        REQUESTS.labels("success").inc()
        QUALITY.observe(result["score"])
        return {
            "task": task,
            "report": result["report"],
            "score": result["score"],
            "retries": result["retries"],
            "latency_ms": round(latency * 1000)
        }
    except Exception as e:
        REQUESTS.labels("error").inc()
        return JSONResponse(500, {"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "ok", "knowledge_base": vectorstore is not None}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)