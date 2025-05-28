import os
import json
import re
from typing import Dict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Load API key from .env if present ---
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
                break

llm = ChatOpenAI(temperature=0.2, model="gpt-4o")

# --- Load command dataset and build / load FAISS index ---
DATA_PATH = "tooth_chart_commands.jsonl"
INDEX_DIR = "rag_tooth_chart_faiss_index"

def load_commands() -> List[dict]:
    cmds = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            cmds.append(json.loads(line))
    return cmds

if os.path.exists(INDEX_DIR):
    faiss_index = FAISS.load_local(
        INDEX_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True
    )
    commands = load_commands()
else:
    commands = load_commands()
    docs = [Document(page_content=c["text"], metadata={"intent": c["intent"]}) for c in commands]
    faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings())
    faiss_index.save_local(INDEX_DIR)


def retrieve_similar(query: str, k: int = 2):
    results = faiss_index.similarity_search_with_score(query, k=k)
    return results  # each result is (Document, score)


# --- Simple chart state ---
chart_state: Dict[str, List[str]] = {}
current_patient: str | None = None

def clear_tooth(command: str) -> str:
    match = re.search(r"tooth\s+(\d+)", command.lower())
    if not match:
        return "Tooth number not found."
    tooth = match.group(1)
    chart_state.pop(tooth, None)
    return f"Cleared data for tooth {tooth}."

def reset_chart(_: str = "") -> str:
    global chart_state
    chart_state = {}
    return "Chart has been reset."

def view_chart(_: str = "") -> str:
    if not chart_state:
        return "The chart is currently empty."
    return json.dumps(chart_state, indent=2)

def summarize_chart(_: str = "") -> str:
    if not chart_state:
        return "No data to summarize."
    summary = [f"Tooth {t}: {', '.join(statuses)}" for t, statuses in chart_state.items()]
    return "\n".join(summary)

def add_note(command: str) -> str:
    return f"Note added: {command}"

def mark_missing(command: str) -> str:
    match = re.search(r"tooth\s+(\d+)", command.lower())
    if not match:
        return "Tooth number not found."
    tooth = match.group(1)
    chart_state[tooth] = ["MISSING"]
    return f"Marked tooth {tooth} as missing."

def add_restoration(command: str) -> str:
    match = re.search(r"tooth\s+(\d+)", command.lower())
    if not match:
        return "Tooth number not found."
    tooth = match.group(1)
    chart_state.setdefault(tooth, []).append("IMPLANT")
    return f"Implant noted for tooth {tooth}."


def start_exam(name: str) -> str:
    global current_patient, chart_state
    current_patient = name
    chart_state = {}
    return f"Started exam for {name}."


def change_tooth_status(command: str) -> str:
    """Update one or more teeth with a status code.

    Accepted formats:
      - "tooth 12 MOD"
      - "tooth 13 is MOB"
      - "teeth 7 8 9 O"
    """
    command = command.lower().replace("teeth", "tooth")
    match = re.search(r"tooth\s+([0-9\s]+).*?([a-zA-Z]+)$", command)
    if not match:
        return "Could not parse tooth command."
    numbers_part, status = match.groups()
    teeth = re.findall(r"\d+", numbers_part)
    status = status.upper()
    for t in teeth:
        chart_state.setdefault(t, [])
        if status not in chart_state[t]:
            chart_state[t].append(status)
    teeth_str = ", ".join(teeth)
    return f"Updated teeth {teeth_str} with {status}."


def exit_exam(_: str = "") -> str:
    global current_patient, chart_state
    current_patient = None
    chart_state = {}
    return "Exited exam."


def restart_exam(_: str = "") -> str:
    global chart_state
    chart_state = {}
    return "Exam restarted."


chart_tools = [
    Tool(name="StartExam", func=start_exam, description="Begin charting for a patient. Input is the patient name."),
    Tool(name="ChangeToothStatus", func=change_tooth_status,
         description="Change tooth status. Examples: 'tooth 12 MOD', 'teeth 7 8 9 O'."),
    Tool(name="ExitExam", func=exit_exam, description="Exit the current exam."),
    Tool(name="RestartExam", func=restart_exam, description="Restart the exam, clearing data."),
    Tool(name="ClearTooth", func=clear_tooth, description="Clear data for a specific tooth. Input is like 'clear tooth 12'."),
    Tool(name="ResetChart", func=reset_chart, description="Clear all chart data."),
    Tool(name="ViewChart", func=view_chart, description="View current chart data."),
    Tool(name="SummarizeChart", func=summarize_chart, description="Summarize affected teeth."),
    Tool(name="AddNote", func=add_note, description="Add a note. Input is free-form clinical note."),
    Tool(name="MarkMissing", func=mark_missing, description="Mark a tooth as missing. Example: 'tooth 6 is missing'."),
]

prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="{retrieved}"),
    SystemMessage(content="""
You are CodyChart, a voice-activated dental charting assistant used by dentists and dental assistants in a clinical setting. 

You are listening to live audio transcriptions from a dental operatory. These transcriptions may contain unrelated conversation, background chatter, or speech not directed at you.

Your job is to:
1. **Detect and act only on valid charting commands**, using similarity to provided examples.
2. **Ignore all other transcribed text** unless it matches the format or intent of supported commands in the provided examples.

You MUST use the available tools to take action—do NOT respond with free text unless specifically required.

Supported command types include:
- Starting, ending, pausing, or canceling a dental exam
- Updating tooth statuses using surface codes (e.g., MOD, B, O)
- Viewing, summarizing, or resetting the chart
- Adding clinical notes or marking teeth as missing or restored

Tooth status updates must follow this format:
    "tooth <tooth_number> <status_code>"

Examples:
- "tooth 12 MOD" → update tooth 12 with MOD
- "teeth 7 8 9 O" → update all three with O

If no valid command is detected in the transcription, respond with nothing or skip processing.

If a required value (e.g., patient name) is missing, ask for clarification.

Use the provided context examples (`{retrieved}`) to determine if a command is similar to supported inputs. Only act if it meets a reasonable similarity threshold.

"""),
    HumanMessage(content="start exam"),
    AIMessage(content="Who is the patient?"),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    HumanMessage(content="{input}")
])


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

agent_core = create_openai_functions_agent(
    llm=llm,
    tools=chart_tools,
    prompt=prompt
)

chart_agent = AgentExecutor(agent=agent_core, tools=chart_tools, memory=memory, verbose=True)

print("🦷 CodyChart ready. Type a command or 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() in {"quit", "exit"}:
        print("👋 Exiting.")
        break
    results = retrieve_similar(user_input)
    top_result_doc, top_score = results[0] if results else (None, 0.0)
    if top_result_doc or top_score > 0.5:
        intent = top_result_doc.metadata.get("intent")
        if intent == "do_nothing":
            continue
        context = "\n---\n".join(
            [doc.page_content for doc, _ in results])
        result = chart_agent.invoke({"input": enriched, "retrieved": context})
        output = result.get("output", str(result))
        print(f"🤖 {output}")
    else:
        print(f"top_result_doc: {top_result_doc}, top_score: {top_score}")
        continue