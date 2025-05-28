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


def retrieve_similar(query: str, k: int = 2) -> str:
    results = faiss_index.similarity_search(query, k=k)
    return "\n---\n".join([r.page_content for r in results])

# --- Simple chart state ---
chart_state: Dict[str, List[str]] = {}
current_patient: str | None = None


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
]

prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="{retrieved}"),
    SystemMessage(content="""
You are CodyChart, a voice-activated dental charting assistant.
You only listen to the dentist or dental assistant.
Understand short commands such as 'start a dental exam for patient X' or
'tooth 12 MOD'. Use the provided tools to change tooth status and manage
the exam workflow.
                  
The return format should always be: tooth <tooth_number> <status_code>
                  
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
    context = retrieve_similar(user_input)
    enriched = f"{context}\nUser command: {user_input}"
    result = chart_agent.invoke({"input": enriched, "retrieved": context})
    output = result.get("output", str(result))
    print(f"🤖 {output}")