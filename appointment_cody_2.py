from datetime import datetime, timedelta
import os
import json
import pandas as pd
import dateparser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# === Конфигурация API ===
with open(".env", "r") as f:
    for line in f:
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
            break

llm = ChatOpenAI(temperature=0.3, model="gpt-4o")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

# === Загрузка пациентов ===
excel_path = "PATIENT_table.xlsx"
df_patients = pd.read_excel(excel_path)

string_columns = [
    "first_name", "last_name", "date_of_birth", "reason", "area_of_the_mouth",
    "desired_doctor", "appointment_datetime", "call_back_number", "email_address",
    "previous_office", "dds_name", "xray_info", "insurance_status",
    "insurance_name", "insurance_id", "credit_card", "exp_date", "cvc_code"
]

for col in string_columns:
    if col not in df_patients.columns:
        df_patients[col] = ""
    df_patients[col] = df_patients[col].astype(str)

# === FAISS RAG ===
faiss_index = FAISS.load_local(
    "rag_appointment_agent_faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

def retrieve_similar_examples(query, k=3):
    results = faiss_index.similarity_search(query, k=k)
    return "\n---\n".join([r.page_content for r in results])

# === Слоты ===
slots_df = pd.read_excel("available_slots_by_doctor.xlsx")
slots_df["datetime"] = pd.to_datetime(slots_df["datetime"])

def get_available_slots_from_file(doctor=None, day=None):
    filtered = slots_df.copy()
    if doctor:
        filtered = filtered[filtered["doctor"].str.lower() == doctor.lower()]
    if day:
        filtered = filtered[filtered["weekday"].str.lower() == day.lower()]
    return filtered.sort_values("datetime")["datetime"].dt.strftime("%A, %B %d at %I:%M %p").tolist()

def get_available_slots_for_day(day_name: str, doctor=None) -> list[str]:
    return get_available_slots_from_file(doctor=doctor, day=day_name)

def get_slots_for_input_date(text: str) -> list[str]:
    parsed_date = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
    if not parsed_date:
        return ["Sorry, I couldn't understand the date."]
    return [
        parsed_date.replace(hour=h, minute=0).strftime("%A, %B %d at %I:%M %p")
        for h in [9, 11, 12, 14, 16, 17]
    ]

def check_patient_exists_simple(full_name_dob: str) -> bool:
    try:
        name_part, dob = full_name_dob.split(",")
        first, last = name_part.strip().split(" ", 1)
        matches = df_patients[
            (df_patients["first_name"].str.lower() == first.lower()) &
            (df_patients["last_name"].str.lower() == last.lower()) &
            (df_patients["date_of_birth"] == dob.strip())
        ]
        return not matches.empty
    except Exception:
        return False

def add_patient_to_db(data: dict) -> str:
    import json

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            raise ValueError("Expected a JSON object with patient fields, got malformed string.")

    global df_patients
    new_entry = {col: str(data.get(col, "")) for col in string_columns}
    df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
    df_patients.to_excel(excel_path, index=False)
    return f"Patient {new_entry.get('first_name')} {new_entry.get('last_name')} added."

# === Tools ===
appointment_tools = [
    Tool(name="GetAvailableSlots", func=get_available_slots_for_day, description="Returns slots for a weekday"),
    Tool(name="GetSlotsForDateInput", func=get_slots_for_input_date, description="Returns slots for a date string"),
    Tool(
        name="AddPatientToDB",
        func=add_patient_to_db,
        description="""Adds a patient to the database.
Pass a JSON object like:
{
  "first_name": "Jane",
  "last_name": "Doe",
  "date_of_birth": "1982-02-04",
  "reason": "toothache",
  "appointment_datetime": "Friday at 11:00 AM",
  "desired_doctor": "Dr. Smith"
}"""
    ),
    Tool(name="CheckPatientExists", func=check_patient_exists_simple, description="Checks if patient exists")
]

# === Prompt ===
prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="{retrieved_examples}"),

    SystemMessage(
        content="""
You are AppointmentAgent, a proactive and kind assistant at a dental clinic.
Your job is to:
- Understand if a user is in pain or describes a symptom.
- Offer the nearest appointment slot for a standard doctor.
- If the user confirms the offer, ask for first name, last name, and date of birth.
- If the user is booking for someone else (e.g., 'for my daughter'), collect that person's name and DOB instead.
- Never assume the user is booking for someone else unless explicitly stated (e.g., 'for my husband').
- If a user mentions a specific day or month like 'next Tuesday' or 'in June', use the GetSlotsForDateInput tool.
- Then, use the tool AddPatientToDB with all the collected data.
- Use other tools when helpful. Do not invent details.
If the user asks which times are available, list the time slots clearly. Always respond to requests like 'Which ones?' with specific appointment options.
You must keep track of what the user has said in the conversation.
If the user asks about opening hours or whether you're open on a specific day (e.g., weekends), respond with business hours. Only offer appointments if the user asks to book.
"""
    ),

    HumanMessage(content="What are your hours?"),
    AIMessage(content="We’re open Monday through Friday, 9:00 AM to 5:00 PM."),
    HumanMessage(content="Are you open on Sunday?"),
    AIMessage(content="We’re closed on weekends, but happy to help you Monday to Friday."),
    HumanMessage(content="I have a terrible toothache."),
    AIMessage(content="I'm sorry to hear that. Would you like to come in this Friday at 11:00 AM?"),
    HumanMessage(content="Yes, that works."),
    AIMessage(content="Great. May I have your full name and date of birth?"),
    HumanMessage(content="My name is Jane Doe 02/04/1982"),
    AIMessage(content="Thank you, Jane. What is the reason for your visit?"),
    HumanMessage(content="Can you book me for a cleaning?"),
    AIMessage(content="Sure. We have Friday at 11:00 AM available. Does that work for you?"),
    HumanMessage(content="What is the nearest available appointment?"),
    AIMessage(content="We have Friday at 11:00 AM available. Does that work for you?"),

    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    HumanMessage(content="{input}")
])

agent_core = create_openai_functions_agent(
    llm=llm,
    tools=appointment_tools,
    prompt=prompt
)

appointment_agent = AgentExecutor(
    agent=agent_core,
    tools=appointment_tools,
    memory=memory,
    verbose=True
)

print("🤖 Hi! I’m AppointmentAgent. Ask me to book your visit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye!")
        break
    retrieved_context = retrieve_similar_examples(user_input)
    enriched_input = f"{retrieved_context}\nNow answer this: {user_input}"
    print("🧠 Retrieved:\n", retrieved_context)
    print("📨 Enriched Input:\n", enriched_input)
    result = appointment_agent.invoke({"input": enriched_input})
    output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
    print(f"🤖 {output}")
