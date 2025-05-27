"""Interactive bot assistant for a dental clinic."""

from __future__ import annotations

from datetime import datetime
import os
import json
import re
import pandas as pd
import dateparser
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# === API configuration ===
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
                break

llm = ChatOpenAI(temperature=0.3, model="gpt-4o")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

# === Загрузка пациентов ===
EXCEL_PATH = "PATIENT_table.xlsx"

PATIENT_COLUMNS = [
    "Today's_Date",
    "last_name",
    "first_name",
    "date_of_birth",
    "Call_Back_Number",
    "Email Address",
    "reason_of_call",
    "pain_duration",
    "area_of_the_mouth",
    "Previous Dental Office",
    "Name of DDS",
    "Office Number",
    "Date_of_last_x-ray",
    "type_of_x-ray",
    "Name of Insurance",
    "VIP",
    "Appointment_date",
]

if os.path.exists(EXCEL_PATH):
    df_patients = pd.read_excel(EXCEL_PATH)
else:
    df_patients = pd.DataFrame(columns=PATIENT_COLUMNS)
    df_patients.to_excel(EXCEL_PATH, index=False)

services: dict = {}
if os.path.exists("Dental Clinic Database and Pricelist.json"):
    with open("Dental Clinic Database and Pricelist.json", "r", encoding="utf-8") as f:
        services = json.load(f)

json_patients: list[dict] = []
if os.path.exists("patients_database.json"):
    with open("patients_database.json", "r", encoding="utf-8") as f:
        json_patients = json.load(f)


def normalize_dob(text: str) -> str:
    text = str(text).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def parse_name_and_dob(text: str):
    date_patterns = [r"\d{4}-\d{1,2}-\d{1,2}", r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", r"\d{1,2}\.\d{1,2}\.\d{2,4}"]
    dob_raw = None
    for pat in date_patterns:
        m = re.search(pat, text)
        if m:
            dob_raw = m.group(0)
            break
    if not dob_raw:
        return None
    dob_norm = normalize_dob(dob_raw)
    if not dob_norm:
        return None
    name_part = text.replace(dob_raw, "").replace(",", " ").strip()
    tokens = [t for t in name_part.split() if t]
    if len(tokens) < 2:
        return None
    first = tokens[0]
    last = " ".join(tokens[1:])
    return first, last, dob_norm

for col in PATIENT_COLUMNS:
    if col not in df_patients.columns:
        df_patients[col] = ""
    df_patients[col] = df_patients[col].astype(str)


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

def check_patient_exists(text: str) -> bool:
    parsed = parse_name_and_dob(text)
    if not parsed:
        return False
    first, last, dob_norm = parsed
    for rec in json_patients:
        rec_norm = normalize_dob(rec.get("DOB", ""))
        if rec.get("Patient Name", "").lower() == f"{first.lower()} {last.lower()}" and rec_norm == dob_norm:
            return True
        if rec.get("Patient Name", "").lower() == f"{last.lower()} {first.lower()}" and rec_norm == dob_norm:
            return True
    return False

def add_patient_to_db(data: dict) -> str:
    import json

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            raise ValueError("Expected a JSON object with patient fields, got malformed string.")

    global df_patients
    new_entry = {col: str(data.get(col, "")) for col in PATIENT_COLUMNS}
    df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
    df_patients.to_excel(EXCEL_PATH, index=False)
    return f"Patient {new_entry.get('first_name')} {new_entry.get('last_name')} added."


def find_service_price(query: str) -> str:
    if not services:
        return "Service database unavailable."
    for cat in services.get("Services", []):
        if isinstance(cat, dict):
            for item in cat.get("items", []) or cat.get("services", []):
                name = item.get("service_name") or item.get("name")
                price = item.get("base_price") or item.get("price")
                if name and name.lower() in query.lower():
                    return f"{name} costs ${price}"
    return "Service not found."

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
    Tool(name="CheckPatientExists", func=check_patient_exists, description="Checks if a patient exists from a free-form 'name and DOB' string"),
    Tool(name="FindServicePrice", func=find_service_price, description="Returns price for a dental service"),
]

# === Prompt ===
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are Codey, a friendly assistant for a dental clinic."),
    SystemMessage(
        content="""
- Greet users and help with appointments.
- If a user wants to book, ask 'For yourself or someone else?'.
- Collect last name, first name and date of birth then validate the data.
- Check existing patients with CheckPatientExists.
- If information is missing, gather phone number, email, previous dental office, dentist name, office number and last x-ray date.
- Record all data with AddPatientToDB.
- Use GetAvailableSlots or GetSlotsForDateInput to suggest the earliest appointments, preferring Friday.
- Answer price or service questions using FindServicePrice.
- Keep replies short and polite.""",
    ),
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
    result = appointment_agent.invoke({"input": user_input})
    output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
    print(f"🤖 {output}")
