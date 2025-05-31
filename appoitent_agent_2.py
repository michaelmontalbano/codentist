from __future__ import annotations
from uuid import uuid4

"""Interactive bot assistant for a dental clinic."""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.chains import RetrievalQA
from datetime import datetime
import os
import json
import re
import pandas as pd
import dateparser
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

# === API configuration ===
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
                break

llm = ChatOpenAI(temperature=0.3, model="gpt-4o")
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    k=5
)

# === Загрузка пациентов ===
EXCEL_PATH = "PATIENT_table.xlsx"
PATIENT_COLUMNS = [
    "patient_id", "Today's_Date", "last_name", "first_name", "date_of_birth", "Call_Back_Number",
    "Email Address", "reason_of_call", "pain_duration", "area_of_the_mouth",
    "Previous Dental Office", "Name of DDS", "Office Number", "Date_of_last_x-ray",
    "type_of_x-ray", "Name of Insurance", "VIP", "Appointment_date", "doctor_name"
]

# === Загрузка доступных слотов врачей ===
DOCTOR_SLOTS_PATH = "available_slots_by_doctor.xlsx"
df_doctor_slots = pd.DataFrame()
if os.path.exists(DOCTOR_SLOTS_PATH):
    try:
        df_doctor_slots = pd.read_excel(DOCTOR_SLOTS_PATH)
        print(f"✅ Loaded doctor slots from {DOCTOR_SLOTS_PATH}")
        print(f"Columns: {list(df_doctor_slots.columns)}")
    except Exception as e:
        print(f"❌ Error loading doctor slots: {e}")
        df_doctor_slots = pd.DataFrame()
else:
    print(f"⚠️ {DOCTOR_SLOTS_PATH} not found. Doctor selection will be limited.")

if os.path.exists(EXCEL_PATH):
    df_patients = pd.read_excel(EXCEL_PATH)
else:
    df_patients = pd.DataFrame(columns=PATIENT_COLUMNS)
    df_patients.to_excel(EXCEL_PATH, index=False)

services: dict = {}
SERVICE_DB_FILE = "Dental Clinic Database and Pricelist.json"
if os.path.exists(SERVICE_DB_FILE):
    with open(SERVICE_DB_FILE, "r", encoding="utf-8") as f:
        services = json.load(f)

json_patients: list[dict] = []

# === Load RAG vectorstore for name/DOB ===
if os.path.exists("parsed_name_dob_for_rag.json"):
    with open("parsed_name_dob_for_rag.json", "r", encoding="utf-8") as f:
        parsed_examples = json.load(f)
    doc_list = [Document(page_content=ex["text"], metadata={"json": ex["parsed"]}) for ex in parsed_examples if
                "parsed" in ex]
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(doc_list, embedding_model)
    rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    rag_retriever = None

# === Load FAISS index for intent detection ===
INTENT_FAISS_INDEX = "intent_faiss_index"
path_to_whole_intents = "extended_intents_dataset_updated_2.json"
if os.path.exists(path_to_whole_intents):
    loader = JSONLoader(file_path=path_to_whole_intents, jq_schema=".[]", text_content=False)
    documents = loader.load()
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(INTENT_FAISS_INDEX)
    intent_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print("⚠️ intents_dataset.json not found. Intent detection will be disabled.")
    intent_retriever = None

# Глобальная переменная для отслеживания текущего пациента
current_patient_id = None


def start_new_patient():
    global current_patient_id, df_patients
    current_patient_id = str(uuid4())
    new_entry = {"patient_id": current_patient_id, "Today's_Date": datetime.now().strftime("%Y-%m-%d")}
    for col in PATIENT_COLUMNS:
        if col not in new_entry:
            new_entry[col] = ""
    df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
    df_patients.to_excel(EXCEL_PATH, index=False)
    return current_patient_id


def is_field_filled(field_name: str) -> bool:
    global current_patient_id
    if current_patient_id and not df_patients.empty:
        mask = df_patients["patient_id"] == current_patient_id
        if mask.any():
            latest = df_patients.loc[mask].iloc[-1]
            filled = bool(str(latest.get(field_name, "")).strip())
            print(
                f"Debug: Checking {field_name} for patient_id {current_patient_id}, filled: {filled}, value: {latest.get(field_name, '')}")
            return filled
    print(f"Debug: No patient_id or df_patients empty, field {field_name} not filled.")
    return False


def get_available_doctors(query: str = "") -> str:
    """Получить список доступных врачей"""
    if df_doctor_slots.empty:
        return "Dr. Smith, Dr. Johnson, Dr. Williams"

    try:
        if 'doctor_name' in df_doctor_slots.columns:
            doctors = df_doctor_slots['doctor_name'].dropna().unique()
        elif 'Doctor' in df_doctor_slots.columns:
            doctors = df_doctor_slots['Doctor'].dropna().unique()
        elif 'doctor' in df_doctor_slots.columns:
            doctors = df_doctor_slots['doctor'].dropna().unique()
        else:
            # Попробуем найти столбец с врачами по первым нескольким столбцам
            possible_cols = [col for col in df_doctor_slots.columns if 'doctor' in col.lower() or 'dr' in col.lower()]
            if possible_cols:
                doctors = df_doctor_slots[possible_cols[0]].dropna().unique()
            else:
                return "Dr. Smith, Dr. Johnson, Dr. Williams"

        return ", ".join(doctors)
    except Exception as e:
        print(f"Error getting doctors: {e}")
        return "Dr. Smith, Dr. Johnson, Dr. Williams"


def get_available_days_for_doctor(doctor_name: str) -> str:
    """Получить доступные дни недели для конкретного врача"""
    if df_doctor_slots.empty:
        return "Monday, Tuesday, Wednesday, Thursday, Friday"

    try:
        # Найти столбец с врачами
        doctor_col = None
        for col in df_doctor_slots.columns:
            if 'doctor' in col.lower() or 'dr' in col.lower():
                doctor_col = col
                break

        if doctor_col is None:
            return "Monday, Tuesday, Wednesday, Thursday, Friday"

        # Фильтровать по врачу
        doctor_mask = df_doctor_slots[doctor_col].str.contains(doctor_name, case=False, na=False)
        doctor_slots = df_doctor_slots[doctor_mask]

        if doctor_slots.empty:
            return "Monday, Tuesday, Wednesday, Thursday, Friday"

        # Собрать уникальные дни
        days = set()
        for _, row in doctor_slots.iterrows():
            weekday_val = row.get('weekday', '')
            if pd.notna(weekday_val) and str(weekday_val).strip():
                day = str(weekday_val).strip()
                # Извлекаем только день недели, если есть время
                if ' ' in day:
                    day = day.split()[0]
                days.add(day)

        return ", ".join(sorted(days)) if days else "Monday, Tuesday, Wednesday, Thursday, Friday"

    except Exception as e:
        print(f"Error getting days for doctor {doctor_name}: {e}")
        return "Monday, Tuesday, Wednesday, Thursday, Friday"


def get_available_times_for_doctor_and_day(params: str) -> str:
    """
    Получить доступные времена для врача в конкретный день.
    params: строка вида 'doctor_name:preferred_day'
    """
    try:
        if isinstance(params, str) and ':' in params:
            doctor_name, preferred_day = params.split(':', 1)
            doctor_name = doctor_name.strip()
            preferred_day = preferred_day.strip()
        else:
            return "Invalid input. Use 'doctor_name:preferred_day'"
    except Exception as e:
        return f"Error parsing input: {str(e)}"

    if df_doctor_slots.empty:
        return "9:00 AM, 11:00 AM, 2:00 PM, 4:00 PM"

    try:
        # Найти столбец с врачами
        doctor_col = None
        for col in df_doctor_slots.columns:
            if 'doctor' in col.lower() or 'dr' in col.lower():
                doctor_col = col
                break

        if doctor_col is None:
            return "9:00 AM, 11:00 AM, 2:00 PM, 4:00 PM"

        # Фильтровать по врачу и дню
        doctor_mask = df_doctor_slots[doctor_col].str.contains(doctor_name, case=False, na=False)
        day_mask = df_doctor_slots['weekday'].str.contains(preferred_day, case=False, na=False)
        filtered_slots = df_doctor_slots[doctor_mask & day_mask]

        if filtered_slots.empty:
            return "9:00 AM, 11:00 AM, 2:00 PM, 4:00 PM"

        # Собрать времена
        times = []
        for _, row in filtered_slots.iterrows():
            weekday_val = row.get('weekday', '')
            datetime_val = row.get('datetime', '')

            # Пробуем извлечь время из weekday (например, "Monday 9:00 AM")
            if pd.notna(weekday_val) and str(weekday_val).strip():
                parts = str(weekday_val).strip().split()
                if len(parts) >= 2:
                    time_part = " ".join(parts[1:])
                    times.append(time_part)
            # Либо из datetime столбца
            elif pd.notna(datetime_val) and str(datetime_val).strip():
                try:
                    dt = pd.to_datetime(datetime_val)
                    time_str = dt.strftime("%I:%M %p")
                    times.append(time_str)
                except:
                    times.append(str(datetime_val))

        return ", ".join(times[:6]) if times else "9:00 AM, 11:00 AM, 2:00 PM, 4:00 PM"

    except Exception as e:
        print(f"Error getting times for doctor {doctor_name} on {preferred_day}: {e}")
        return "9:00 AM, 11:00 AM, 2:00 PM, 4:00 PM"

def pick_random_doctor() -> str:
    """Выбрать случайного врача из доступных"""
    doctors_str = get_available_doctors()
    doctors_list = [d.strip() for d in doctors_str.split(',')]
    if doctors_list:
        import random
        return random.choice(doctors_list)
    return "Dr. Smith"


# Глобальный объект для LLM и кэш
llm_intent = ChatOpenAI(temperature=0, model="gpt-4o")
intent_cache = {}


def detect_intent(user_input: str) -> str:
    print(f"Debug: Processing user_input: {user_input}")

    # Проверяем специальные случаи для выбора врача
    user_lower = user_input.lower().strip()

    # Паттерны для "не важно" или "выберите сами"
    no_preference_patterns = [
        "doesn't matter", "don't care", "any doctor", "any", "whatever",
        "you choose", "pick for me", "choose for me", "select for me",
        "не важно", "всё равно", "любой", "выберите сами"
    ]

    # Паттерны для выбора конкретного врача
    doctor_selection_patterns = [
        "dr.", "doctor", "i'll take", "i want", "i choose", "i prefer"
    ]

    # Паттерны для предпочтений по дню
    day_preference_patterns = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
        "morning", "afternoon", "evening", "утром", "днем", "вечером"
    ]

    # Паттерны для предпочтений по времени
    time_preference_patterns = [
        "am", "pm", "o'clock", "morning", "afternoon", "evening",
        "утра", "дня", "вечера", ":"
    ]

    if any(pattern in user_lower for pattern in no_preference_patterns):
        print("[🤖 Detected intent: doctor_no_preference]")
        return "doctor_no_preference"

    # Если пользователь упоминает конкретного врача
    doctors_str = get_available_doctors()
    available_doctors = [d.strip().lower() for d in doctors_str.split(',')]
    if any(doctor in user_lower for doctor in available_doctors):
        print("[🤖 Detected intent: doctor_selection]")
        return "doctor_selection"

    # Проверяем предпочтения по дню
    if any(pattern in user_lower for pattern in day_preference_patterns):
        print("[🤖 Detected intent: day_preference]")
        return "day_preference"

    # Проверяем предпочтения по времени
    if any(pattern in user_lower for pattern in time_preference_patterns):
        print("[🤖 Detected intent: time_preference]")
        return "time_preference"

    if user_input in intent_cache:
        print(f"[🤖 Retrieved intent from cache: {intent_cache[user_input]}]")
        return intent_cache[user_input]

    if not intent_retriever:
        print("Debug: Intent retriever is not available.")
        return "unknown"

    examples = intent_retriever.invoke(user_input)
    print(f"Debug: Retrieved examples: {[doc.page_content for doc in examples]}")

    # Парсим JSON из page_content
    example_texts = []
    for doc in examples:
        try:
            content_json = json.loads(doc.page_content)
            intent = content_json.get("intent", "unknown")
            text = content_json.get("text", doc.page_content)
            example_texts.append(f'User: {text}\nIntent: {intent}')
        except json.JSONDecodeError:
            example_texts.append(f'User: {doc.page_content}\nIntent: {doc.metadata.get("intent", "unknown")}')

    example_texts_str = "\n".join(example_texts)

    prompt = f"""
        You are an intent classifier. Based on the examples and the user input, extract the intent label.
        Return only the intent name, like: make_appointment, ask_price, provide_contact_info, greet, etc.

        ### Examples ###
        {example_texts_str}

        ### User input ###
        User: {user_input}

        ### Output ###
        """

    try:
        response = llm_intent.invoke(prompt)
        intent = response.content.strip().lower()
        print(f"[🤖 Detected intent: {intent}]")
        intent_cache[user_input] = intent
        return intent
    except Exception as e:
        print(f"❌ Intent detection failed: {str(e)} (type: {type(e).__name__})")
        return "unknown"


def set_patient_field(field_and_value: str) -> str:
    """
    Set patient field with format 'field:value' or JSON string
    """
    global df_patients, current_patient_id
    print(f"Debug: Received data in set_patient_field: {field_and_value} (type: {type(field_and_value)})")

    # Try to parse as JSON first
    try:
        if isinstance(field_and_value, str) and field_and_value.startswith('{'):
            data = json.loads(field_and_value)
            field = data["field"]
            value = str(data["value"])
        elif isinstance(field_and_value, str) and ':' in field_and_value:
            # Handle format "field:value"
            field, value = field_and_value.split(':', 1)
            field = field.strip()
            value = value.strip()
        else:
            return f"Invalid format. Use 'field:value' or JSON format."
    except Exception as e:
        return f"Error parsing field data: {str(e)}"

    if field == "date_of_birth":
        parsed_date = dateparser.parse(value)
        if parsed_date:
            value = parsed_date.strftime("%Y-%m-%d")
        else:
            return "Invalid date format. Please use MM/DD/YYYY."

    print(f"Debug: Setting {field} to {value}")

    # Обновляем запись текущего пациента
    if current_patient_id and not df_patients.empty:
        mask = df_patients["patient_id"] == current_patient_id
        if mask.any():
            df_patients.loc[mask, field] = value
        else:
            # Если пациент не найден, создаем новую запись
            new_entry = {col: "" for col in PATIENT_COLUMNS}
            new_entry["patient_id"] = current_patient_id
            new_entry["Today's_Date"] = datetime.now().strftime("%Y-%m-%d")
            new_entry[field] = value
            df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        # Создаем новую запись если нет текущего пациента
        new_entry = {col: "" for col in PATIENT_COLUMNS}
        new_entry["patient_id"] = current_patient_id or str(uuid4())
        new_entry["Today's_Date"] = datetime.now().strftime("%Y-%m-%d")
        new_entry[field] = value
        df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
        if not current_patient_id:
            current_patient_id = new_entry["patient_id"]

    print(f"Debug: Updated df_patients for patient {current_patient_id}")
    df_patients.to_excel(EXCEL_PATH, index=False)
    return f"{field} updated."


def extract_name_dob_with_rag(user_input: str) -> dict | None:
    if not rag_retriever:
        return None
    examples = rag_retriever.invoke(user_input)
    example_texts = "\n".join([f'User: {doc.page_content}\n{json.dumps(doc.metadata["json"])}' for doc in examples])
    prompt = f"""
You are a data extractor. Based on the examples below and the user input, extract first_name, last_name, and date_of_birth.
### Examples ###
{example_texts}
### User input ###
User: {user_input}
### Output ###
"""
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception:
        return None


def validate_phone(number: str) -> bool:
    digits = re.sub(r"\D", "", str(number))
    return len(digits) >= 7


def validate_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$", str(email)) is not None


def check_new_vip_status(first: str, last: str, dob: str) -> int:
    return 3


def add_patient_to_db(data: dict) -> str:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            raise ValueError("Expected a JSON object with patient fields, got malformed string.")
    global df_patients
    new_entry = {col: str(data.get(col, "")) for col in PATIENT_COLUMNS}
    if not new_entry.get("Today's_Date"):
        new_entry["Today's_Date"] = datetime.now().strftime("%Y-%m-%d")
    if new_entry.get("Call_Back_Number") and not validate_phone(new_entry["Call_Back_Number"]):
        return "Invalid phone number provided."
    if new_entry.get("Email Address") and not validate_email(new_entry["Email Address"]):
        return "Invalid email address provided."
    if not new_entry.get("VIP"):
        new_entry["VIP"] = str(check_new_vip_status(new_entry.get("first_name"), new_entry.get("last_name"),
                                                    new_entry.get("date_of_birth")))
    mask = (
                   df_patients["first_name"].str.lower() == new_entry.get("first_name", "").lower()
           ) & (
                   df_patients["last_name"].str.lower() == new_entry.get("last_name", "").lower()
           ) & (
                   df_patients["date_of_birth"] == new_entry.get("date_of_birth")
           )
    if mask.any():
        df_patients.loc[mask, PATIENT_COLUMNS] = pd.DataFrame([new_entry]).values
    else:
        df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
    df_patients.to_excel(EXCEL_PATH, index=False)
    return f"Patient {new_entry.get('first_name')} {new_entry.get('last_name')} saved."


def find_service_price(query: str) -> str:
    if not services:
        return "Service database unavailable."
    for cat in services.get("Services", []):
        for item in cat.get("items", []) or cat.get("services", []):
            name = item.get("service_name") or item.get("name")
            price = item.get("base_price") or item.get("price")
            if name and name.lower() in query.lower():
                return f"{name} costs ${price}"
    return "Service not found."


def get_slots_for_input_date(text: str) -> list[str]:
    parsed_date = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
    if not parsed_date:
        return ["Sorry, I couldn't understand the date."]
    return [parsed_date.replace(hour=h, minute=0).strftime("%A, %B %d at %I:%M %p") for h in [9, 11, 12, 14, 16, 17]]


def get_available_slots_for_day(day_name: str, doctor=None) -> list[str]:
    return ["Friday at 11:00 AM", "Friday at 2:00 PM"]


appointment_tools = [
    Tool(name="GetAvailableSlots", func=get_available_slots_for_day, description="Returns slots for a weekday"),
    Tool(name="GetSlotsForDateInput", func=get_slots_for_input_date, description="Returns slots for a date string"),
    Tool(name="AddPatientToDB", func=add_patient_to_db, description="Adds patient record to database"),
    Tool(name="FindServicePrice", func=find_service_price, description="Returns price for a dental service"),
    Tool(name="SetPatientField", func=set_patient_field,
         description="Updates a single patient field. Use format 'field_name:field_value' (e.g., 'last_name:Smith')"),
    Tool(name="GetAvailableDoctors", func=get_available_doctors,
         description="Returns list of available doctors. Can be called without parameters or with optional query string."),
    Tool(name="GetAvailableDays", func=get_available_days_for_doctor,
         description="Returns available days for a specific doctor. Requires doctor name as parameter."),
    Tool(name="GetAvailableTimes", func=get_available_times_for_doctor_and_day,
         description="Returns available times for a specific doctor on a specific day. Requires doctor name and day as parameters.")
]


def handle_intent_response(user_input: str, intent: str) -> str:
    """Handle simple intent-based responses that don't need tools"""
    print(f"[🤖 Handling intent: {intent}]")

    # Только простые интенты, которые не требуют использования инструментов
    if intent == 'greet':
        return "Hello! How can I assist you today? Are you looking to book an appointment or do you need help with something else?"
    elif intent == 'make_appointment':
        return "I'd be happy to help you book an appointment! Is this appointment for yourself or someone else?"
    elif intent == 'identify_booking_for':
        return "Thank you! Let's start with the last name for this appointment."
    elif intent == 'doctor_no_preference':
        # Выбираем случайного врача
        selected_doctor = pick_random_doctor()
        set_patient_field(f"doctor_name:{selected_doctor}")
        return f"Perfect! I'll assign you to {selected_doctor}. What day of the week works best for you?"
    elif intent == 'unknown':
        return "I'm not sure I understand. Are you trying to book an appointment or ask about our services?"
    else:
        # Все остальные интенты (включая сбор данных) передаем агенту
        return None


prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
        You are Codey, a dental clinic assistant. You help patients book appointments and answer questions.

        🟢 IMPORTANT: Handle data collection and tool usage for appointment booking in the following order:
        1. Last name
        2. First name  
        3. Date of birth
        4. Doctor selection (ask "Which doctor would you like to see?" and use GetAvailableDoctors tool)
        5. Day preference (ask "What day of the week works best for you?" and use GetAvailableDays tool)
        6. Time preference (ask specific day preference, then use GetAvailableTimes tool)
        7. Phone number
        8. Complete booking

        🟢 Data Collection Intents:
        - If intent is 'provide_last_name': Extract the last name from the user message and use SetPatientField tool with format "last_name:extracted_name". Then ask for first name.
        - If intent is 'provide_first_name': Extract the first name from the user message and use SetPatientField tool with format "first_name:extracted_name". Then ask for date of birth.
        - If intent is 'provide_dob': Extract the date from the user message and use SetPatientField tool with format "date_of_birth:extracted_date". Then ask "Which doctor would you like to see?" and use GetAvailableDoctors tool.
        - If intent is 'doctor_selection': Extract the doctor name from the user message, use SetPatientField with "doctor_name:extracted_doctor_name", then ask "What day of the week works best for you?" and use GetAvailableDays tool.
        - If intent is 'doctor_no_preference': Select the first available doctor, use SetPatientField with "doctor_name:selected_doctor", then ask "What day of the week works best for you?" and use GetAvailableDays tool.
        - If intent is 'day_preference': Extract the preferred day, then use GetAvailableTimes tool to show available times for that day and the selected doctor.
        - If intent is 'time_preference' or user selects a time: Use SetPatientField with "Appointment_date:extracted_time", then ask for phone number.
        - If intent is 'provide_contact_info': Determine if it's phone or email, use SetPatientField appropriately, then complete the booking process.

        🟢 Service Intents:
        - If intent is 'ask_price': Use FindServicePrice tool to look up the requested service price.

        🟡 Important Rules:
        - Always ask for ONE piece of information at a time
        - Use the SetPatientField tool ONCE per response after collecting information
        - Keep responses short and conversational
        - Extract the relevant information from the user's message before using tools
        - Do NOT repeat questions if the field was already set successfully
        - After collecting date of birth, ALWAYS ask about doctor preference and show available doctors
        - After doctor selection, ALWAYS ask about day preference first, then show available times for that specific day
        - Never show all available slots at once - first ask for day preference, then show times for that day only
        - When user says "doesn't matter", "pick for me", "you choose" etc. for doctor selection, treat it as doctor_no_preference
        """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User message: {user_message}\nDetected intent: {intent}\n\nHandle this request appropriately."),
    MessagesPlaceholder(variable_name="agent_scratchpad")
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
    verbose=True,
    handle_parsing_errors=True
)

print("🤖 Hello! I am Codey - your virtual assistant. How can I help you?")
start_new_patient()

# Main conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("🤖 Goodbye!")
        break

    # Detect intent
    intent = detect_intent(user_input)

    # Check if we can handle the intent directly (простые интенты без инструментов)
    direct_response = handle_intent_response(user_input, intent)

    if direct_response:
        print(f"🤖 {direct_response}")

        # Если это был выбор врача "не важно", то нужно спросить о предпочтениях по дню
        if intent == 'doctor_no_preference':
            selected_doctor = pick_random_doctor()
            set_patient_field(f"doctor_name:{selected_doctor}")
            # Показываем доступные дни
            days = get_available_days_for_doctor(selected_doctor)
            print(f"🤖 Available days for {selected_doctor}: {days}")
    else:
        # Use agent for tool-based intents (data collection, price lookup, etc.)
        result = appointment_agent.invoke({
            "input": f"User message: {user_input}\nDetected intent: {intent}",
            "user_message": user_input,
            "intent": intent
        })

        # Extract and display output
        output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
        print(f"🤖 {output}")