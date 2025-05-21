from datetime import datetime, timedelta
import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader

# === Конфигурация API ===
with open(".env", "r") as f:
    for line in f:
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
            break

llm = ChatOpenAI(temperature=0.3, model="ft:gpt-3.5-turbo-0125:personal::BXyx8p6l")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Загрузка базы услуг ===
with open("Dental Clinic Database and Pricelist.json", "r", encoding="utf-8") as f:
    pricelist = json.load(f)

# Список врачей из базы
doctors = list({item.get("doctor") for item in pricelist.get("services", []) if item.get("doctor")})

def get_available_slots_for_day(day_name):
    now = datetime.now()
    for i in range(1, 15):
        day = now + timedelta(days=i)
        if day.strftime("%A").lower() == day_name.lower():
            return [
                day.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%A, %B %d at %I:%M %p")
                for hour in [9, 11, 14, 16]
            ]
    return []

def find_price(query):
    for item in pricelist.get("services", []):
        if item["name"].lower() in query.lower():
            return f"Yes, we offer {item['name']}. The price is ${item['price']}. Would you like to schedule an appointment for this service?"
    return search_clinic_site(query)

def search_clinic_site(query):
    try:
        loader = WebBaseLoader("https://www.rifkinraanan.com/")
        docs = loader.load()
        for doc in docs:
            if query.lower() in doc.page_content.lower():
                return f"Here is what I found on the clinic website: {doc.page_content[:300]}... Would you like to schedule an appointment?"
        return "I couldn't find relevant information on the clinic website."
    except Exception as e:
        return f"Error accessing clinic website: {e}"

# === Загрузка базы пациентов ===
with open("patients_database.json", "r", encoding="utf-8") as f:
    patient_records = json.load(f)

excel_path = "PATIENT_table.xlsx"
df_patients = pd.read_excel(excel_path)

def check_patient_exists(last, first, dob):
    for patient in patient_records:
        if (patient.get("First Name", "").lower() == first.lower() and
            patient.get("Last Name", "").lower() == last.lower() and
            patient.get("Date of Birth", "") == dob):
            return True
    return False

def add_patient(first_name, last_name, dob, reason="", area="", desired_doctor="", appointment_datetime=""):
    new_entry = {
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": dob,
        "reason": reason,
        "area_of_the_mouth": area,
        "desired_doctor": desired_doctor,
        "appointment_datetime": appointment_datetime
    }
    global df_patients
    df_patients = pd.concat([df_patients, pd.DataFrame([new_entry])], ignore_index=True)
    df_patients.to_excel(excel_path, index=False)

def log_interaction(user_input, assistant_reply):
    with open("chat_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"[User]: {user_input}\n")
        log_file.write(f"[Codey]: {assistant_reply}\n\n")

# === Промт ассистенту ===
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content="""
        You are Codey, a friendly and knowledgeable virtual assistant at a dental clinic.
        - Help users with services, pricing, and appointments.
        - If the user mentions symptoms or pain, ask clarifying questions and offer to book a visit.
        - Use tools provided to get service prices.
        - Be concise, empathetic, and helpful.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    HumanMessage(content="{input}")
])

tools = [
    Tool(
        name="FindServicePrice",
        func=find_price,
        description="Use to find price of dental services or check clinic website if not found"
    )
]

from langchain.agents import create_openai_functions_agent

agent_core = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent = AgentExecutor(
    agent=agent_core,
    tools=tools,
    memory=memory,
    verbose=True
)

# === Цикл чата ===
print("🤖 Hello! I’m Codey, your virtual dental assistant. Type 'help' to see options.")

last_name = ""
first_name = ""
date_of_birth = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "exit", "quit"]:
        print("🤖 Goodbye! Stay healthy!")
        break

    if user_input.lower() == "help":
        print("""
Available commands:
• Ask about service price → e.g., "How much is a cleaning?"
• Check availability → e.g., "available time", "next opening"
• Book appointment → use "appointment"
• Exit → "bye", "exit"
""")
        continue

    if "appointment" in user_input.lower():
        last_name = input("🤖 Please enter your last name: ")
        first_name = input("🤖 Please enter your first name: ")
        date_of_birth = input("🤖 Please enter your date of birth (YYYY-MM-DD): ")

        if check_patient_exists(last_name, first_name, date_of_birth):
            print("🤖 You are already in the system. Proceeding to appointment scheduling.")
        else:
            print("🤖 You are new. We'll collect more information.")
            reason = input("🤖 What is the reason for your visit? ")
            area = input("🤖 What area of the mouth? (UL, LL, UR, LR): ")
            add_patient(first_name, last_name, date_of_birth, reason, area)
            print("🤖 Thank you! You’ve been added to our patient list.")

            desired_doctor = ""
            while True:
                doctor_input = input("🤖 Do you have a preferred doctor, or should we assign one for you? ")
                if doctor_input.strip().lower() in ["any", "no", "none", "whoever", "doesn’t matter", "doesnt matter", "не важно", "без разницы"]:
                    desired_doctor = "Any"
                    break
                elif doctor_input in doctors:
                    desired_doctor = doctor_input
                    break
                else:
                    print("🤖 I didn’t recognize that name. You can choose one of our doctors or just say 'any'.")

            df_patients.loc[
                (df_patients["first_name"] == first_name) &
                (df_patients["last_name"] == last_name) &
                (df_patients["date_of_birth"] == date_of_birth),
                "desired_doctor"
            ] = desired_doctor
            df_patients.to_excel(excel_path, index=False)
            print(f"🤖 Great! We'll assign {desired_doctor} for your visit.")

            preferred_day = input("🤖 What day of the week works best for your appointment? (e.g., Monday, Tuesday): ")
            day_slots = get_available_slots_for_day(preferred_day)
            if day_slots:
                print("🤖 On that day, we have the following time options:")
                print(", ".join([s.split("at")[-1].strip() for s in day_slots]))
                selected_time = input("🤖 What time works best for you? (please type like 9:00 AM, 2:00 PM): ")
                match = [s for s in day_slots if selected_time in s]
                if match:
                    selected_slot = match[0]
                else:
                    selected_slot = day_slots[0]
                    print(f"🤖 That time isn't available. I've scheduled you for the nearest available time: {selected_slot}.")
                df_patients.loc[
                    (df_patients["first_name"] == first_name) &
                    (df_patients["last_name"] == last_name) &
                    (df_patients["date_of_birth"] == date_of_birth),
                    "appointment_datetime"
                ] = selected_slot
                df_patients.to_excel(excel_path, index=False)
                print(f"🤖 Perfect! Your appointment is set for {selected_slot}.")

                # === Дополнительная информация ===
                call_back_number = input("🤖 What is the best number to call you back? ")
                email_address = input("🤖 Could you provide your email address? ")
                prev_office = input("🤖 Have you been to another dental office before? If yes, what's its name? ")
                dds_name = input("🤖 Do you remember the name of the dentist (DDS) there? ")
                xray_info = input("🤖 When were your last x-rays taken? (bwx/fmx/don't know): ")

                for field in ["call_back_number", "email_address", "previous_office", "dds_name", "xray_info"]:
                    if field not in df_patients.columns:
                        df_patients[field] = ""

                df_patients.loc[
                    (df_patients["first_name"] == first_name) &
                    (df_patients["last_name"] == last_name) &
                    (df_patients["date_of_birth"] == date_of_birth),
                    ["call_back_number", "email_address", "previous_office", "dds_name", "xray_info"]
                ] = [call_back_number, email_address, prev_office, dds_name, xray_info]

                df_patients.to_excel(excel_path, index=False)
                # === Сбор информации о страховке и оплате ===
                insurance_status = input("🤖 Do you have dental insurance? (Yes / No / Cash / In Network / Out of Network): ")
                insurance_name = input("🤖 What is the name of your insurance provider? ") if insurance_status.lower() in ["yes", "in network", "out of network"] else ""
                insurance_id = input("🤖 What is your insurance ID? ") if insurance_name else ""
                credit_card = input("🤖 Please provide your credit card number: ")
                exp_date = input("🤖 What is the expiration date? (MM/YY): ")
                cvc_code = input("🤖 What is the CVC code?: ")

                for field in ["insurance_status", "insurance_name", "insurance_id", "credit_card", "exp_date", "cvc_code"]:
                    if field not in df_patients.columns:
                        df_patients[field] = ""

                df_patients.loc[
                    (df_patients["first_name"] == first_name) &
                    (df_patients["last_name"] == last_name) &
                    (df_patients["date_of_birth"] == date_of_birth),
                    ["insurance_status", "insurance_name", "insurance_id", "credit_card", "exp_date", "cvc_code"]
                ] = [insurance_status, insurance_name, insurance_id, credit_card, exp_date, cvc_code]

                # Ensure all fields are treated as strings before saving
                string_columns = [
                    "call_back_number", "email_address", "previous_office", "dds_name", "xray_info",
                    "insurance_status", "insurance_name", "insurance_id", "credit_card", "exp_date", "cvc_code"
                ]
                df_patients[string_columns] = df_patients[string_columns].astype(str)

                df_patients.to_excel(excel_path, index=False)
                print("🤖 Thank you! All your information has been saved. We'll see you soon!")
            else:
                print("🤖 Sorry, we couldn’t find any slots on that day in the next two weeks. Please try another day.")
        continue

    try:
        response = agent.invoke({"input": user_input})
        reply = response["output"] if isinstance(response, dict) else response
        print(f"📝 [user_input]: {user_input}")
        print("🤖", reply)
        log_interaction(user_input, reply)

    except Exception as e:
        print("🤖 Sorry, something went wrong. Please try again later.")
        print(f"(error: {e})")