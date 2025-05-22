from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import requests

EMAIL = "lily@dds.cloud"
PASSWORD = "Camden4141280*"
DOWNLOAD_PATH = os.path.abspath("downloads")
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

chrome_options = Options()
prefs = {
    "download.default_directory": DOWNLOAD_PATH,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 15)

try:
    driver.get("https://app.mangovoice.com/login")
    wait.until(EC.presence_of_element_located((By.XPATH, '//input[@placeholder="email / username"]'))).send_keys(EMAIL)
    driver.find_element(By.XPATH, '//input[@placeholder="password"]').send_keys(PASSWORD + Keys.RETURN)
    time.sleep(6)

    driver.get("https://app.mangovoice.com/calls")
    wait.until(EC.presence_of_element_located((By.XPATH, '//input[@placeholder="Search a name or number"]')))
    time.sleep(5)

    SCROLL_PAUSE_TIME = 2
    scrolls = 15
    for _ in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)

    rows = driver.find_elements(By.CSS_SELECTOR, "div.mango-list-item")
    print(f"🔎 Found call rows: {len(rows)}")

    for i in range(len(rows)):
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, "div.mango-list-item")
            row = rows[i]

            print(f"\n👉 Row {i+1}, content: {row.text.strip()}")
            driver.execute_script("arguments[0].scrollIntoView(true);", row)
            time.sleep(1.0)
            ActionChains(driver).move_to_element(row).click().perform()
            time.sleep(3)
            print(f"🧭 URL after click: {driver.current_url}")

            try:
                audio_tag = wait.until(EC.presence_of_element_located((By.XPATH, '//audio')))
                audio_src = audio_tag.get_attribute("src")
                if audio_src:
                    print(f"🎧 Found audio file: {audio_src}")
                    filename = os.path.join(DOWNLOAD_PATH, f"call_{i+1}.mp3")
                    r = requests.get(audio_src)
                    with open(filename, "wb") as f:
                        f.write(r.content)
                    print(f"✅ Saved: {filename}")
                else:
                    print("⚠️ <audio> tag found, but src attribute is empty.")
            except Exception as e:
                print(f"❌ Failed to find <audio>: {e}")

        except Exception as e:
            print(f"❌ General error on row {i+1}: {e}")

finally:
    driver.quit()
    print("\n🏁 All files processed.")
