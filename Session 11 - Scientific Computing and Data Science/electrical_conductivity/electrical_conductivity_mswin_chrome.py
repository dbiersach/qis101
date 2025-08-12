#!/usr/bin/env python3
"""electrical_conductivity_mswin_chrome.py"""

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Set up Selenium with webdriver-manager
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

# Open the webpage
url = "https://www.schoolmykids.com/learn/periodic-table/electrical-conductivity-of-all-the-elements"
driver.get(url)

# Wait up to 10 seconds for JavaScript to load an HTML <table> on the page.
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))

# Get the rendered HTML
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# Find and parse the table
table = soup.find("table")
rows = table.find_all("tr")

# Separate the header row from the data rows
headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
data = [[td.get_text(strip=True) for td in row.find_all("td")] for row in rows[1:]]

# Create and sort DataFrame
df = pd.DataFrame(data, columns=headers)

# Convert the electrical conductivity to a number for proper sorting
df["Element Electrical Conductivity (S/m)"] = pd.to_numeric(
    df["Element Electrical Conductivity (S/m)"], errors="coerce"
)
# Drop rows without a number for electrical conductivity
df = df.dropna(subset=["Element Electrical Conductivity (S/m)"])

# Sort by descending electrical conductivity, and select top 10
df_top10 = df.sort_values(
    "Element Electrical Conductivity (S/m)", ascending=False
).head(10)

# Reset the index to remove the original row #s
df_top10 = df_top10[
    ["Element Name", "Element Electrical Conductivity (S/m)"]
].reset_index(drop=True)

# Insert ordinal integers 1-10 the 1st column as "Rank"
df_top10.insert(0, "Rank", range(1, len(df_top10) + 1))

# Print the dataframe without any ID column
print(df_top10.to_string(index=False))
