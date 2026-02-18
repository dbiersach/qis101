#!/usr/bin/env -S uv run
"""electrical_conductivity_macos_safari.py"""

import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

# Set up Safari WebDriver
driver = webdriver.Safari()

# Load the webpage
url = "https://www.schoolmykids.com/learn/periodic-table/electrical-conductivity-of-all-the-elements"
driver.get(url)

# Give the page time to load dynamic content (JavaScript-rendered)
time.sleep(5)

# Use BeautifulSoup to parse the rendered HTML
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# Find the table and extract headers + rows
table = soup.find("table")
rows = table.find_all("tr")

headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
data = [
    [td.get_text(strip=True) for td in row.find_all("td")]
    for row in rows[1:]
    if row.find_all("td")
]

# Create a DataFrame
df = pd.DataFrame(data, columns=headers)

# Print column names to locate the conductivity column
print("Available columns:", df.columns.tolist())

# Identify and process the conductivity column
conductivity_col = [col for col in df.columns if "conductivity" in col.lower()]
if conductivity_col:
    col_name = conductivity_col[0]
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df = df.dropna(subset=[col_name])
    df_sorted = df.sort_values(col_name, ascending=False).head(10)

    print("\nTop 10 Elements by Electrical Conductivity:")
    print(df_sorted[["Element Name", "Element Symbol", col_name]])
else:
    print("No conductivity column found.")
