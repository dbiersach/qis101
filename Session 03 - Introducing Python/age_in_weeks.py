#!/usr/bin/env -S uv run
"""age_in_weeks.py"""

age_years = 0
age_weeks = age_years * 52

print(f"I am {age_years} years old", end=", ")
print(f"which is {age_weeks:,} weeks.")
