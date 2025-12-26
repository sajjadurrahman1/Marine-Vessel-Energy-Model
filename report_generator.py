"""
ai_report_generator.py

Generate a technical summary of simulation results using an LLM.
"""

import json
import os
from typing import List, Dict

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REPORT_SYSTEM_PROMPT = """
You are a maritime R&D engineer.

You will be given JSON data from a ship energy simulation comparing
multiple decarbonisation scenarios. Write a clear, concise technical
summary (max ~350 words) including:

- Key numerical findings (fuel use, travel time, CO2 emissions)
- Comparison between scenarios (e.g. HFO vs LNG vs BIO, with/without WHR)
- Trade-offs between fuel consumption, emissions, and time constraints
- A reasoned recommendation of the best scenario for decarbonisation.

Write in a professional, engineering-oriented tone.
"""


def generate_ai_summary(results: List[Dict]) -> str:
    """
    Send simulation results to LLM and get a technical summary as text.
    """
    results_json = json.dumps(results, indent=2)

    prompt = f"Here is the simulation result data in JSON:\n\n{results_json}\n\nWrite the summary now."

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.output[0].content[0].text
