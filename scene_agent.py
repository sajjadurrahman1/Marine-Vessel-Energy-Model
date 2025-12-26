"""
ai_scenario_agent.py

Agentic layer:
- Ask LLM to propose multiple decarbonisation scenarios
- Run the simulation for each scenario
- Aggregate numeric results
"""

import json
import os
from typing import List, Dict

from openai import OpenAI

from simulation import optimise_speed

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCENARIO_SYSTEM_PROMPT = """
You are a maritime decarbonisation R&D assistant.

Your task is to propose 3–5 scenarios for a ship energy simulation.
Each scenario should specify:
- fuel_type: one of ["HFO", "LNG", "BIO"]
- with_whr: boolean (true if waste heat recovery is used)
- distance_nm: voyage distance in nautical miles
- time_window_h: maximum travel time in hours

Return ONLY a JSON array of objects, for example:
[
  {"fuel_type": "HFO", "with_whr": false, "distance_nm": 1000, "time_window_h": 60},
  ...
]

Do not include any text before or after the JSON.
"""


def propose_scenarios_with_llm() -> List[Dict]:
    """
    Ask the LLM to propose decarbonisation scenarios.
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SCENARIO_SYSTEM_PROMPT}
        ],
    )

    content = response.output[0].content[0].text

    try:
        scenarios = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON array: {content}") from e

    if not isinstance(scenarios, list):
        raise ValueError("Expected a JSON array of scenarios.")

    return scenarios


def run_scenarios_and_collect_results() -> List[Dict]:
    """
    Use LLM to propose scenarios, then run optimisation for each one.
    Returns a list of result dicts.
    """
    scenarios = propose_scenarios_with_llm()
    results = []

    for sc in scenarios:
        fuel_type = sc.get("fuel_type", "HFO")
        with_whr = bool(sc.get("with_whr", True))
        distance_nm = float(sc.get("distance_nm", 1000.0))
        time_window_h = float(sc.get("time_window_h", 60.0))

        opt = optimise_speed(
            distance_nm=distance_nm,
            time_window_h=time_window_h,
            fuel_type=fuel_type,
            with_whr=with_whr,
        )

        combined = {
            "scenario_input": sc,
            "optimal_result": opt,
        }
        results.append(combined)

    return results
