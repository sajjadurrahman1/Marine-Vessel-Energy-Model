"""
ai_param_parser.py

Use an LLM to convert natural-language descriptions of a voyage
into structured parameters for the simulation.
"""

import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are an assistant that extracts parameters for a ship energy simulation.

Given a user's natural-language request, you MUST respond with a single valid JSON object
(with no extra text) containing these keys:

- distance_nm (float)      : voyage distance in nautical miles
- time_window_h (float)    : maximum allowed travel time in hours
- fuel_type (string)       : one of ["HFO", "LNG", "BIO"]
- with_whr (bool)          : true if waste heat recovery should be enabled
- scenario (string)        : one of ["cruise", "port", "maneuvering"]

If some values are not specified, choose reasonable defaults:
- distance_nm: 1000
- time_window_h: 60
- fuel_type: "HFO"
- with_whr: true
- scenario: "cruise"
"""


def parse_user_request_with_llm(user_prompt: str) -> dict:
    """
    Send user prompt to LLM and parse returned JSON with parameters.

    Raises ValueError if JSON cannot be parsed.
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.output[0].content[0].text
    try:
        params = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {content}") from e

    # Basic sanitising / filling defaults
    params.setdefault("distance_nm", 1000.0)
    params.setdefault("time_window_h", 60.0)
    params.setdefault("fuel_type", "HFO")
    params.setdefault("with_whr", True)
    params.setdefault("scenario", "cruise")

    return params
