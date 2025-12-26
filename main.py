"""
main.py

Entry point for the Marine Vessel Energy Model + AI extensions.

Options:
1) Classic optimisation (no AI)
2) Natural-language scenario → simulation (AI parameter parser)
3) AI-driven multi-scenario exploration + AI report
"""

from simulation import optimise_speed
from ai_param_parser import parse_user_request_with_llm
from ai_scenario_agent import run_scenarios_and_collect_results
from ai_report_generator import generate_ai_summary


def classic_optimisation():
    print("\n--- Classic optimisation (no AI) ---")
    try:
        distance = float(input("Voyage distance [nm] (default 1000): ") or "1000")
        time_window = float(input("Time window [h] (default 60): ") or "60")
        fuel_type = (input("Fuel type [HFO/LNG/BIO] (default HFO): ") or "HFO").upper()
        with_whr_input = input("Use waste heat recovery? [y/n] (default y): ") or "y"
        with_whr = with_whr_input.lower().startswith("y")
    except ValueError:
        print("Invalid input, using defaults.")
        distance = 1000.0
        time_window = 60.0
        fuel_type = "HFO"
        with_whr = True

    result = optimise_speed(
        distance_nm=distance,
        time_window_h=time_window,
        fuel_type=fuel_type,
        with_whr=with_whr,
    )

    print("\nOptimal result:")
    print(f"  Fuel:      {result['fuel_total_t']:.2f} t")
    print(f"  CO2:       {result['co2_total_t']:.2f} t")
    print(f"  Speed:     {result['speed_knots']:.2f} kn")
    print(f"  Time:      {result['travel_time_h']:.1f} h")
    print(f"  Fuel type: {result['fuel_type']}")
    print(f"  WHR:       {result['with_whr']}")


def ai_natural_language_scenario():
    print("\n--- AI natural-language scenario ---")
    user_prompt = input("Describe your voyage and decarbonisation options:\n> ")

    params = parse_user_request_with_llm(user_prompt)
    print("\nParsed parameters (from AI):")
    for k, v in params.items():
        print(f"  {k}: {v}")

    result = optimise_speed(
        distance_nm=float(params["distance_nm"]),
        time_window_h=float(params["time_window_h"]),
        fuel_type=str(params["fuel_type"]),
        with_whr=bool(params["with_whr"]),
    )

    print("\nOptimal result for this scenario:")
    print(f"  Fuel:      {result['fuel_total_t']:.2f} t")
    print(f"  CO2:       {result['co2_total_t']:.2f} t")
    print(f"  Speed:     {result['speed_knots']:.2f} kn")
    print(f"  Time:      {result['travel_time_h']:.1f} h")
    print(f"  Fuel type: {result['fuel_type']}")
    print(f"  WHR:       {result['with_whr']}")


def ai_agentic_exploration():
    print("\n--- AI-driven multi-scenario exploration ---")
    print("Letting the AI propose scenarios and run simulations...\n")

    results = run_scenarios_and_collect_results()

    for i, item in enumerate(results, start=1):
        sc = item["scenario_input"]
        opt = item["optimal_result"]
        print(f"Scenario {i}: {sc}")
        print(f"  -> Fuel: {opt['fuel_total_t']:.2f} t, "
              f"CO2: {opt['co2_total_t']:.2f} t, "
              f"Speed: {opt['speed_knots']:.2f} kn, "
              f"Time: {opt['travel_time_h']:.1f} h\n")

    print("Generating AI technical summary...\n")
    summary = generate_ai_summary(results)
    print(summary)


def main():
    while True:
        print("\n=== Marine Vessel Energy Model (with AI) ===")
        print("1) Classic optimisation (no AI)")
        print("2) Natural-language scenario (AI parameter parsing)")
        print("3) AI-driven multi-scenario exploration + report")
        print("4) Exit")

        choice = input("Choose an option [1-4]: ").strip()

        if choice == "1":
            classic_optimisation()
        elif choice == "2":
            ai_natural_language_scenario()
        elif choice == "3":
            ai_agentic_exploration()
        elif choice == "4":
            print("Bye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
