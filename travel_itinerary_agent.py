"""Travel Planner Agent — a LangChain agent that analyzes travel preferences
and creates a personalized daily itinerary.

Setup: pip install -r requirements.txt, copy .env.example to .env, add your key.
Run: python travel_planner_agent.py
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trip_planner")

load_dotenv()
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

TRAVEL_PREFERENCE_PROMPT = PromptTemplate(
    input_variables=["travel_destination", "travel_startdate", "travel_enddate", "travel_budget", "travel_interests", "travel_pace", "travel_groupsize", "travel_constraints"],
    template="""You are a travel preference analyzer. Given the following travel details,
create a structured travel preference profile.

Destination: {travel_destination}
Start Date: {travel_startdate}
End Date: {travel_enddate}
Budget: {travel_budget}
Interests: {travel_interests}
Pace: {travel_pace}
Group Size: {travel_groupsize}
Constraints: {travel_constraints}

Analyze the user's travel preferences and create a preference profile keeping in mind the following factors:
- Travel Priorities
- Geographical Grouping
- Timing Needs
- Planning constraints

Return ONLY the preference profile, nothing else.""",
)

ITINERARY_PROMPT = PromptTemplate(
    input_variables=["preference_profile"],
    template="""You are travel itinerary planner, Given the following travel preference profile, prepare a daily itinerary.

Preference Profile: {preference_profile}

Prepare the itinerary with:
- Well-organized opening note about the selected destination (1 short paragraph)
- Date and time for each activity
- Estimate cost for each activity
- Well-organized closing note about the trip (1 short paragraph)

Return ONLY the itinerary, nothing else.""",
)

@tool
def analyze_trip_preferences(travel_destination: str, travel_startdate: str, travel_enddate: str, travel_budget: str, travel_interests: str, travel_pace: str, travel_groupsize: str, travel_constraints: str) -> str:
    """Create a preference profile based on the user's travel idea. Use BEFORE build_daily_itinerary."""
    logger.info("[analyze_trip_preferences] is analyzing preferences for: %s", travel_destination)
    return llm.invoke(TRAVEL_PREFERENCE_PROMPT.format(travel_destination=travel_destination, travel_startdate=travel_startdate, travel_enddate=travel_enddate, travel_budget=travel_budget, travel_interests=travel_interests, travel_pace=travel_pace, travel_groupsize=travel_groupsize, travel_constraints=travel_constraints)).content


@tool
def build_daily_itinerary(preference_profile: str) -> str:
    """Build a daily itinerary based on the travel preference profile. Use AFTER analyze_trip_preferences."""

    logger.info("[build_daily_itinerary] building itinerary")

    result = llm.invoke(
        ITINERARY_PROMPT.format(
            preference_profile=preference_profile
        )
    ).content


    return result
# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """
Act as a thoughtful travel planner.

You MUST:
1. Call analyze_trip_preferences first.
2. Pass its output to build_daily_itinerary.
3. After build_daily_itinerary returns its result, present the COMPLETE
   itinerary returned by that tool to the user.
4. Do not summarize or replace the itinerary with a description.
5. Avoid claiming live availability.
6. Clearly label cost estimates as estimates.
"""
agent = create_agent(model=llm, tools=[analyze_trip_preferences, build_daily_itinerary], system_prompt=SYSTEM_PROMPT)

def run_travel_planner(
    travel_destination: str,
    travel_startdate: str,
    travel_enddate: str,
    travel_budget: str,
    travel_interests: str,
    travel_pace: str,
    travel_groupsize: str,
    travel_constraints: str,
) -> str:

    """Run the agent on user inputs and return the daily itinerary."""
    result = agent.invoke({"messages": [HumanMessage(content=f"Destination: {travel_destination}, Start Date: {travel_startdate}, End Date: {travel_enddate}, Budget: {travel_budget}, Interests: {travel_interests}, Pace: {travel_pace}, Group Size: {travel_groupsize}, Constraints: {travel_constraints}")]})
    return result["messages"][-1].content


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def get_input(prompt: str) -> str:
    """Get user input and exit gracefully if the user wants to quit."""
    value = input(prompt).strip()

    if value.lower() in ("quit", "exit", "q"):
        print("\nExiting Travel Planner. Goodbye!")
        sys.exit(0)

    return value

def main() -> None:
    print("\nTRAVEL PLANNER AGENT (LangChain + OpenAI)")
    print("Enter your travel details. Type 'quit' at any time to exit.\n")

    while True:
        travel_destination = get_input("Your desired destination: ")
        travel_startdate = get_input("Your planned start date: ")
        travel_enddate = get_input("Your planned end date: ")
        travel_budget = get_input("Your approx budget: ")
        travel_interests = get_input("Your vacation interests: ")
        travel_pace = get_input(
            "Your desired trip pace (slow/medium/fast): "
        )
        travel_groupsize = get_input("Your group size: ")
        travel_constraints = get_input("Any travel constraints: ")

        try:
            dailyitinerary = run_travel_planner(
                travel_destination,
                travel_startdate,
                travel_enddate,
                travel_budget,
                travel_interests,
                travel_pace,
                travel_groupsize,
                travel_constraints
            )

            print("\n" + "=" * 60)
            print(dailyitinerary)
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error("Agent failed: %s", e)

if __name__ == "__main__":
    main()
