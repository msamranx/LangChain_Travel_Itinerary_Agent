# Travel Planner - LangChain Single Agent Project

A beginner-friendly project that demonstrates how to build a **single AI agent** using **LangChain + OpenAI**.

The agent collects travel details from the user, analyzes their preferences using one tool, and then uses a second tool to generate a personalized daily itinerary.

## What You'll Learn

* How LangChain works with LLMs, prompts, tools, messages, and agents
* How to create tools using the `@tool` decorator
* How an agent decides which tool to call and in what order
* How the LLM extracts tool arguments from a user's message
* How `PromptTemplate` can structure LLM input and output
* How one tool's output can be used as input to another tool
* How the agent's tool-calling loop works: **think → act → observe → repeat**
* How `HumanMessage` is used to send user information to a LangChain agent
* How the final `AIMessage` is retrieved from the agent's message history

## How It Works

The project uses a single LangChain agent with two tools:

1. `analyze_trip_preferences`
2. `build_daily_itinerary`

The agent is instructed to analyze the travel preferences before building the itinerary.

```text
User enters travel details
        |
        v
[HumanMessage created]
        |
        v
[Agent examines user request]
        |
        | decides to call
        v
[Tool: analyze_trip_preferences]
        |
        | sends travel details to LLM
        v
[Travel Preference Profile]
        |
        | returned to Agent
        v
[Agent examines tool result]
        |
        | decides to call
        v
[Tool: build_daily_itinerary]
        |
        | preference profile passed to LLM
        v
[Detailed Daily Itinerary]
        |
        | returned to Agent
        v
[Agent generates final response]
        |
        v
Final response returned to user
```

The important concept is that the Python code does **not directly call the two tools in sequence**.

Instead, the tools are made available to the agent:

```python
agent = create_agent(
    model=llm,
    tools=[analyze_trip_preferences, build_daily_itinerary],
    system_prompt=SYSTEM_PROMPT
)
```

The LLM uses the system prompt, tool descriptions, tool parameters, and conversation history to decide which tool to call and what arguments to provide.

## Agent Tools

### 1. `analyze_trip_preferences`

This tool receives the raw travel information entered by the user:

* Destination
* Start date
* End date
* Budget
* Interests
* Travel pace
* Group size
* Travel constraints

It uses an LLM and a `PromptTemplate` to convert this information into a structured **travel preference profile**.

The profile considers factors such as:

* Travel priorities
* Geographical grouping
* Timing needs
* Planning constraints

### 2. `build_daily_itinerary`

This tool receives the preference profile produced by the first tool.

It then asks the LLM to generate a daily itinerary containing:

* An opening note about the destination
* Date and time for activities
* Estimated activity costs
* A closing note about the trip

The itinerary is then returned to the agent.

## Understanding the Agent Loop

The application starts the agent using:

```python
result = agent.invoke({
    "messages": [
        HumanMessage(
            content="Destination: Vietnam, Start Date: ..."
        )
    ]
})
```

`agent.invoke()` starts the agent workflow.

The agent may make multiple LLM calls while completing a single request.

A simplified message flow looks like this:

```text
HumanMessage
     |
     v
AIMessage
(tool call: analyze_trip_preferences)
     |
     v
ToolMessage
(preference profile)
     |
     v
AIMessage
(tool call: build_daily_itinerary)
     |
     v
ToolMessage
(daily itinerary)
     |
     v
AIMessage
(final response)
```

The application retrieves the final message using:

```python
return result["messages"][-1].content
```

`result["messages"]` contains the agent's message history.

`[-1]` selects the last message, and `.content` retrieves its text.

## Prerequisites

* Python 3.10 or higher
* An OpenAI API key

OpenAI API keys can be created from the OpenAI Platform API Keys page.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/NisargKadam/Langchain_sample_project.git

cd Langchain_sample_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows, you can also create or copy the `.env` file manually.

Open `.env` and replace the placeholder with your OpenAI API key:

```text
OPENAI_API_KEY=sk-your-actual-key-here
```

Do not commit your real `.env` file or API key to Git.

## Run

Run the travel planner from the terminal:

```bash
python travel_planner_agent.py
```

The application will ask for your travel preferences:

```text
Your desired destination:
Your planned start date:
Your planned end date:
Your approx budget:
Your vacation interests:
Your desired trip pace (slow/medium/fast):
Your group size:
Any travel constraints:
```

After entering the information, the LangChain agent processes the request and calls the appropriate tools.

Logging is enabled, so the terminal also shows when each tool is executed.

## Example

**Input:**

```text
Your desired destination: Vietnam
Your planned start date: 03/01/2027
Your planned end date: 15/01/2027
Your approx budget: 100000 rupees
Your vacation interests: water sports, drinking coffee
Your desired trip pace (slow/medium/fast): medium
Your group size: 5
Any travel constraints: cannot travel in buses
```

The agent first converts these raw inputs into a travel preference profile.

For example:

```text
Travel Priorities:
- Water sports
- Local coffee experiences

Travel Pace:
- Medium

Group:
- 5 travelers

Planning Constraints:
- Avoid bus transportation
```

The second tool then uses this profile to generate a day-by-day itinerary.

The exact itinerary may vary because it is generated dynamically by the LLM.

## Project Structure

```text
.
├── travel_planner_agent.py    # Main LangChain agent
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── .gitignore                 # Keeps secrets and venv out of Git
└── README.md                  # Project documentation
```

## Tech Stack

* **LangChain** — Framework for building LLM-powered applications and agents
* **OpenAI GPT-4.1-mini** — LLM used by the agent and its tools
* **python-dotenv** — Loads the OpenAI API key from the `.env` file
* **Python** — Application language

## Key LangChain Concepts Demonstrated

### PromptTemplate

`PromptTemplate` provides structured instructions and dynamically inserts travel information into prompts sent to the LLM.

### Tools

The `@tool` decorator converts regular Python functions into tools that the LangChain agent can call.

The function name, docstring, parameters, and type hints help describe the tool to the LLM.

### HumanMessage

`HumanMessage` represents the user's request inside the LangChain message system.

The individual CLI inputs are combined into one `HumanMessage` before being sent to the agent.

### Tool Calling

The agent does not simply execute every available tool.

The LLM examines the conversation, system prompt, available tools, and their argument schemas before deciding which tool to call.

For example, it can generate a structured tool call similar to:

```text
analyze_trip_preferences(
    travel_destination="Vietnam",
    travel_startdate="03/01/2027",
    travel_enddate="15/01/2027",
    travel_budget="100000 rupees",
    travel_interests="water sports, drinking coffee",
    travel_pace="medium",
    travel_groupsize="5",
    travel_constraints="cannot travel in buses"
)
```

LangChain receives this tool-call request and executes the corresponding Python function.

### Tool-to-Tool Information Flow

The first tool returns a preference profile to the agent.

The agent sees that result in its message history and can then use it as the `preference_profile` argument when calling `build_daily_itinerary`.

This demonstrates an important characteristic of agent-based applications:

```text
Tool 1
   |
   v
Agent / LLM
   |
   v
Tool 2
```

The tools are not directly calling each other. The **agent coordinates the workflow between them**.

## Current Limitations

This project is intended primarily as a LangChain learning exercise.

The generated itinerary relies on the LLM's knowledge and does not currently use external travel APIs for:

* Live flight prices
* Hotel availability
* Current attraction prices
* Real-time transportation schedules
* Live weather
* Currency exchange rates

Therefore, prices and travel information generated by the model should be treated as **estimates rather than live booking information**.

A future version of the project could add dedicated APIs or LangChain tools for live travel data.
