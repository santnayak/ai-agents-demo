"""
Gateway Trace Demo — with Tool Calls

Sends chat completions through the mudipu-gateway so every request is
intercepted, traced, and forwarded to OpenRouter.

The agent loop handles tool calls locally: when the model requests a tool
the function is executed here and the result is fed back to the model.
All turns (including tool calls) are traced through the gateway.

Usage:
    python run.py

Requirements:
    - mudipu-gateway running at MUDIPU_GATEWAY_URL (default http://localhost:9090)
    - OPENROUTER_API_KEY set in .env (or environment)
    - openai package installed:  pip install openai
    - mudipu SDK installed:      pip install -e <path-to-mudipu-python>
"""

import json
import os
import random
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ── Load env ──────────────────────────────────────────────────────────────────
_here = os.path.dirname(__file__)
_root = os.path.abspath(os.path.join(_here, "../../.."))
load_dotenv(os.path.join(_root, ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GATEWAY_URL = os.getenv("MUDIPU_GATEWAY_URL", "http://localhost:9090")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
    print("❌  OPENROUTER_API_KEY not set. Add it to your .env file.")
    sys.exit(1)

# ── mudipu SDK — session only ─────────────────────────────────────────────────
from mudipu import MudipuTracer, set_config, MudipuConfig

set_config(MudipuConfig(
    enabled=True,
    auto_export=False,
    platform_enabled=False,
))

tracer = MudipuTracer(session_name="Gateway Demo", tags=["demo", "openrouter", "tools"])
session = tracer.start_session()
session_id = str(session.session_id)

TENANT_ID = os.getenv("MUDIPU_TENANT_ID", "")
if not TENANT_ID:
    print("⚠️  MUDIPU_TENANT_ID not set — traces won't be linked to an organization.")
    print("   Add MUDIPU_TENANT_ID=<your-org-uuid> to your .env\n")

print("=" * 60)
print("Mudipu Gateway Trace Demo  (tool calls enabled)")
print("=" * 60)
print(f"  Gateway : {GATEWAY_URL}")
print(f"  Model   : {MODEL}")
print(f"  Session : {session_id}")
print("=" * 60)
print("\nEvery message is routed through the gateway and traced.")
print("Try asking: 'What time is it?', 'Calculate 42 * 17', 'Weather in London'")
print("Type 'quit' to exit.\n")

# ── Tool definitions (OpenAI function-calling schema) ─────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Safely evaluate a mathematical expression. "
                "Supports +, -, *, /, (), %, and basic operators."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '(10 + 5) * 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a city (mock data).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. 'London'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_number",
            "description": "Generate a random integer within a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_val": {
                        "type": "integer",
                        "description": "Minimum value (inclusive, default 1)",
                    },
                    "max_val": {
                        "type": "integer",
                        "description": "Maximum value (inclusive, default 100)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/()%. ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters"
    try:
        return str(eval(expression))  # noqa: S307 — input sanitised above
    except Exception as exc:
        return f"Error: {exc}"


def get_weather(city: str) -> str:
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Windy"]
    temp = random.randint(10, 30)
    condition = random.choice(conditions)
    return f"Weather in {city}: {condition}, {temp}°C"


def get_random_number(min_val: int = 1, max_val: int = 100) -> str:
    return str(random.randint(min_val, max_val))


TOOL_MAP = {
    "get_current_time": lambda args: get_current_time(),
    "calculate": lambda args: calculate(args["expression"]),
    "get_weather": lambda args: get_weather(args["city"]),
    "get_random_number": lambda args: get_random_number(
        args.get("min_val", 1), args.get("max_val", 100)
    ),
}


def run_tool(name: str, arguments_json: str) -> str:
    """Execute a tool by name and return its string result."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        args = {}
    handler = TOOL_MAP.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    try:
        result = handler(args)
        print(f"   🔧 Tool [{name}] → {result}")
        return str(result)
    except Exception as exc:
        return f"Error running {name}: {exc}"


# ── OpenAI client factory → gateway ──────────────────────────────────────────
# We create a new client per user query so that x-mudipu-trace-id is included
# in the default_headers (and therefore reliably sent on every HTTP request in
# the tool-call loop).  Using extra_headers= on .create() can be unreliable
# depending on SDK version.
_gateway_url = f"{GATEWAY_URL}/v1"
_shared_headers = {
    "x-mudipu-session-id": session_id,
    "x-mudipu-tenant-id": TENANT_ID,
    "HTTP-Referer": "https://mudipu.dev",
    "X-Title": "Mudipu Gateway Demo",
}

def make_traced_client(trace_id: str) -> OpenAI:
    """Return a gateway-pointed OpenAI client with this query's trace_id baked in."""
    return OpenAI(
        base_url=_gateway_url,
        api_key=OPENROUTER_API_KEY,
        default_headers={**_shared_headers, "x-mudipu-trace-id": trace_id},
    )

conversation: list[dict] = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant with access to tools. "
            "Use tools whenever they help you give a better answer. "
            "Be concise."
        ),
    },
]

# ── Interactive loop ──────────────────────────────────────────────────────────
try:
    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            # Generate one stable trace_id for this user query so all LLM calls
            # in the tool-call loop share the same Trace record in mudipu.
            import uuid
            trace_id = str(uuid.uuid4())

            # New client with trace_id baked into default_headers — guaranteed
            # to be sent on every HTTP request in the loop below.
            client = make_traced_client(trace_id)

            # ── Agentic tool-call loop ────────────────────────────────────────
            while True:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=512,
                )

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Always append the assistant message (may contain tool_calls)
                conversation.append(message.model_dump(exclude_unset=False))

                if finish_reason == "tool_calls" or message.tool_calls:
                    # Execute each requested tool and append results
                    for tc in message.tool_calls:
                        result = run_tool(tc.function.name, tc.function.arguments)
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    # Loop back to let the model produce its final reply
                    continue

                # No more tool calls — print the final answer
                reply = message.content or ""
                print(f"\n🤖 Assistant: {reply}\n")
                break

        except Exception as exc:
            print(f"\n❌ Error: {exc}\n")
            conversation.pop()  # Remove the failed user message

finally:
    tracer.end_session()
    print(f"\n👋 Session {session_id} ended. Check the mudipu dashboard for traces.")

