# Demo 1: Simple Agent

This demo showcases a basic AI agent with simple tools for common tasks.

## Features

- **Simple Agent Architecture**: A straightforward agent that can use tools to complete tasks
- **Basic Tools**: 
  - `get_current_time`: Get the current date and time
  - `calculate`: Perform mathematical calculations
  - `get_random_number`: Generate random numbers within a range
  - `get_weather`: Get weather information (mock data)
  - `search_web`: Search the web (mock implementation)
- **Mudipu Tracing**: Built-in execution tracing using Mudipu SDK for comprehensive observability
  - Automatic LLM call tracing
  - Automatic tool execution tracing
  - JSON and HTML trace exports
  - Rich performance metrics and statistics
- **Multiple LLM Support**: Works with OpenAI, Anthropic, or Google Gemini models

## Architecture

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │ Query
       ▼
┌─────────────────────────────────┐
│      Simple Agent               │
│  ┌──────────────────────────┐  │
│  │  Conversation History    │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │    LLM Client            │  │
│  │  (OpenAI/Anthropic/      │  │
│  │   Gemini)                │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │    Tool Registry         │  │
│  └──────────────────────────┘  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│         Tools                   │
│  • Time   • Calculator          │
│  • Random • Weather (mock)      │
│  • Web Search (mock)            │
└─────────────────────────────────┘
```

## How It Works

1. **User Input**: User provides a query or task
2. **Agent Processing**: 
   - Agent sends conversation history + available tools to LLM
   - LLM decides if it needs to use any tools or can respond directly
3. **Tool Execution**: If tools are needed:
   - Agent executes the requested tools
   - Tool results are added back to the conversation
   - Process repeats until task is complete
4. **Response**: Agent returns final response to user

## Running the Demo

```bash
# From the simple_agent directory
python run.py
```

## Viewing Traces

After running the agent, traces are automatically saved in the `traces/` directory.

**View traces with Mudipu analyzer:**
```bash
# If you have mudipu CLI installed
mudipu view traces/

# View specific trace
mudipu view traces/session_abc123_trace.json

# Generate HTML report
mudipu analyze traces/ --html report.html
```

**Or view JSON traces directly:**
```bash
cat traces/*.json | python -m json.tool
```

## Example Interactions

**Example 1: Time Query**
```
You: What time is it?
Agent: [Uses get_current_time tool]
Response: "The current time is 2026-02-22 14:30:45"
```

**Example 2: Calculation**
```
You: Calculate 15% of 250
Agent: [Uses calculate tool with expression "250 * 0.15"]
Response: "15% of 250 is 37.5"
```

**Example 3: Multi-step Task**
```
You: Get me a random number between 1-100 and calculate its square
Agent: [Uses get_random_number tool, gets 42]
      [Uses calculate tool with expression "42 * 42"]
Response: "I got the random number 42, and its square is 1764"
```

## Configuration

Set environment variables in `.env` file (see `.env.example` in project root):

- `LLM_PROVIDER`: Choose "openai", "anthropic", or "gemini"
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI)
- `ANTHROPIC_API_KEY`: Your Anthropic API key (if using Anthropic)
- `GEMINI_API_KEY`: Your Google Gemini API key (if using Gemini)

## Key Concepts

- **Tool Registration**: Tools are registered with the agent's tool registry
- **Conversation History**: Maintains context across interactions
- **Agentic Loop**: Agent iteratively calls LLM and executes tools until task is complete
- **Tracing**: Execution traces help debug and understand agent behavior

## Trace Analysis & Visualization 🔍

The agent automatically saves execution traces to the `traces/` directory in both JSON and HTML formats:

```bash
# View HTML trace in browser (auto-generated)
open traces/*.html  # Mac
start traces\\*.html  # Windows

# Or view JSON for programmatic analysis
cat traces/*.json | python -m json.tool
```

**What You'll See:**
- 🎯 Complete request payloads (messages + tools)
- 💬 Full LLM responses with tool calls
- 🔧 Tool execution details (args + results + timing)
- 📊 Token usage per turn and session totals
- 🔄 Session-based analysis with timeline view
- 📄 HTML export for sharing and archiving
- 🌲 Hierarchical view: Session → Run → Turn → Step

**Example Output:**
```
┌─────────────────────────────────────────────────┐
│            🔍 Trace Analysis                    │
├─────────────────────────────────────────────────┤
│ Status: completed                               │
│ Duration: 1234.56ms                             │
│ Steps: 8                                        │
└─────────────────────────────────────────────────┘

Execution Timeline
└── Turn 1
    ├── 🤖 llm_call (234.5ms)
    ├── 🔧 tool_call - calculate (12.3ms)
    └── 🤖 llm_call (345.6ms)
```

See the [Mudipu SDK documentation](../../../mudipu-sdk-python/README.md) for complete tracing details.

## Next Steps

Check out:
- **Demo 2**: Complex tools with planning and execution
- **Demo 3**: Agentic RAG with vector search and retrieval
