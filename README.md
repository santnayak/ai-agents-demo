# AI Agent Learning Project

My personal project for learning AI agent patterns and debugging tools. Building from simple agents to more complex architectures.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your-key-here"  # or OPENAI_API_KEY, ANTHROPIC_API_KEY

# Run simple agent
cd examples/01_basics/simple_agent
python run.py

# View trace
open traces/*.html
```

## What's Implemented

**Simple Agent** (`examples/01_basics/simple_agent/`)
- Basic agent loop with tool calling
- Supports OpenAI, Anthropic, Gemini
- Tools: calculator, time, random number, weather (mock), web search (mock)
- Auto-generates execution traces (JSON + HTML)

See the [simple_agent README](examples/01_basics/simple_agent/README.md) for details.

## Tracing with Mudipu SDK

All agents use my [mudipu-sdk-python](mudipu-sdk-python/) for tracing:
- Captures complete request/response payloads
- Tracks tool executions with timing
- Shows token usage per turn
- Exports to JSON and HTML automatically

Check traces in the `traces/` directory after running an agent.

## Project Structure

```
examples/01_basics/simple_agent/  # Implemented
examples/shared/llm/              # Multi-provider LLM clients  
examples/shared/schemas/          # Data models
mudipu-sdk-python/                # Custom tracing SDK
```

## Planned Examples

Future implementations I'm working on:
- **Patterns**: Planning agents, RAG, ReAct, multi-agent systems
- **Failures**: Context explosion, infinite loops, tool hallucination
- **Safety**: Input validation, sandboxing, dangerous tool handling
- **Advanced**: Memory, self-correction, parallel execution

## License

MIT