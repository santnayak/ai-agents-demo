"""Run the simple agent demo."""
import os
from dotenv import load_dotenv

from agent import SimpleAgent
from tools import create_simple_tools


def main():
    """Run the simple agent demo."""
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("Simple Agent Demo")
    print("=" * 60)
    print("\nThis demo showcases a simple agent with basic tools:")
    print("- get_current_time: Get current date/time")
    print("- calculate: Perform mathematical calculations")
    print("- get_random_number: Generate random numbers")
    print("- get_weather: Get weather info (mock)")
    print("- search_web: Search the web (mock)")
    print("\n" + "=" * 60)
    
    # Choose provider
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("OPENAI_MODEL") if provider == "openai" else os.getenv("ANTHROPIC_MODEL") if provider == "anthropic" else os.getenv("GEMINI_MODEL")
    print(f"\nUsing LLM provider: {provider}")
    
    # Create agent
    agent = SimpleAgent(
        provider=provider,
        model=model,
        temperature=0.7,
        max_iterations=10,
        verbose=True
    )
    
    # Register tools
    tools = create_simple_tools()
    for tool in tools:
        agent.register_tool(tool)
    
    print(f"\nRegistered {len(tools)} tools")
    print("\n" + "=" * 60)
    
    # Example queries
    example_queries = [
        "What time is it?",
        "Calculate 123 * 456 + 789",
        "Give me a random number between 1 and 1000",
        "What's the weather like in San Francisco?",
        "Calculate the square root of 144 (hint: 12*12)",
    ]
    
    print("\nExample queries you can try:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\n" + "=" * 60)
    print("\nType 'quit' or 'exit' to end the conversation")
    print("=" * 60)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            # Run agent
            response = agent.run(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
