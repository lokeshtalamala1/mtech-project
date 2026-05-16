from agent import run_agent
# from mcp.server.fastmcp import FastMCP

def main():
    print("🧠 MCP Security Agent (type 'exit' to quit)\n")

    while True:
        user_input = input(">> ")

        if user_input.lower() == "exit":
            break

        response = run_agent(user_input)

        print("\n🔍 Response:\n")
        print(response)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()