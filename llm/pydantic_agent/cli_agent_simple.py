import os
import subprocess
import warnings
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# ---- Configuration ----
MODEL = "google-gla:gemini-2.5-pro"

system_prompt = """
You are a command-line assistant with real access to the system.

You can:
- Run actual shell commands using the `run_shell` tool (bash is available).
- Read and write files on disk.
- Perform web searches when asked about online information.

Whenever the user asks to perform a command-line or system task (like listing files, checking disk usage, etc.),
ALWAYS call the `run_shell` tool with the appropriate command (e.g., 'ls', 'pwd', 'cat').
Trust the tool output as the real command result.
Never claim a command is unavailable or simulated.
Keep responses concise and factual.
"""

if "GOOGLE_API_KEY" not in os.environ:
    warnings.warn("GOOGLE_API_KEY environment variable is not set. Gemini model may not work properly.")

agent = Agent(MODEL, system_prompt=system_prompt)

# ---- Tool definitions ----

class ShellArgs(BaseModel):
    """Arguments for running shell commands."""
    command: str

@agent.tool
def run_shell(ctx: RunContext[None], args: ShellArgs) -> str:
    """Execute a real shell command in bash and return its output."""
    print("Executing shell command:", args.command)
    try:
        result = subprocess.run(
            args.command,
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )
        output = (result.stdout + result.stderr).strip()
        return output or "(no output)"
    except Exception as e:
        return f"Error running shell command: {e}"

class FileReadArgs(BaseModel):
    path: str

@agent.tool
def read_file(ctx: RunContext[None], args: FileReadArgs) -> str:
    """Read and return the contents of a file."""
    try:
        with open(args.path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

class FileWriteArgs(BaseModel):
    path: str
    content: str

@agent.tool
def write_file(ctx: RunContext[None], args: FileWriteArgs) -> str:
    """Write text content to a file."""
    try:
        with open(args.path, "w", encoding="utf-8") as f:
            f.write(args.content)
        return f"Wrote {len(args.content)} bytes to {args.path}"
    except Exception as e:
        return f"Error writing file: {e}"

class WebSearchArgs(BaseModel):
    query: str

@agent.tool
def web_search(ctx: RunContext[None], args: WebSearchArgs) -> str:
    """Perform a simple (simulated) web search."""
    # TODO: Integrate real API like DuckDuckGo or SerpAPI
    return f"Simulated search results for: {args.query}"

# ---- CLI Loop ----

def main():

    print("Pydantic-AI CLI Agent (Gemini)")
    print("Type 'exit' or Ctrl+C to quit.\n")

    # Maintain conversation context between turns
    ctx = None

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting...")
                break

            # Run agent with context for multi-turn conversation
            result = agent.run_sync(
                user_input,
                message_history=ctx.new_messages() if ctx else None
            )
            print("-" * 40)
            print(result.output)
            print("-" * 40)
            print(result.all_messages())
            print("-" * 40)
            ctx = result
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
