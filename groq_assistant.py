import os
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chat history for context
history = []

def ask_groq(prompt):
    """Send a message to Groq LLM and get a response"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history + [{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error] {e}"

def main():
    console.print(Panel("🤖 [bold]Groq AI Personal Assistant[/] — type 'quit' to exit.", expand=False))

    while True:
        user_input = Prompt.ask("[bold cyan]You[/]")
        if user_input.lower() in {"quit", "exit"}:
            console.print("Goodbye! 👋")
            break

        # Call Groq API
        answer = ask_groq(user_input)
        # Append to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        console.print(Panel(answer, title="Assistant"))

if __name__ == "__main__":
    main()