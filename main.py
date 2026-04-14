import uuid
from typing import Any

from langgraph.types import Command

from supervisor import supervisor_agent


def main():
    print("Multi-Agent Research System (type 'exit' to quit)")
    print("-" * 50)

    # Create a unique thread ID for this conversation session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        stream_agent(supervisor_agent, {"messages": [("user", user_input)]}, config)
        handle_hitl(supervisor_agent, config)


def stream_agent(agent, inputs, config):
    for chunk in agent.stream(inputs, config=config):
        print_tool_calls_from_model_output(chunk)
        print_tool_results_output(chunk)


def print_tool_calls_from_model_output(chunk):
    if "model" in chunk and "messages" in chunk["model"]:
        for msg in chunk["model"]["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    args_str = ", ".join(
                        f'{k}="{truncate_content(str(v))}"'
                        for k, v in call["args"].items()
                    )
                    print(f"🔧 {call['name']}({args_str})")

            if hasattr(msg, "content") and msg.content:
                print(f"\nAgent: {msg.content}\n")


def print_tool_results_output(chunk):
    if "tools" in chunk and "messages" in chunk["tools"]:
        for msg in chunk["tools"]["messages"]:
            content = msg.content if hasattr(msg, "content") else str(msg)

            lines = content_to_lines(content)

            if lines and len(lines) > 1:
                print(f"  📎 {truncate_content(lines[0])}")
                for line in lines[1:3]:
                    if line.strip():
                        print(f"     {truncate_content(line.strip())}")
                if len(lines) > 4:
                    print("     ...")
            else:
                print(f"  📎 {truncate_content(content)}")
            print()


def content_to_lines(content: str | Any) -> list[str] | list[Any]:
    return content.strip().splitlines() if isinstance(content, str) else []


def truncate_content(content: str | Any) -> str | Any:
    return content[:200] + "..." if isinstance(content, str) and len(content) > 200 else content


def handle_hitl(agent, config):
    """Check for pending interrupts and handle approve/edit/reject loop."""
    while True:
        state = agent.get_state(config)
        if not state.interrupts:
            break

        interrupt = state.interrupts[0]

        action_requests = interrupt.value.get("action_requests", [])
        if not action_requests:
            break

        action = action_requests[0]
        tool_name = action.get("name", "unknown")
        tool_args = action.get("args", {})

        print("\n" + "=" * 60)
        print("⏸️  ACTION REQUIRES APPROVAL")
        print("=" * 60)
        print(f"  Tool    : {tool_name}")

        filename = tool_args.get("filename", "report.md")
        content = tool_args.get("content", "")

        if tool_name == "save_report":
            print(f"  File    : {filename}")
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(f"\n--- Report preview ---\n{preview}\n--- End preview ---\n")
        else:
            import json
            print(f"  Args    : {json.dumps(tool_args, ensure_ascii=False)[:400]}")

        decision = input("  👉 approve / edit / reject: ").strip().lower()

        if decision == "approve":
            handle_approve(agent, config, filename)
            break

        elif decision == "edit":
            feedback = input("  ✏️  Your feedback: ").strip()
            if not feedback:
                print("  Feedback cannot be empty.")
                continue
            handle_edit(agent, config, feedback)

        elif decision == "reject":
            handle_reject(agent, config)
            break

        else:
            print("  Please enter 'approve', 'edit', or 'reject'.")


def handle_approve(agent, config, filename):
    resume_cmd = Command(resume={"decisions": [{"type": "approve"}]})
    stream_agent(agent, resume_cmd, config)
    print(f"\n  ✅ Approved! Report saved to output/{filename}")


def handle_edit(agent, config, feedback):
    resume_cmd = Command(resume={
        "decisions": [{
            "type": "reject",
            "message": (
                f"User requested edits to the report. Feedback: {feedback}\n"
                f"Revise the report accordingly and call save_report again "
                f"with the same filename."
            ),
        }]
    })
    print("\n  ✏️  Revising report based on feedback...\n")
    stream_agent(agent, resume_cmd, config)


def handle_reject(agent, config):
    resume_cmd = Command(resume={
        "decisions": [{"type": "reject", "message": "User rejected the report."}]
    })
    stream_agent(agent, resume_cmd, config)
    print("\n  ❌ Report saving cancelled.")


if __name__ == "__main__":
    main()
