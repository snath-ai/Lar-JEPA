"""
Custom Logger and Tracker Example

This example demonstrates how to inject custom AuditLogger and TokenTracker
instances into GraphExecutor for advanced observability control.

New in v1.3.0: Logger and Tracker can now be customized via dependency injection.
"""

from lar import (
    GraphExecutor,
    LLMNode,
    AuditLogger,
    TokenTracker
)

print("=" * 60)
print("Custom Logger/Tracker Example")
print("=" * 60)

# Option 1: Use defaults (automatic creation)
print("\n[Option 1] Using automatic logger/tracker (default)")
executor_default = GraphExecutor(log_dir="my_custom_logs")
# AuditLogger and TokenTracker are created automatically

# Option 2: Inject custom instances (advanced)
print("\n[Option 2] Injecting custom logger/tracker")

custom_logger = AuditLogger(log_dir="advanced_logs")
custom_tracker = TokenTracker()

executor_custom = GraphExecutor(
    logger=custom_logger,
    tracker=custom_tracker
)

# Now you can share the same tracker across multiple executors
executor_shared = GraphExecutor(
    logger=AuditLogger(log_dir="shared_logs"),
    tracker=custom_tracker  # Same tracker instance
)

# Build a simple test graph
node1 = LLMNode(
    model_name="ollama/phi4",
    prompt_template="Count to 3",
    output_key="count"
)

# Run with custom executor
print("\n[Running] Testing custom logger/tracker...")
result = list(executor_custom.run_step_by_step(node1, {}))

# Access the logger directly
print("\n[Logger] Audit history:")
history = custom_logger.get_history()
print(f"  Steps logged: {len(history)}")

# Access the tracker directly
print("\n[Tracker] Token usage:")
summary = custom_tracker.get_summary()
print(f"  Total tokens: {summary['total_tokens']}")
print(f"  By model: {summary['tokens_by_model']}")

# Show that shared tracker aggregates across executors
print("\n[Advanced] Running second executor with shared tracker...")
node2 = LLMNode(
    model_name="ollama/phi4",
    prompt_template="Say hello",
    output_key="greeting"
)
result2 = list(executor_shared.run_step_by_step(node2, {}))

print("\n[Shared Tracker] Aggregated token usage:")
aggregated = custom_tracker.get_summary()
print(f"  Total tokens (both runs): {aggregated['total_tokens']}")

print("\n" + "=" * 60)
print("✅ Custom Logger/Tracker Demo Complete")
print("=" * 60)
print("\nKey Takeaways:")
print("  1. GraphExecutor auto-creates logger/tracker if not provided")
print("  2. You can inject custom instances for advanced control")
print("  3. Shared tracker enables cost aggregation across workflows")
print("  4. Custom logger enables centralized audit trail management")
