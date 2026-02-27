# Metacognition Examples

This directory contains examples of **self-modifying agents** - AI systems that can dynamically change their own behavior at runtime using the `DynamicNode` primitive.

## What is Metacognition in Lár?

Metacognition means an agent can:
1. Analyze a problem
2. Design a custom solution graph
3. Validate it for safety
4. Execute the generated graph
5. Return control to the main flow

This is fundamentally different from static graphs where all nodes are defined at development time.

## When to Use Metacognition

**Use DynamicNode when:**
- Problem complexity varies widely (simple vs. deep research)
- You need runtime code generation (calculations, data transforms)
- Self-healing is required (automatic error recovery)
- Modular sub-agents need to be loaded dynamically

**DO NOT use DynamicNode when:**
- Problem structure is known and fixed
- Security is paramount (dynamic graphs are inherently riskier)
- Debugging is critical (static graphs are easier to trace)
- Performance matters (dynamic graph building has overhead)

## Security Considerations

All metacognition examples use `TopologyValidator` to enforce safety:

**What Validator Prevents:**
- Infinite loops (cycle detection)
- Unauthorized tool usage (allowlist enforcement)
- Malformed graphs (structural validation)

**What Validator CANNOT Prevent:**
- Malicious code in allowed tools
- Prompt injection in LLM-generated graphs
- Resource exhaustion (lots of nodes)

**Production Checklist:**
1. Always use `TopologyValidator` with a minimal `allowed_tools` list
2. Sandbox any code execution (Docker, e2b, WebAssembly)
3. Set `max_nodes` limits in validator (not yet implemented)
4. Log all generated graphs to audit trail
5. Add circuit breakers for healing attempts
6. Test with adversarial prompts

## Examples Overview

| Example | Capability | Risk Level | Production Ready |
|---------|-----------|------------|------------------|
| 1_dynamic_depth.py | Adaptive complexity | Low | Yes with limits |
| 2_tool_inventor.py | Code generation | HIGH | No - needs sandboxing |
| 3_self_healing.py | Error recovery | Medium | Yes with circuit breaker |
| 4_adaptive_deep_dive.py | Recursive research | Medium | Yes with depth limits |
| 5_expert_summoner.py | Sub-agent loading | Low | Yes |

## Running the Examples

All examples use local Ollama models by default for cost-free testing:

```bash
# Install Ollama: https://ollama.ai
ollama pull phi4

# Run an example
python 1_dynamic_depth.py
```

To use OpenAI or other providers, modify the `llm_model` parameter in each example.

## Understanding the Code Structure

Every metacognition example follows this pattern:

```python
# 1. Define allowed tools
def safe_tool():
    return "result"

# 2. Create validator
validator = TopologyValidator(allowed_tools=[safe_tool])

# 3. Define the metacognitive prompt
PROMPT = """
Design a graph to solve: {problem}
Output JSON with nodes and entry_point.
"""

# 4. Create DynamicNode
dynamic_architect = DynamicNode(
    llm_model="ollama/phi4",
    prompt_template=PROMPT,
    validator=validator,
    next_node=exit_node,
    context_keys=["problem"]
)

# 5. Execute
executor.run_step_by_step(dynamic_architect, {"problem": "..."})
```

## Debugging Generated Graphs

To see what graph the LLM generated:

```python
executor = GraphExecutor(log_dir="debug_logs")
list(executor.run_step_by_step(dynamic_node, initial_state))

# Check the audit log
import json
with open("debug_logs/run_<id>.json") as f:
    log = json.load(f)
    
# Look for DynamicNode step
for step in log["steps"]:
    if step["node"] == "DynamicNode":
        print(json.dumps(step["state_diff"]["added"]["_generated_graph_spec"], indent=2))
```

## Common Patterns

### Pattern 1: Adaptive Worker Count

Problem varies in complexity, allocate resources accordingly:
```python
# Simple query: 1 researcher
# Complex query: 5 parallel researchers
```
See: `1_dynamic_depth.py`

### Pattern 2: Self-Programming

Generate and execute code for novel computations:
```python
# User: "Calculate 20th Fibonacci"
# Agent: Writes fib function → Executes it → Returns 6765
```
See: `2_tool_inventor.py`

### Pattern 3: Error Recovery

Detect failures and synthesize recovery procedures:
```python
# DB connection fails → Rotate credentials → Retry
```
See: `3_self_healing.py`

### Pattern 4: Expert Dispatch

Route to specialized sub-agents based on domain:
```python
# Legal question → Load legal_expert.json
# Medical question → Load medical_expert.json
```
See: `5_expert_summoner.py`

## Limitations

**Current DynamicNode Implementation:**
- Only supports LLMNode and ToolNode factories
- BatchNode/RouterNode must be built manually
- No native support for loading serialized graphs (workaround in example 5)

**Planned for v1.4:**
- Full node type support in factory
- `LoaderNode` for JSON graph files
- `max_nodes` parameter in TopologyValidator
- Graph simplification/optimization

## Best Practices

1. **Start Static, Add Dynamic Later**: Build your core graph statically. Add DynamicNode only where variability is needed.

2. **Fail Gracefully**: Always provide a `default_node` in case validation fails.

3. **Log Everything**: DynamicNode automatically logs generated graphs to audit trail. Review them.

4. **Test Edge Cases**: What if LLM generates an empty graph? A single-node graph? A graph with 100 nodes?

5. **Limit Recursion**: If using DynamicNode inside a generated graph, set a `max_depth` in your state.

## Further Reading

- API Documentation: `docs/api-reference/dynamicnode.md`
- Topology Validation: `docs/api-reference/topologyvalidator.md`
- Core Concepts: `docs/core-concepts/9-metacognition.md`
- Glass Box Philosophy: `docs/core-concepts/1-philosophy.md`

## Questions?

Open an issue: https://github.com/snath-ai/lar/issues
