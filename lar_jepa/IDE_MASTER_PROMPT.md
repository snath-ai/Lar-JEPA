# Lár Framework - Master Context
> **Note to AI**: You are coding with **Lár**, a graph-based agent framework that treats "Code as the Graph".
> Before generating any code, you MUST ingest this context.

## Core Principles
1.  **Strict Typing**: Every Node and Function MUST have Pydantic/Python type hints.
2.  **Explicit Linking**: Connect nodes using `.next_node = target` or `RouterNode(path_map={...})`.
3.  **No "Magic"**: Do not assume global state. Use `state.get()` and `state.set()`.
4.  **No Hidden Prompts**: All prompts must be visible in the `prompt_template` argument.

## The 18 Core Patterns
You are expected to recognize and implement these standard Lár patterns:

| Level | Patterns |
| :--- | :--- |
| 🟢 **Basic** | Simple Triage, RAG Chain |
| 🟡 **Logic** | Cyclic Agent, HITL, Memory, Self-Correction, Multi-Agent, Streaming, JSON Mode, Sub-Graphs |
| 🔴 **Advanced** | Dynamic Routing, Parallel Execution (BatchNode), Map-Reduce, Branching, Wait-for-User, Integration Builder |
| 🟣 **Ops** | A/B Tester, Time Traveller (Persistence) |

## Code Patterns to Follow

### 1. Defining a Node
```python
from lar import LLMNode

architect = LLMNode(
    model_name="gemini/gemini-1.5-pro",
    prompt_template="Analyze {input}...",
    output_key="plan"
)
```

### 2. Defining a Tool
```python
from lar import ToolNode

def my_tool(state):
    return "result"

tool = ToolNode(
    tool_function=my_tool,
    input_keys=["__state__"],
    output_key="result",
    next_node=None
)
```

### 3. Defining a Router
```python
from lar import RouterNode

def decide(state):
    return "go_left"

router = RouterNode(
    decision_function=decide,
    path_map={
        "go_left": left_node,
        "go_right": right_node
    }
)
```

### 4. Defining a Parallel Batch
```python
from lar import BatchNode, LLMNode

# 1. Define independent nodes (Fan-Out)
worker_a = LLMNode(..., output_key="res_a")
worker_b = LLMNode(..., output_key="res_b")

# 2. Wrap them in a BatchNode
# They run in parallel threads. State is cloned for each, then merged.
batch = BatchNode(
    nodes=[worker_a, worker_b],
    next_node=aggregator_node
)
```

### 5. Defining a Cleanup Node
```python
from lar import ClearErrorNode

cleanup = ClearErrorNode(
    error_key="error",
    next_node=final_node
)
```

### 6. Defining a Human Jury Node (HITL)
```python
from lar import HumanJuryNode

jury = HumanJuryNode(
    prompt="Should we proceed with this action?",
    choices=["approve", "reject"],
    output_key="user_decision",
    context_keys=["plan", "risk_assessment"],
    next_node=None
)
```

### 7. Defining a Dynamic Node with Safety Validation
```python
from lar import DynamicNode, TopologyValidator

# 1. Define allowed tools for the validator
def safe_tool(state):
    return "safe result"

# 2. Create validator with allowlist
validator = TopologyValidator(allowed_tools=[safe_tool])

# 3. Create DynamicNode (LLM designs subgraph at runtime)
dynamic = DynamicNode(
    llm_model="gemini/gemini-1.5-pro",
    prompt_template="Design a graph to solve: {task}",
    validator=validator,
    next_node=final_node
)
```

## Running Agents
Always include a `if __name__ == "__main__":` block that uses `GraphExecutor` to run the graph instantly for verification.
