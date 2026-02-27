# Reasoning Models in Lár (System 2)

Lár v1.4.1+ supports **Reasoning Models** (also known as "System 2" models), which generate a "Chain of Thought" or "Reasoning Trace" before their final answer.

Supported models include:
- **DeepSeek R1** (API & Distilled/Ollama)
- **OpenAI o1** (preview/mini)
- **Liquid Thinking** (`ollama/liquid-thinking`)

## How it Works

Lár automatically detects reasoning content and separates it from the final answer to keep your application logic clean while preserving a full audit trail.

1.  **Metadata Capture (Standard)**:
    If the model provider (e.g., DeepSeek API, OpenAI, OpenRouter) returns a dedicated `reasoning_content` field, Lár captures it in `state["__last_run_metadata"]["reasoning_content"]`.

2.  **Regex Fallback (Local/Raw)**:
    If the model outputs raw text with `<think>...</think>` tags (common in local Ollama setups), Lár extracts the content inside the tags, moves it to metadata, and cleans the final answer.

## Examples

- `1_deepseek_r1.py`: Using DeepSeek R1 via Ollama.
- `2_openai_o1.py`: Using OpenAI's o1 model series.
- `3_liquid_thinking.py`: Specific example for Liquid Thinking models.
