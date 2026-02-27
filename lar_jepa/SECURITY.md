# Security & Compliance

Lár is designed to be the "Glass Box" alternative to "Black Box" agent frameworks. 
For enterprise deployments in regulated industries (FinTech, Healthcare, Defense), we prioritize **determinism**, **auditability**, and **air-gap capability**.

---

## 1. Security By Design

### A. Immutable Audit Logs (compliance)
Every Lár agent generates a cryptographically-linkable JSON Flight Log (`lar_logs/*.json`) for every run.
*   **What is logged**: Every input, prompt, tool execution, and output.
*   **Why**: This enables 21 CFR Part 11 compliance capability. You can prove exactly *why* an agent took an action.
*   **Mechanism**: State transitions are immutable.

### B. Air-Gap & Data Sovereignty
Lár is fully decoupled from the internet.
*   **No Telemetry**: The engine sends zero data to Snath AI or any third party.
*   **Local Models**: Switch from `gpt-4` to `ollama/phi4` or `deepseek-coder` in one line.
*   **Offline Mode**: Verify your agent's safety without ever sending bytes to a cloud provider.

### C. Pattern 10: The Security Firewall
We provide a standard architectural pattern (Pattern 10) for preventing jailbreaks.
*   **Mechanism**: `GuardrailNode` (Regex/Keyword) -> `Router` -> `LLM`.
*   **Benefit**: Blocks malicious inputs (e.g., "Ignore previous instructions") *before* they reach the LLM, saving cost and preventing unauthorized actions.
*   **Example**: See [`examples/10_security_firewall.py`](examples/10_security_firewall.py).

---

## 2. Supported Versions

Security patches and bug fixes are applied to the `latest minor release` of the Lár Engine framework.

| Status      | Version Format                               | Security Support                                         |
|-------------|-----------------------------------------------|-----------------------------------------------------------|
| Supported   | Latest Minor Release (e.g., 1.0.x)            | Full patches and critical bug fixes are provided.         |
| Unsupported | Previous Minor Releases (e.g., 0.x.x)         | No new security patches will be issued. Users should upgrade immediately. |

---

## 3. Reporting a Vulnerability

We take the security of the *Lár Engine* framework and its users seriously. If you discover a vulnerability, we ask that you follow the process of **responsible disclosure**.

- **Reporting Channel**: Please report vulnerabilities exclusively via email to: [**axdithya@snath.ai**](mailto:axdithya@snath.ai) or [**vinay@snath.ai**](mailto:vinay@snath.ai)
- **Response Time**: We commit to acknowledging receipt of your email within 24 hours and providing an initial assessment of the vulnerability within 48 hours.
- **Public Disclosure**: **Do not open public GitHub Issues, Pull Requests, or discuss vulnerabilities on public forums (e.g., social media or Discord)**. Public disclosure before a fix is released endangers all users.