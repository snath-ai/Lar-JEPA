# Release Notes: Lár v1.5.1

## Cryptographic Audit Logs

Lár v1.5.1 introduces **Cryptographic Audit Logs**, a critical feature for enterprise compliance (EU AI Act, SOC2, HIPAA).

While Lár has always provided a "Glass Box" execution trace via the `AuditLogger`, v1.5.1 allows you to cryptographically sign these JSON logs using an HMAC-SHA256 signature. 

### Why this matters:
If an agent executes a high-stakes action (like a financial transaction or a medical data routing), you don't just need a log of what happened—you need mathematical proof that the log wasn't tampered with after the fact. 

### How to use it:
Pass an `hmac_secret` string when initializing your `GraphExecutor`.

```python
from lar import GraphExecutor

# The engine will automatically sign the final AuditLog output
executor = GraphExecutor(log_dir="compliance_logs", hmac_secret="super_secret_key")
```

The resulting `JSON` log file will contain a `signature` payload. If any value in the JSON file (like the token cost, the nodes visited, or the LLM's reasoning trace) is altered, the signature verification will fail.

### Examples
We have added three new compliance patterns to demonstrate this:
*   [8_hmac_audit_log.py](examples/compliance/8_hmac_audit_log.py): Demonstrates running an agent, generating the signed log, and writing a manual verification hook to prove the payload is untampered.
*   [9_high_risk_trading_hmac.py](examples/compliance/9_high_risk_trading_hmac.py): A more complex mock algorithmic trading example showing how to secure high-stakes LLM execution paths with KMS keys.
*   [10_pharma_clinical_trials_hmac.py](examples/compliance/10_pharma_clinical_trials_hmac.py): Demonstrates securing patient data routing (GxP / FDA 21 CFR Part 11) using securely signed ledgers.

## Minor Fixes
*   Fixed a bug where `FunctionalNode` required a `name` kwarg during instantiation.
