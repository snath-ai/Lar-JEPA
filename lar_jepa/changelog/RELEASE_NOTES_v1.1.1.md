# Lár v1.1.1 Patch Release

This patch adds higher-fidelity timestamps to the audit logs.

## Fixes
- **Audit Log Timestamps**: Added a precise `timestamp` field to every *individual step* in the execution history. Previously, only the final run summary was timestamped. This allows for exact replay timing analysis.

## Upgrading
```bash
pip install lar-engine==1.1.1
```
