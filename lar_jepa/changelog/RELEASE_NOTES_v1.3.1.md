# Lár v1.3.1 Release Notes

**"The Security Patch"**

This patch release addresses a critical security gap in the `TopologyValidator`'s structural integrity check.

## Security Fixes

### 1. Structural Integrity Enforcement
- **Fix**: The `TopologyValidator` now correctly raises a `SecurityError` if a node in a dynamic graph links to a non-existent `next` node. Previously, this check was a placeholder (`pass`).
- **Impact**: prevents "broken link" attacks or malformed dynamic graphs from executing partially and entering undefined states.

## Verification
- Verified by `tests/test_v1_3_features.py::test_validator_structural_integrity`.
