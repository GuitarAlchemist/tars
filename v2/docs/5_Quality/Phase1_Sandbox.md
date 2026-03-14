# QA Report: Phase 1 - Foundation & Sandbox

**Date:** November 26, 2025
**Version:** v2.0-alpha

## 1. Overview

This document captures the verification evidence for Phase 1 of the TARS v2 implementation, specifically focusing on the Docker Sandbox isolation and the core event bus functionality.

## 2. Test Results

### 2.1 Docker Sandbox Isolation

**Objective:** Verify that the `tars-sandbox` container can execute Python scripts but has **NO** internet access.

**Test File:** `tests/Tars.Tests/SandboxTests.fs`

**Test Cases:**

1. `Can run python script in sandbox`: Runs `print('Hello from Sandbox')`.
    * **Expected:** Exit Code 0, Stdout contains "Hello from Sandbox".
2. `Sandbox has no internet access`: Runs `urllib.request.urlopen('http://google.com')`.
    * **Expected:** Exit Code != 0 (Script fails due to connection error).

**Evidence:**

```text
Passed!  - Failed:     0, Passed:     2, Skipped:     0, Total:     2, Duration: 651 ms - Tars.Tests.dll (net10.0)
```

**Implementation Details:**

* **Image:** `tars-sandbox` (Python 3.11-slim, non-root user).
* **Runtime:** `NetworkMode = "none"` enforced in `Sandbox.fs`.

### 2.2 Core Foundation (Golden Run)

**Objective:** Verify the end-to-end flow of the CLI `demo-ping` command.

**Test File:** `tests/Tars.Tests/GoldenRun.fs`

**Evidence:**

```text
[15:09:28 INF] Starting TARS v2 Demo Ping...
[15:09:28 INF] DemoAgent received: PING
[15:09:31 INF] Ping sent.
DEBUG: Ping sent (Console).
[15:09:28 INF] Publishing message...
```

## 3. Conclusion

Phase 1 requirements for the secure runtime environment have been met and verified with automated tests.
