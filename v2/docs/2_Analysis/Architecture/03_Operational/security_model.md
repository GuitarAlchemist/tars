# TARS V2 Security Model

**Date:** November 22, 2025
**Status:** Design
**Priority:** P0 (CRITICAL)
**Context:** Defining security boundaries for an agent that executes arbitrary code.

---

## Threat Model

### Attack Vectors

| Threat | Risk Level | Impact |
| :--- | :--- | :--- |
| **Malicious Prompt Injection** | HIGH | Agent executes `rm -rf /` or exfiltrates data |
| **Credential Leakage** | CRITICAL | API keys exposed in logs or code |
| **Lateral Movement** | HIGH | Agent escapes sandbox, accesses host system |
| **Supply Chain Attack** | MEDIUM | Compromised MCP server injects malicious code |
| **Denial of Service** | MEDIUM | Infinite loop consumes all resources |

---

## Defense in Depth: 5-Layer Security

### Layer 1: Process Isolation (Docker)

**Every agent execution runs in a disposable Docker container.**

```yaml
# tars-sandbox/Dockerfile
FROM python:3.11-slim

# Non-root user
RUN useradd -m -u 1000 tars
USER tars

# Read-only filesystem except /workspace
VOLUME /workspace
WORKDIR /workspace

# No network by default
# --network=none (overridden per task)
```

**Key Properties:**

- **Ephemeral**: Container is destroyed after task completion
- **Isolated**: No access to host filesystem or network
- **Resource-limited**: CPU/memory caps enforced

**Docker Compose Integration:**

```yaml
services:
  tars-agent:
    build: ./tars-sandbox
    network_mode: none  # Default: no network
    read_only: true
    tmpfs:
      - /tmp:size=100M
    volumes:
      - ./workspace:/workspace:rw
    security_opt:
      - no-new-privileges:true
      - seccomp=unconfined  # For Python REPL
```

### Layer 2: Filesystem Access Control

**Bind mounts with explicit permissions.**

| Path | Access | Purpose |
| :--- | :--- | :--- |
| `/workspace` | **Read-Write** | User's project directory |
| `/usr/local/lib/python3.11` | **Read-Only** | Python stdlib |
| `/home/tars/.tars` | **Read-Write** | TARS config/state |
| All other paths | **No Access** | Blocked |

**Implementation:**

```fsharp
type FilesystemPolicy = {
    AllowedReadPaths: string list
    AllowedWritePaths: string list
    DeniedPaths: string list
}

let defaultPolicy = {
    AllowedReadPaths = ["/workspace"; "/usr/local/lib"]
    AllowedWritePaths = ["/workspace"; "/home/tars/.tars"]
    DeniedPaths = ["/etc"; "/root"; "/proc"; "/sys"]
}
```

### Layer 3: Network Policy (Firewall)

**Outbound-only, allowlist-based.**

```yaml
# network-policy.yaml (for Kubernetes)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tars-agent-policy
spec:
  podSelector:
    matchLabels:
      app: tars-agent
  policyTypes:
    - Egress
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # Allow HTTPS to approved domains
    - to:
        - podSelector: {}
      ports:
        - protocol: TCP
          port: 443
      # Only to: api.openai.com, api.anthropic.com
```

**For Docker (using iptables):**

```bash
# Block all outbound except:
iptables -A OUTPUT -p tcp --dport 443 -d api.openai.com -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -d api.anthropic.com -j ACCEPT
iptables -A OUTPUT -j DROP
```

### Layer 4: Credential Management (Vault)

**Never store secrets in code or config files.**

```fsharp
type ICredentialVault =
    abstract member GetSecret: key:string -> Async<string option>
    abstract member SetSecret: key:string * value:string -> Async<unit>

// Production: HashiCorp Vault
type VaultCredentialStore(vaultUrl: string) =
    interface ICredentialVault with
        member _.GetSecret(key) = async {
            let! response = Http.get $"{vaultUrl}/v1/secret/data/{key}"
            return Some response.data.value
        }

// Development: Environment Variables
type EnvCredentialStore() =
    interface ICredentialVault with
        member _.GetSecret(key) = async {
            return Environment.GetEnvironmentVariable(key) |> Option.ofObj
        }
```

**Usage in Docker:**

```yaml
services:
  tars-agent:
    environment:
      # Secrets injected at runtime, NEVER committed to git
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
```

**Rotation Policy:**

- API keys rotated every 90 days
- GitHub tokens scoped to minimum permissions (read:repo, write:issues)
- Revoke immediately on suspected compromise

### Layer 5: Human-in-the-Loop (HITL)

**For high-risk operations, require user approval.**

```fsharp
type RiskLevel = Low | Medium | High | Critical

type Operation = {
    Description: string
    Code: string
    Risk: RiskLevel
}

let requiresApproval (op: Operation) =
    match op.Risk with
    | Critical -> true  // Always require approval
    | High -> true      // Self-modification, file deletion
    | Medium -> false   // Code generation, API calls
    | Low -> false      // Read-only operations
```

**UI Flow:**

```
┌─────────────────────────────────────────────┐
│ ⚠️  APPROVAL REQUIRED                       │
├─────────────────────────────────────────────┤
│ TARS wants to execute:                      │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ // Delete all .tmp files                │ │
│ │ Directory.Delete("/workspace/*.tmp")    │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Risk: HIGH                                  │
│ Reason: File deletion                       │
│                                             │
│ [✓ Approve]  [✗ Reject]  [? Explain]       │
└─────────────────────────────────────────────┘
```

---

## Sandboxing Decision Matrix

| Mode | Isolation | Network | Approval |
| :--- | :--- | :--- | :--- |
| **Consultant (External Repos)** | Docker (strict) | Denied by default | Auto-approved (Low/Medium risk) |
| **Architect (Self-Modification)** | Docker (relaxed) | Allowed (curated list) | **ALWAYS requires approval** |
| **Research (Read-Only)** | Optional | Full access | Auto-approved |

---

## Code Review Gate (for Self-Modification)

**Workflow:**

1. Agent generates F# code for new skill
2. Code is saved to `/workspace/.tars/pending/{skillId}.fs`
3. TARS sends approval request to user via TUI
4. User reviews code (with syntax highlighting)
5. If approved:
   - Code is moved to `/workspace/.tars/skills/`
   - F# compiler compiles to DLL
   - IL verification runs
   - DLL loaded into `SkillRegistry`

**Safety Checks (Pre-Compilation):**

```fsharp
let isSafeCode (code: string) : Result<unit, string> =
    [
        checkNoUnsafeKeywords code  // e.g., "System.IO.File.Delete"
        checkNoReflection code      // No Activator.CreateInstance
        checkNoNativeInterop code   // No DllImport
    ]
    |> List.tryFind Result.isError
    |> Option.defaultValue (Ok ())

let unsafeKeywords = [
    "System.IO.File.Delete"
    "System.Diagnostics.Process.Start"
    "System.Runtime.InteropServices"
]
```

---

## Audit Log

**Every security-relevant event is logged.**

```fsharp
type SecurityEvent = {
    Timestamp: DateTime
    EventType: string  // "CodeExecution", "CredentialAccess", "ApprovalRequest"
    UserId: string
    AgentId: string
    Details: Map<string, obj>
    Approved: bool option
}

// Example log entry
{
    Timestamp = "2025-11-22T14:30:00Z"
    EventType = "CodeExecution"
    UserId = "user@example.com"
    AgentId = "agent-consultant-001"
    Details = Map [
        ("Code", "import os; os.listdir('/workspace')")
        ("Risk", "Low")
    ]
    Approved = Some true
}
```

**Log Retention:** 90 days (configurable)
**Log Export:** JSON, sent to SIEM (Security Information and Event Management) system

---

## Implementation Checklist

- [ ] **Week 1**: Docker sandbox with network isolation
- [ ] **Week 2**: Filesystem bind mounts with read-only enforcement
- [ ] **Week 3**: Credential vault integration (start with env vars)
- [ ] **Week 4**: HITL approval UI in TUI
- [ ] **Week 5**: Audit logging + code review gate

---

## References

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [gVisor (Alternative Sandbox)](https://gvisor.dev/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [OpenTelemetry Security Events](https://opentelemetry.io/docs/specs/otel/logs/)
