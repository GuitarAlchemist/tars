# Phase 12: Web of Things (WoT) Integration

**Status**: 🔜 Planned (Q1 2025)  
**Goal**: Enable TARS to discover, understand, and interact with IoT devices using W3C WoT Thing Descriptions.

---

## Overview

Phase 12 extends TARS's capabilities into the physical world by integrating with IoT devices through the W3C Web of Things standard.

**Reference**: https://w3c.github.io/wot-thing-description/

WoT Thing Descriptions provide a standardized way to describe IoT device capabilities, enabling TARS to reason about and interact with physical systems.

---

## 12.1 Thing Description Parser

**Priority**: High  
**Files**: `Tars.Connectors/WoT/ThingDescriptionParser.fs` (new)

| Task | Status | Description |
|------|--------|-------------|
| Parse JSON-LD Thing Description | 🔜 | Standard format support |
| Extract affordances | 🔜 | Properties, actions, events |
| Map TD security definitions | 🔜 | To credential vault |
| TD templates and composition | 🔜 | Reusable descriptions |

### Example Thing Description

```json
{
  "@context": "https://www.w3.org/2022/wot/td/v1.1",
  "id": "urn:dev:ops:32473-WoTLamp-1234",
  "title": "MyLampThing",
  "securityDefinitions": {
    "basic_sc": {"scheme": "basic", "in": "header"}
  },
  "security": "basic_sc",
  "properties": {
    "status": {
      "type": "string",
      "forms": [{"href": "https://mylamp.example.com/status"}]
    }
  },
  "actions": {
    "toggle": {
      "forms": [{"href": "https://mylamp.example.com/toggle"}]
    }
  }
}
```

---

## 12.2 Thing Discovery

**Priority**: Medium  
**Files**: `Tars.Connectors/WoT/ThingDiscovery.fs` (new)

| Task | Status | Description |
|------|--------|-------------|
| mDNS/DNS-SD discovery | 🔜 | Local network discovery |
| Parse directory links | 🔜 | From Thing Descriptions |
| Knowledge graph integration | 🔜 | Things as entities |
| CLI `tars wot discover` | 🔜 | User interface |

---

## 12.3 GoT + WoT Integration

**Priority**: Medium  
**Reference**: Combine Graph-of-Thoughts with WoT for intelligent IoT orchestration

| Task | Status | Description |
|------|--------|-------------|
| Inject Thing capabilities | 🔜 | Into GoT reasoning context |
| Plan multi-device operations | 🔜 | GoT device orchestration |
| WoT actions as primitives | 🔜 | GoT execution nodes |
| Device interaction history | 🔜 | Store in knowledge ledger |

### Example Use Case

```
User: "Turn on the living room lights when I arrive home"

GoT Reasoning:
1. [WoT] Discover devices → Find LivingRoomLight
2. [Knowledge] Check current state → Light is off
3. [WoT] Get thing properties → Supports 'on' action
4. [Plan] Create trigger rule → Presence → Light.on()
5. [Memory] Store automation → Knowledge ledger
```

---

## 12.4 Protocol Bindings

**Priority**: Low  
**Files**: `Tars.Connectors/WoT/Bindings/` (new)

| Task | Status | Description |
|------|--------|-------------|
| HTTP/HTTPS binding | 🔜 | Most common protocol |
| MQTT binding | 🔜 | Pub/sub devices |
| CoAP binding | 🔜 | Constrained devices |
| WebSocket binding | 🔜 | Real-time updates |

### Binding Architecture

```
┌───────────────────────────────────────┐
│           TARS Cortex                  │
└───────────────┬───────────────────────┘
                │
┌───────────────▼───────────────────────┐
│     Protocol Binding Abstraction       │
└───┬───────┬───────┬───────┬───────────┘
    │       │       │       │
┌───▼───┐┌──▼──┐┌───▼──┐┌───▼────┐
│ HTTP  ││MQTT ││ CoAP ││WebSocket│
└───────┘└─────┘└──────┘└────────┘
```

---

## 12.5 Security Considerations

| Concern | Mitigation |
|---------|------------|
| Device authentication | Use credential vault for tokens |
| Network isolation | Sandbox network access |
| Action authorization | Permission model per device |
| Audit trail | Log all device interactions |

---

## Success Criteria

- [ ] TARS can parse and understand Thing Descriptions
- [ ] `tars wot discover` finds local IoT devices
- [ ] GoT can reason about multi-device scenarios
- [ ] Device interactions are logged to knowledge ledger
- [ ] At least 2 protocol bindings implemented (HTTP, MQTT)

---

## Dependencies

- Phase 8: GoT (for reasoning integration)
- Phase 9: Knowledge Ledger (for device state storage)
- Phase 6: Security Core (for credential management)

---

*Phase 12 planned: Q1 2025*
