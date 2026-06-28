# Cloud-Agent Skill Library

This directory contains explicit skills for cloud agents (like Jules or Codex) when working AFK (Away From Keyboard) from GitHub issues.

Cloud agents cannot access local TARS skills or local MCP state. However, they can reliably read repo files and issue bodies. This repo-readable skill library provides standardized behavior patterns that cloud agents must follow when executing delegated issues.

## Available Skills

- **Anti-Ball-of-Mud** (`anti-ball-of-mud.skill.md`): Used to detect architecture entropy, propose narrow seams, and prevent agent-driven architectural degradation.
- **Docs-Only Contract** (`docs-only-contract.skill.md`): Used for documentation-only updates, enforcing that no runtime behavior is changed.
- **Evidence Bundle** (`evidence-bundle.skill.md`): Used for grounding claims and design decisions in explicitly referenced source material.
- **Governed Delegation** (`governed-delegation.skill.md`): Used for navigating the Demerzel governance constraints and respecting AFK halt markers.

## Usage

Cloud agents should parse the `issue_meta` block in the GitHub issue body and use these definitions to guide their approach, ensuring that they respect project constraints, execute required checks, and provide the expected evidence in the pull request body.
