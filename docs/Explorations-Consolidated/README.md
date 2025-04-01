# TARS Explorations

This directory contains consolidated explorations organized by topic rather than version.

## Overview

TARS explorations are in-depth analyses and reflections on various topics related to AI, programming languages, and the TARS framework. They are generated using TARS's deep thinking capabilities and represent evolving thoughts on complex subjects.

## Structure

Each subdirectory represents a topic and contains multiple versions of explorations on that topic. Files are named according to their version (e.g., `v1.md`, `v2.md`).

## Metadata

Metadata for all explorations is stored in `metadata.json`. This file contains information about:

- Topics
- Versions
- Creation dates
- Modification dates
- Original file paths

## Benefits of Consolidated Structure

This consolidated structure offers several advantages:

1. **Topic-Centric Organization**: Explorations are grouped by topic, making it easier to find related content.
2. **Version History**: All versions of an exploration are kept together, showing the evolution of thinking.
3. **Reduced GitHub Friction**: Fewer directories and a more logical structure reduce complexity in the repository.
4. **Improved Discoverability**: The README provides a clear index of available topics.

## Generating New Explorations

To generate new explorations, use the TARS CLI:

```bash
tarscli deep-thinking --topic "Your Topic" --model llama3
```

New explorations will be automatically added to the appropriate topic directory, with version numbers incremented as needed.

## Viewing Explorations

Explorations are stored as markdown files and can be viewed in any markdown viewer or editor. The latest version of each topic is linked in the README for easy access.
