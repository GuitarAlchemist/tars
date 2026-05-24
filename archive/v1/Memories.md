# Task Management
- Break down tasks into very granular items in TODOs files.
- Use all TODOs .md file to track progress
- If a task is too big instead of answering "I'm sorry. I tried to call a tool, but provided too large of an input. How would you like to proceed?", please update existing TODOs or create a new one

# Project Structure
- Use relative paths instead of absolute paths in metascripts when referencing project directories.
- Clean up and consolidate messy project structure.
- Metascripts should not reference C# code directly and should avoid hardcoding values.
- Use full paths instead of relative paths in report links.
- F# files should be moved to a dedicated .fsproj file rather than being mixed with C# code.
- User questions mixing F# files in a C# project, suggesting potential concerns about language mixing in the codebase.
- User is considering migrating C# code from the engine into F#.
- KnowledgeRepository component may be unused and could be considered for removal.

# Metrics and Estimation
- Verify the accuracy of percentage completion metrics rather than blindly trusting them.

# TARS
- Focus on making TARS auto-improvement capability work first before expanding to other project directions.
- The TARS project lacks clear direction for auto-improvement that produces tangible results.
- Use metascripts to auto-improve TARS and implement concepts from exploration documents in the docs/Explorations/v1/Chats directory.
- Real F# compilation capability may be needed for TARS metascripts.
- Focus on metascript integration as a priority for the Tree-of-Thought implementation.
- User believes Tree-of-Thought implementation was already completed in a previous session.
