# TARS Autonomous Self-Improvement

TARS includes an autonomous self-improvement system that allows it to analyze and improve its own codebase without human intervention. This feature enables TARS to evolve and enhance its capabilities over time.

## Overview

The autonomous self-improvement system:

1. **Runs in the Background**: Operates as a background process, allowing you to continue using your computer
2. **Time-Limited**: Runs for a specified duration (default: 60 minutes)
3. **Safe Stopping**: Gracefully stops when the time limit is reached or when manually stopped
4. **State Persistence**: Maintains state between runs, allowing it to resume where it left off
5. **Prioritized Processing**: Intelligently selects files to process based on their potential value

## Using the Auto-Improve Command

The `auto-improve` command is used to start, monitor, and control the autonomous self-improvement process:

```bash
tarscli auto-improve [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--time-limit` | `60` | Time limit in minutes |
| `--model` | `llama3` | Model to use for improvements |
| `--status` | - | Show status of autonomous improvement |
| `--stop` | - | Stop autonomous improvement |

### Examples

Start autonomous improvement with default settings (60 minutes, llama3 model):
```bash
tarscli auto-improve
```

Start autonomous improvement with custom settings:
```bash
tarscli auto-improve --time-limit 120 --model codellama
```

Check the status of autonomous improvement:
```bash
tarscli auto-improve --status
```

Stop autonomous improvement:
```bash
tarscli auto-improve --stop
```

## How It Works

### File Selection

The autonomous self-improvement system processes files from two main sources:

1. **Documentation Files**: `C:\Users\spare\source\repos\tars\TarsCli\doc`
2. **Chat Transcripts**: `C:\Users\spare\source\repos\tars\docs\Explorations\v1\Chats`

Files are prioritized based on their potential value for improvement:
- Documentation files are processed first, as they're more structured
- Chat transcripts are prioritized by size, with larger files processed first (as they might contain more useful information)

### Improvement Process

For each file, the system:

1. **Analyzes** the file to identify potential improvements
2. **Proposes** specific improvements
3. **Applies** the improvements if they meet quality criteria
4. **Records** the changes for future reference

### State Management

The system maintains its state in a JSON file (`auto_improvement_state.json`) in the project root directory. This file contains:

- **Processed Files**: Files that have already been processed
- **Pending Files**: Files that are waiting to be processed
- **Current File**: The file currently being processed
- **Last Improved File**: The last file that was successfully improved
- **Total Improvements**: The total number of improvements made

This state persistence allows the system to resume where it left off if it's stopped and restarted.

### Safe Stopping

When the time limit is reached or the stop command is issued, the system:

1. Completes processing of the current file
2. Saves its state
3. Gracefully exits

This ensures that no files are left in an inconsistent state.

## Monitoring Progress

You can monitor the progress of autonomous improvement using the `--status` option:

```bash
tarscli auto-improve --status
```

This will show:
- Whether the process is running
- Start time and time limit
- Elapsed and remaining time
- Number of files processed and remaining
- Current file being processed
- Last improved file
- Total number of improvements made

## Best Practices

1. **Start with a Short Time Limit**: Begin with a short time limit (e.g., 30 minutes) to see how the system performs
2. **Use a Powerful Model**: Better models (e.g., codellama) generally produce better improvements
3. **Run During Off Hours**: The process can be resource-intensive, so consider running it when you're not actively using your computer
4. **Check Status Periodically**: Monitor progress to ensure the system is working as expected
5. **Review Improvements**: After the process completes, review the improvements to ensure they meet your standards

## Technical Details

The autonomous self-improvement system is implemented in the `AutoImprovementService` class, which:

1. Manages the improvement process
2. Maintains state between runs
3. Handles graceful stopping
4. Provides status information

The service uses the `SelfImprovementService` to analyze and improve files, and the `OllamaService` to generate improvements using the specified model.

## Future Enhancements

Future versions of the autonomous self-improvement system may include:

1. **More Sophisticated Prioritization**: Using machine learning to prioritize files based on their potential value
2. **Improvement Quality Metrics**: Measuring the quality of improvements to guide future improvements
3. **Self-Optimization**: Allowing the system to optimize its own improvement strategies
4. **Distributed Processing**: Running the improvement process across multiple machines
5. **Web Interface**: Providing a web interface for monitoring and controlling the improvement process
