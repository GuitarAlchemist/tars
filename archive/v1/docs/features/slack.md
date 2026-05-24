# TARS Slack Integration

TARS includes a Slack integration feature that allows it to post announcements, feature updates, milestones, and auto-improvement updates to a Slack channel. This enables teams to stay informed about TARS's progress and activities.

## Overview

The Slack integration provides:

1. **Announcements**: Post important announcements to a Slack channel
2. **Feature Updates**: Share information about new features
3. **Milestones**: Celebrate achievements and milestones
4. **Auto-Improvement Updates**: Get updates on TARS's autonomous self-improvement progress

## Setup

To use the Slack integration, you need to set up a webhook URL:

1. **Create a Slack App**:
   - Go to [https://api.slack.com/apps](https://api.slack.com/apps)
   - Click "Create New App" and select "From scratch"
   - Enter a name for your app (e.g., "TARS") and select your workspace
   - Click "Create App"

2. **Enable Incoming Webhooks**:
   - In your app settings, click on "Incoming Webhooks"
   - Toggle "Activate Incoming Webhooks" to On
   - Click "Add New Webhook to Workspace"
   - Select the channel where you want TARS to post messages
   - Click "Allow"

3. **Copy the Webhook URL**:
   - After allowing the app, you'll see a webhook URL
   - Copy this URL

4. **Configure TARS**:
   - Use the `slack set-webhook` command to set the webhook URL:
     ```bash
     tarscli slack set-webhook --url https://hooks.slack.com/services/XXX/YYY/ZZZ
     ```

## Using the Slack Command

TARS provides a `slack` command for managing the Slack integration:

```bash
tarscli slack [subcommand] [options]
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `set-webhook` | Set the Slack webhook URL |
| `test` | Test the Slack integration |
| `announce` | Post an announcement to Slack |
| `feature` | Post a feature update to Slack |
| `milestone` | Post a milestone to Slack |

### Examples

#### Set Webhook URL

```bash
tarscli slack set-webhook --url https://hooks.slack.com/services/XXX/YYY/ZZZ
```

This command sets the webhook URL for the Slack integration.

#### Test Integration

```bash
tarscli slack test --message "Hello from TARS" --channel #tars
```

This command sends a test message to the specified channel (or the default channel if not specified).

#### Post an Announcement

```bash
tarscli slack announce --title "New Release" --message "TARS v1.0 is now available!"
```

This command posts an announcement to the default channel.

#### Post a Feature Update

```bash
tarscli slack feature --name "GPU Acceleration" --description "TARS now supports GPU acceleration for faster processing."
```

This command posts a feature update to the default channel.

#### Post a Milestone

```bash
tarscli slack milestone --name "1000 Users" --description "TARS has reached 1000 active users!"
```

This command posts a milestone to the default channel.

## Automatic Updates

TARS automatically posts updates to Slack for certain events:

### Auto-Improvement Updates

When TARS is running in autonomous self-improvement mode, it will:

1. Post an update every 5 improvements
2. Post a summary when the auto-improvement session completes

These updates include:
- Number of improvements made
- Number of files processed
- Recently improved files
- Time spent on improvements

## Configuration

The Slack integration can be configured using:

1. **User Secrets**:
   - The webhook URL is stored securely using .NET User Secrets
   - Use the `slack set-webhook` command to set the webhook URL

2. **Configuration File**:
   - You can set the default channel in the configuration file:
     ```json
     {
       "Slack": {
         "DefaultChannel": "#tars"
       }
     }
     ```

## Security Considerations

- The webhook URL is stored securely using .NET User Secrets
- The webhook URL is not included in source control
- The webhook URL is not displayed in logs or console output

## Implementation Details

The Slack integration is implemented using:

1. **SlackIntegrationService**: Handles posting messages to Slack
2. **SecretsService**: Manages the secure storage of the webhook URL
3. **CLI Commands**: Provides a user interface for managing the Slack integration

## Extending the Slack Integration

To add support for posting new types of updates to Slack:

1. Add a new method to the `SlackIntegrationService` class
2. Add a new command to the `slack` command group
3. Update the relevant service to call the new method

Example:

```csharp
// Add a new method to SlackIntegrationService
public async Task<bool> PostCustomUpdateAsync(string title, string message, string? channel = null)
{
    // Implementation
}

// Call the method from your service
await _slackService.PostCustomUpdateAsync("Custom Update", "This is a custom update.");
```
