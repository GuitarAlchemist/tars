# TARS Secrets Management

TARS includes a secrets management system for securely storing and retrieving sensitive information such as API keys. This system ensures that sensitive data is not stored in the codebase or exposed in logs.

## Overview

The secrets management system provides:

1. **Secure Storage**: Sensitive information is stored securely using .NET User Secrets
2. **User Interaction**: Prompts for missing secrets when needed
3. **CLI Commands**: Commands for managing secrets
4. **Service Integration**: Seamless integration with services that require API keys

## Using the Secrets Command

TARS provides a `secrets` command for managing secrets:

```bash
tarscli secrets [subcommand] [options]
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `list` | List all secret keys |
| `set` | Set a secret value |
| `remove` | Remove a secret |
| `clear` | Clear all secrets |

### Examples

#### List Secrets

```bash
tarscli secrets list
```

This command lists all secret keys stored in the system.

#### Set a Secret

```bash
tarscli secrets set --key HuggingFace:ApiKey --value YOUR_API_KEY
```

If you omit the `--value` parameter, TARS will prompt you to enter the value securely:

```bash
tarscli secrets set --key HuggingFace:ApiKey
```

#### Remove a Secret

```bash
tarscli secrets remove --key HuggingFace:ApiKey
```

This command removes a specific secret.

#### Clear All Secrets

```bash
tarscli secrets clear
```

This command removes all secrets. Use with caution!

## Automatic Secret Prompting

TARS services that require API keys will automatically prompt for them if they are not found in the secrets store. For example, when using the Hugging Face service to download a model:

```bash
tarscli huggingface download --model microsoft/phi-2
```

If the Hugging Face API key is not found, TARS will prompt you to enter it:

```
No HuggingFace ApiKey found in secrets.
Would you like to set up a HuggingFace ApiKey? [Y/n]: y
Enter your HuggingFace ApiKey: ********
HuggingFace ApiKey saved to secrets.
Downloading model: microsoft/phi-2
```

## Secret Storage Location

Secrets are stored in the following locations:

- **Windows**: `%APPDATA%\Microsoft\UserSecrets\tars-cli-secrets\secrets.json`
- **Linux/macOS**: `~/.microsoft/usersecrets/tars-cli-secrets/secrets.json`

These locations are managed by the .NET User Secrets system and are not included in source control.

## Secret Naming Conventions

TARS uses a hierarchical naming convention for secrets:

```
ServiceName:KeyName
```

For example:
- `HuggingFace:ApiKey`
- `OpenAI:ApiKey`
- `Azure:SubscriptionId`

This convention helps organize secrets and avoid naming conflicts.

## Security Considerations

- **Never commit secrets to source control**
- **Never log secrets or display them in console output**
- **Use the `AskForSecret` method when prompting for sensitive information**
- **Clear secrets when they are no longer needed**

## Implementation Details

The secrets management system is implemented using the following components:

1. **SecretsService**: Manages the storage and retrieval of secrets
2. **UserInteractionService**: Handles user prompts and input
3. **CLI Commands**: Provides a user interface for managing secrets

Services that require API keys use the `UserInteractionService.AskForApiKeyAsync` method to retrieve keys, which:

1. Checks if the key exists in the secrets store
2. If not, checks if it exists in the configuration
3. If still not found, prompts the user to enter it
4. Stores the key in the secrets store for future use

## Extending the Secrets System

To add support for a new service that requires an API key:

1. Inject the `UserInteractionService` and `SecretsService` into your service
2. Use the `AskForApiKeyAsync` method to retrieve the API key
3. Follow the naming convention `ServiceName:KeyName` for your secret

Example:

```csharp
private async Task<string> GetApiKeyAsync(bool required = false)
{
    return await _userInteractionService.AskForApiKeyAsync("MyService", "ApiKey", required);
}
```
