# CONFIG Block

The CONFIG block defines configuration settings for TARS programs.

## Syntax

```
CONFIG [name] {
    property1: value1;
    property2: value2;
    ...
}
```

The name is optional and can be used to reference the configuration elsewhere in the program.

## Properties

### Common Properties

| Property | Type | Description |
|----------|------|-------------|
| `version` | string | Version of the program |
| `author` | string | Author of the program |
| `description` | string | Description of the program |
| `license` | string | License of the program |
| `created` | string | Creation date of the program |
| `updated` | string | Last update date of the program |

### Additional Properties

| Property | Type | Description |
|----------|------|-------------|
| `tags` | array | Tags for categorizing the program |
| `repository` | string | URL of the program's repository |
| `homepage` | string | URL of the program's homepage |
| `documentation` | string | URL of the program's documentation |
| `bugs` | string | URL for reporting bugs |

## Examples

### Basic Configuration

```
CONFIG {
    version: "1.0";
    author: "John Doe";
    description: "Example TARS program";
}
```

### Detailed Configuration

```
CONFIG {
    version: "1.0.0";
    author: "John Doe";
    description: "A program for analyzing code quality";
    license: "MIT";
    created: "2025-03-29";
    updated: "2025-03-29";
    tags: ["code-analysis", "quality", "refactoring"];
    repository: "https://github.com/johndoe/tars-example";
    homepage: "https://johndoe.github.io/tars-example";
    documentation: "https://johndoe.github.io/tars-example/docs";
    bugs: "https://github.com/johndoe/tars-example/issues";
}
```

### Named Configuration

```
CONFIG MyConfig {
    version: "1.0";
    author: "John Doe";
    description: "Example TARS program";
}
```

## Usage

The CONFIG block is typically placed at the beginning of a TARS program to provide metadata about the program. It can be referenced elsewhere in the program using its name:

```
CONFIG MyConfig {
    version: "1.0";
    author: "John Doe";
}

ACTION {
    print("Program version: " + MyConfig.version);
    print("Author: " + MyConfig.author);
}
```

## Best Practices

- Always include a CONFIG block in your TARS programs
- Provide at least a version, author, and description
- Use semantic versioning for the version property
- Keep the description concise but informative
- Include a license to clarify how the program can be used
- Use tags to categorize the program
- Update the updated property when making changes to the program
