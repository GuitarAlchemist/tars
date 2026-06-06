# Hugging Face Integration for TARS

TARS now includes integration with Hugging Face, allowing you to browse, download, and install the best coding LLMs directly from the Hugging Face Hub.

## Features

- **Search for Models**: Search the Hugging Face Hub for models based on keywords and tasks
- **Find Best Coding Models**: Automatically find the highest-rated coding models
- **View Model Details**: Get detailed information about any model
- **Download Models**: Download models directly from Hugging Face
- **Install to Ollama**: Convert and install Hugging Face models to Ollama for local use

## Prerequisites

- Git must be installed and available in your PATH
- Ollama must be installed and running
- (Optional) A Hugging Face API key for accessing private models

## Configuration

Add your Hugging Face API key to the `appsettings.json` file:

```json
"HuggingFace": {
  "ApiKey": "your-api-key-here",
  "DefaultTask": "text-generation"
}
```

## Commands

### Search for Models

Search for models on Hugging Face based on keywords and tasks:

```
tarscli huggingface search --query "code generation" --task text-generation --limit 5
```

Parameters:
- `--query`: Search query (required)
- `--task`: Task type (default: "text-generation")
- `--limit`: Maximum number of results to return (default: 10)

### Find Best Coding Models

Get the best coding models from Hugging Face, sorted by downloads and likes:

```
tarscli huggingface best --limit 3
```

Parameters:
- `--limit`: Maximum number of results to return (default: 10)

### View Model Details

Get detailed information about a specific model:

```
tarscli huggingface details --model microsoft/phi-2
```

Parameters:
- `--model`: Model ID (required)

### Download a Model

Download a model from Hugging Face to your local machine:

```
tarscli huggingface download --model microsoft/phi-2
```

Parameters:
- `--model`: Model ID (required)

### Install a Model to Ollama

Download a model from Hugging Face and install it to Ollama:

```
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

Parameters:
- `--model`: Model ID (required)
- `--name`: Name to use in Ollama (optional)

### List Installed Models

List all models downloaded from Hugging Face:

```
tarscli huggingface list
```

## Examples

### Finding and Installing the Best Coding Model

1. Find the best coding models:
   ```
   tarscli huggingface best --limit 3
   ```

2. Get details about a specific model:
   ```
   tarscli huggingface details --model microsoft/phi-2
   ```

3. Install the model to Ollama:
   ```
   tarscli huggingface install --model microsoft/phi-2 --name phi2
   ```

4. Use the model with TARS:
   ```
   tarscli self-analyze --file test_code.cs --model phi2
   ```

## How It Works

1. **Searching**: The service uses the Hugging Face API to search for models based on your criteria
2. **Downloading**: Models are downloaded using Git to clone the model repository
3. **Installation**: A Modelfile is created for Ollama, and the model is converted to Ollama format
4. **Usage**: The installed model can be used with any TARS command that accepts a model parameter

## Recommended Models for Coding

- **microsoft/phi-2**: A compact yet powerful model for code generation
- **codellama/CodeLlama-7b-hf**: Specialized for code completion and generation
- **bigcode/starcoder**: Trained on a large corpus of code from GitHub
- **Salesforce/codegen-16B-mono**: High-quality code generation model
- **replit/replit-code-v1-3b**: Optimized for code completion

## Troubleshooting

- **API Key Issues**: Ensure your API key is correctly set in `appsettings.json`
- **Download Failures**: Check your internet connection and Git installation
- **Installation Failures**: Ensure Ollama is running and has sufficient disk space
- **Model Compatibility**: Not all models are compatible with Ollama; check the model architecture

## Future Enhancements

- Support for fine-tuning models
- Integration with TARS self-improvement system
- Automatic model benchmarking
- Model version management
- Support for other model formats and frameworks
