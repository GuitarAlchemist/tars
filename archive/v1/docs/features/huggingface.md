# Hugging Face Integration

TARS includes comprehensive integration with Hugging Face, allowing you to browse, download, and install the best coding language models directly from the Hugging Face Hub. This document provides a detailed overview of the Hugging Face integration, its capabilities, and how to use it effectively.

## Overview

The Hugging Face integration is designed to:

1. **Search for models** on the Hugging Face Hub
2. **Find the best coding models** based on downloads and likes
3. **Download models** to your local machine
4. **Install models to Ollama** for use with TARS

![Hugging Face Integration](../images/huggingface_integration.svg)

## Key Features

### Model Search

TARS can search for models on Hugging Face based on keywords and tasks:

```bash
tarscli huggingface search --query "code generation" --task text-generation --limit 5
```

This will return a list of models matching your search criteria, including information about the author, downloads, likes, and tags.

### Best Coding Models

TARS can find the best coding models on Hugging Face, sorted by downloads and likes:

```bash
tarscli huggingface best --limit 3
```

This will return a list of the top coding models, helping you identify the most popular and effective models for code-related tasks.

### Model Details

TARS can provide detailed information about specific models:

```bash
tarscli huggingface details --model microsoft/phi-2
```

This will display comprehensive information about the model, including:
- Author
- Downloads and likes
- Last modified date
- License
- Tags and languages
- Files and sizes

### Model Download

TARS can download models from Hugging Face to your local machine:

```bash
tarscli huggingface download --model microsoft/phi-2
```

This will clone the model repository to your local machine, making it available for offline use.

### Model Installation

TARS can install models from Hugging Face to Ollama for use with TARS:

```bash
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

This will:
1. Download the model from Hugging Face
2. Convert it to Ollama format
3. Make it available for use with TARS commands

### Installed Models

TARS can list the models you've installed from Hugging Face:

```bash
tarscli huggingface list
```

This will display a list of the models you've downloaded from Hugging Face.

## Architecture

The Hugging Face integration is implemented as a service with several components:

### HuggingFaceService

The `HuggingFaceService` is the main entry point for Hugging Face operations. It:

1. Communicates with the Hugging Face API
2. Manages model downloads and installations
3. Integrates with Ollama for model conversion

### Model Search

The model search functionality uses the Hugging Face API to find models based on:
- Keywords
- Tasks (e.g., text-generation, code-generation)
- Popularity metrics (downloads, likes)

### Model Download

The model download functionality uses Git to clone model repositories from Hugging Face, ensuring you have the complete model with all its files.

### Model Conversion

The model conversion functionality creates an Ollama-compatible Modelfile and uses the Ollama CLI to create a local model that can be used with TARS.

## Using the Hugging Face Integration

### Searching for Models

To search for models on Hugging Face:

```bash
tarscli huggingface search --query "code generation" --task text-generation --limit 5
```

Example output:

```
Hugging Face - Search Models
Query: code generation
Task: text-generation
Limit: 5

Found 5 models:

microsoft/phi-2
Author: microsoft
Downloads: 1,234,567
Likes: 4,321
Tags: text-generation, code-generation

codellama/CodeLlama-7b-hf
Author: codellama
Downloads: 987,654
Likes: 3,210
Tags: text-generation, code-generation

bigcode/starcoder
Author: bigcode
Downloads: 876,543
Likes: 2,109
Tags: text-generation, code-generation

Salesforce/codegen-16B-mono
Author: Salesforce
Downloads: 765,432
Likes: 1,098
Tags: text-generation, code-generation

replit/replit-code-v1-3b
Author: replit
Downloads: 654,321
Likes: 987
Tags: text-generation, code-generation
```

### Finding the Best Coding Models

To find the best coding models on Hugging Face:

```bash
tarscli huggingface best --limit 3
```

Example output:

```
Hugging Face - Best Coding Models
Limit: 3

Found 3 best coding models:

microsoft/phi-2
Author: microsoft
Downloads: 1,234,567
Likes: 4,321
Tags: text-generation, code-generation

codellama/CodeLlama-7b-hf
Author: codellama
Downloads: 987,654
Likes: 3,210
Tags: text-generation, code-generation

bigcode/starcoder
Author: bigcode
Downloads: 876,543
Likes: 2,109
Tags: text-generation, code-generation
```

### Getting Model Details

To get detailed information about a specific model:

```bash
tarscli huggingface details --model microsoft/phi-2
```

Example output:

```
Hugging Face - Model Details
Model ID: microsoft/phi-2

microsoft/phi-2
Author: microsoft
Downloads: 1,234,567
Likes: 4,321
Last Modified: 2023-12-15T12:34:56
License: MIT
Tags: text-generation, code-generation
Languages: en

Files:
- config.json (12.34 KB)
- model.safetensors (2.50 GB)
- tokenizer.json (1.23 MB)
- tokenizer_config.json (4.56 KB)
```

### Downloading a Model

To download a model from Hugging Face:

```bash
tarscli huggingface download --model microsoft/phi-2
```

Example output:

```
Hugging Face - Download Model
Model ID: microsoft/phi-2

Downloading model: microsoft/phi-2
Cloning into '/path/to/tars/models/huggingface/microsoft_phi-2'...
remote: Enumerating objects: 123, done.
remote: Counting objects: 100% (123/123), done.
remote: Compressing objects: 100% (100/100), done.
remote: Total 123 (delta 23), reused 0 (delta 0)
Receiving objects: 100% (123/123), 2.50 GB | 10.5 MB/s, done.
Resolving deltas: 100% (23/23), done.

Successfully downloaded model: microsoft/phi-2
```

### Installing a Model to Ollama

To install a model from Hugging Face to Ollama:

```bash
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

Example output:

```
Hugging Face - Install Model
Model ID: microsoft/phi-2
Ollama Name: phi2

Downloading model: microsoft/phi-2
Cloning into '/path/to/tars/models/huggingface/microsoft_phi-2'...
remote: Enumerating objects: 123, done.
remote: Counting objects: 100% (123/123), done.
remote: Compressing objects: 100% (100/100), done.
remote: Total 123 (delta 23), reused 0 (delta 0)
Receiving objects: 100% (123/123), 2.50 GB | 10.5 MB/s, done.
Resolving deltas: 100% (23/23), done.

Converting model to Ollama format: microsoft/phi-2
Creating phi2: 100% [==================================================] 2.50 GB

Successfully installed model: microsoft/phi-2
```

### Listing Installed Models

To list the models you've installed from Hugging Face:

```bash
tarscli huggingface list
```

Example output:

```
Hugging Face - Installed Models

Found 3 installed models:

microsoft_phi-2
codellama_CodeLlama-7b-hf
bigcode_starcoder
```

## Configuration

The Hugging Face integration can be configured through the `appsettings.json` file:

```json
{
  "HuggingFace": {
    "ApiKey": "your-api-key",
    "DefaultTask": "text-generation"
  }
}
```

- **ApiKey**: Your Hugging Face API key (optional, required for private models)
- **DefaultTask**: The default task to use when searching for models

## Best Practices

### Model Selection

When selecting models for use with TARS:

1. **Consider Size**: Larger models generally provide better results but require more resources
2. **Check License**: Ensure the model's license is compatible with your use case
3. **Look at Metrics**: Downloads and likes can indicate a model's quality and popularity
4. **Read Documentation**: Check the model card for information about capabilities and limitations

### Resource Management

When working with models:

1. **Check Disk Space**: Models can be large, so ensure you have sufficient disk space
2. **Monitor Memory Usage**: Running models requires significant memory
3. **Consider GPU Acceleration**: GPU acceleration can significantly improve performance
4. **Clean Up Unused Models**: Remove models you're not using to free up space

### Security Considerations

When using the Hugging Face integration:

1. **API Keys**: Keep your API keys secure
2. **Model Verification**: Verify models from untrusted sources
3. **Content Filtering**: Be aware that models may generate inappropriate content

## Recommended Models for Coding

Based on performance and popularity, here are some recommended models for coding tasks:

### Small Models (< 3B parameters)

- **microsoft/phi-2**: A compact yet powerful model for code generation
- **replit/replit-code-v1-3b**: Optimized for code completion

### Medium Models (3B - 10B parameters)

- **codellama/CodeLlama-7b-hf**: Specialized for code completion and generation
- **bigcode/starcoder**: Trained on a large corpus of code from GitHub

### Large Models (> 10B parameters)

- **Salesforce/codegen-16B-mono**: High-quality code generation model
- **codellama/CodeLlama-34b-hf**: Large model with advanced code capabilities

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter issues with your API key:

1. Ensure your API key is correctly set in `appsettings.json`
2. Verify that your API key has the necessary permissions
3. Check if you've reached API rate limits

#### Download Failures

If model downloads fail:

1. Check your internet connection
2. Ensure Git is installed and available in your PATH
3. Verify that you have sufficient disk space
4. Try downloading a smaller model first

#### Installation Failures

If model installation to Ollama fails:

1. Ensure Ollama is running
2. Check if Ollama has sufficient disk space
3. Verify that the model architecture is compatible with Ollama
4. Try installing a known-compatible model first

## Future Enhancements

The Hugging Face integration is continuously evolving. Planned enhancements include:

1. **Model Fine-tuning**: Support for fine-tuning models on your own data
2. **Model Benchmarking**: Tools for evaluating model performance
3. **Model Version Management**: Support for managing multiple versions of models
4. **Model Recommendation System**: Intelligent recommendations based on your needs
5. **Integration with Self-Improvement**: Using Hugging Face models for self-improvement
