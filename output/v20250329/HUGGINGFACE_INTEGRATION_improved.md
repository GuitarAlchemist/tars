# Hugging Face Integration for TARS: A Guide to Browsing, Downloading, and Installing LLMs

TARS now integrates with Hugging Face, allowing you to effortlessly explore, download, and install Large Language Models (LLMs) from the Hugging Face Hub.

## Features

### Search and Discover

* Browse models based on keywords and tasks using `tarscli huggingface search`
* Find the highest-rated coding models with `tarscli huggingface best`

### Model Management

* View detailed information about any model using `tarscli huggingface details`
* Download a model from Hugging Face to your local machine with `tarscli huggingface download`
* Install a model to Ollama, including name and version management, using `tarscli huggingface install`

### Model Usage

* List all installed models with `tarscli huggingface list`
* Use the installed model with TARS commands that accept a model parameter (e.g., `tarscli self-analyze`)

## Prerequisites

* Git must be installed and available in your PATH
* Ollama must be installed and running
* Optional: A Hugging Face API key for accessing private models

## Configuration

To use the Hugging Face integration, add your Hugging Face API key to the `appsettings.json` file: