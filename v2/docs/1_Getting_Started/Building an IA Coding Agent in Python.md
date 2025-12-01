Source: YouTube video: <https://www.youtube.com/watch?v=Q3Gb7Rjre3U>

This document provides a detailed, beginner-friendly transcript detailing the steps involved in building an AI coding agent, based entirely on the provided source material.

## AI Coding Agent: Step-by-Step Build Guide

This tutorial details how to build an AI coding agent from scratch in Python, relying on the Anthropic Claude 4.5 Sonnet model. The finished agent lives inside a single `main.py` file, which is just over 200 lines long.

### Overview and Setup

The agent is built sequentially, starting simply and adding complexity with each step, using a series of seven scripts (01 to 07).

#### 1. Prerequisites

1. **Repository Setup:** Clone the GitHub repository (forked from Francis Pson's work). The link is available in the description.
2. **IDE Setup:** Open the repository in your favorite Integrated Development Environment (IDE), such as Cursor.
3. **Package Manager:** Install and use **`uv`**, an extremely fast Python package and project manager that can essentially replace `pip`.
4. **API Key:** Obtain an **API key from Anthropic**. This key is necessary to use the new Claude 4.5 Sonnet model. You can find the API keys at `console.entropic.com`.

#### 2. Running the Basic Script (01 basic script)

The goal of the first script is simply to ensure the setup is working and the Anthropic API key is plugged in.

1. **Set the API Key:** Open your terminal and run the following command, replacing `YOUR_KEY` with your actual Anthropic API key:

    ```bash
    export ENTROPIC_API_KEY=YOUR_KEY
    ```

    This key will be active within this terminal session.
2. **Navigate:** Change the directory to the `runbook` folder, where all the tutorial files reside.
3. **Execute:** Run the first script using `uv`:

    ```bash
    uv run 01
    ```

4. **Expected Output:** You should see the message: **"AI agent starting now"**.

#### 3. Defining the Agent Class and Tools Structure (02 agent class)

This step defines the tools structure and sets up the foundational `AIAgent` class.

1. **Dependencies:** The script utilizes `uv` features to manage external dependencies like **`entropic`** and **`pyantic`** without needing a virtual environment. Default Python libraries like `os`, `system`, and `typing` are also imported.
2. **Tool Definition:** Tools are standardized using **Pyantic**, a necessary library when building AI agents with Python. A tool definition must include a **name**, a **description**, and an **input schema**.
3. **Agent Initialization:** The `AIAgent` class is initialized with three key attributes:
    * **`client`**: The Anthropic client created using the Python SDK.
    * **`messages`**: A list of dictionaries that stores the conversation history (specifying `role: user` or `assistant`).
    * **`tools`**: A list containing the tool definitions.
4. **Execute:** Run the second script:

    ```bash
    uv run 02
    ```

5. **Expected Output:** You should see: **"Agent initialized"**.

#### 4. Specifying the Tools (03 define tools)

The goal here is to specify the tools available to the agent so it understands what actions it can take, and what parameters those actions require. This step focuses only on the specification, not the actual Python functions.

The minimal viable product (MVP) coding agent requires three capabilities:

| Tool Name | Description | Input Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `read_file` | Reads the content of a specified file. | Requires a `file_path`. | Necessary for the agent to understand the environment. |
| `list_files` | Lists all files and directories in a specified path. | Requires a `path`. | Necessary for the agent to browse. |
| `edit_file` | Edits a file by replacing old text with new text (and creates the file if it doesn't exist). | Requires `path`, `old_text`, and `new_text`. | Allows the agent to modify code/files. |

1. **Modification:** The agent's initialization now calls an internal function, `setup_tools`, to load these definitions into the `tools` list.
2. **Execute:** Run the third script:

    ```bash
    uv run 03
    ```

3. **Expected Output:** You should see: **"Agent initialized with three tools"**.

#### 5. Implementing the Tool Logic (04 implement tools)

This crucial step adds the actual Python logic that allows the tools to execute and interact with the operating system.

1. **Tool Functions:** The logic for `read_file`, `list_files`, and `edit_file` is implemented using internal Python libraries like `os`.
    * **`read_file`:** Uses `with open` to read file contents based on a path.
    * **`list_files`:** Uses `os.listdir()` to loop over and list files in a directory, appending whether the item is a file or a directory.
    * **`edit_file`:** Reads the file content, uses Python's internal **`content.replace`** method to swap the old text with the new text, and then writes the content back to the file.
2. **Robustness:** The implementation in the agent class includes **`try` and `except` logic** for error handling, making the agent more robust and allowing it to course-correct if execution fails.
3. **Execution Logic:** Implement the `execute_tool` function. This function loops through the output the agent suggests, checks which tool was decided upon (e.g., `read_file`), and executes the corresponding Python function.
4. **Test Execution:** The main script tests the functionality by directly calling `agent.list_files()` (note: this test does not use the AI or the `execute_tool` function; it just verifies the Python function works).
5. **Execute:** Run the fourth script:

    ```bash
    uv run 04
    ```

6. **Expected Output:** The terminal lists all the files within the current directory (`runbook`).

#### 6. Adding the Chat Method (05 adding chat method)

This step introduces the core agent loop, enabling interaction and tool use via the Anthropic SDK.

1. **The `chat` Method:** A new `chat` method is created which takes user input and manages the conversation loop.
2. **Request Preparation:** The user input is taken and appended to the `messages` list (Role: `user`). The agent's tool specifications are converted into the format required by the Anthropic Python SDK.
3. **The Loop (Core Agent Logic):** The system enters a loop where it sends the messages and tools to the Claude model.
4. **Parsing the Response:** The agent parses the model's answer to determine the next action:
    * **Text Response:** If the agent decides to reply with text (no tools used), the response is printed.
    * **Tool Use:** If the agent decides it needs to execute a tool (e.g., `read a file`), the tool specification (following Anthropic's standard syntax) is appended to the messages list.
5. **Execution and Re-run:** If tool use is detected, the `execute_tool` function is called. The results of the executed tool are added back to the `messages` list. This updated context is then sent back to the Large Language Model (LLM) for a final answer. This mechanism is the core principle of function calling/tool use.
6. **Execute:** Run the fifth script:

    ```bash
    uv run 05
    ```

7. **Test:** Running a query like "what files are in the current directory" will demonstrate the agent loop working. It makes two LLM calls: one to decide it needs the `list_files` tool, and a second to get the final answer after the tool results are known.

#### 7. Creating an Interactive CLI (06 interactive cli)

To make the agent usable in the terminal, an interactive Command Line Interface (CLI) is built.

1. **CLI Setup:** The `argparse` library is imported.
2. **Main Loop:** The `main` function is modified to run a **`while true` loop** that continuously captures user input.
3. **Exit:** The loop checks for `exit` or `quit` input to break the loop and end the application.
4. **Execute:** Run the sixth script:

    ```bash
    uv run 06
    ```

5. **Interaction:** The assistant is now interactive, allowing continuous questions (e.g., "What can you do?") and tool execution (e.g., "list all files in this tier").

#### 8. Adding Personality and Functionality (07 and main.py)

The final steps involve adding a system prompt to refine the agent's behavior and enabling logging.

1. **System Prompt:** A **`system_prompt`** is added to the agent setup. This is where instructions on how the agent should operate are defined (e.g., "You operate within a terminal environment and you output only with plain markdown"). This prompt can be tweaked to debug behavior (e.g., "don't use any asterisk characters").
2. **Logging:** Logging is enabled to keep track of the agent's actions.
3. **Final Execution:** Run the final script or the comprehensive `main.py` file:

    ```bash
    uv run 07
    # or
    uv run main
    ```

4. **Demonstration:** The fully functioning agent can now perform complex tasks, such as being asked to create a `test.py` file, add a function to it, expand its functionality, and finally empty the file entirely.

***

## Translation to a .NET/F\# Background

The detailed steps provided for building this AI coding agent rely exclusively on **Python**. The implementation uses specific Python libraries and tools:

* **Package Management:** `uv` and `pip`.
* **LLM Interaction:** The Anthropic Python SDK.
* **Data Validation/Schema:** `pyantic` for defining tool specifications.
* **System Interaction:** Python's built-in `os` library.

**The provided sources do not contain any information regarding how this AI agent architecture would translate to a .NET or F# background.**

However, the core concepts detailed—which are central to almost all agentic AI applications—remain the same regardless of the programming language used:

1. **Tool Specification:** Defining clear schemas (name, description, input parameters) for functions the LLM can access (handled by Pyantic in Python).
2. **State Management:** Maintaining conversation history in a structured list of messages.
3. **The Agent Loop:** Sending the tool specification and message history to the LLM.
4. **Decision Parsing:** Interpreting the LLM's response to determine if it requires a tool, or if it can provide a final text response.
5. **Tool Execution:** Implementing the actual system functions (reading/editing files).
6. **Context Injection:** Feeding the results of the executed tool back into the conversation history before running the LLM again for a final answer.

To implement this architecture in .NET or F#, a developer would need to replace the Python-specific components (like Pyantic for schema validation and the Anthropic Python SDK) with equivalent libraries and SDKs available in the .NET ecosystem.
