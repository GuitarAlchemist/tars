# What is vLLM

**User:** Stephane Pareilleux (spareilleux@gmail.com)  
**Created:** 11/26/2025 0:56:37  
**Updated:** 11/26/2025 0:59:47  
**Exported:** 11/26/2025 14:56:33  
**Link:** [https://chatgpt.com/c/69269693-df64-832c-acc6-e12684864532](https://chatgpt.com/c/69269693-df64-832c-acc6-e12684864532)  

## Prompt:
What is vLLM

## Response:
vLLM is a turbo-charged inference engine for large language models. Picture an LLM server that stops wasting GPU memory, feeds tokens into the model like a well-trained pit crew, and keeps latency down even when many requests pile up at once. That’s vLLM.

A few core pieces make it interesting:

**PagedAttention**  
It treats KV-cache (the memory of the model during generation) like a virtual memory system. Instead of allocating giant contiguous blocks, it chops everything into fixed GPU pages. This avoids fragmentation, lets you pack far more concurrent requests, and keeps GPU memory from being eaten alive.

**Continuous batching**  
New prompts can jump into a running batch. That means you don’t wait for other prompts to finish before the GPU sees more work. The GPU stays fed, and throughput skyrockets.

**LoRA support**  
It can load base models plus fine-tuned adapters without duplicating memory.

**OpenAI-compatible server**  
You run a vLLM server locally or on your cluster and hit it with the same API calls you’d send to OpenAI or Gemini. This makes it an instant drop-in replacement for cost-saving or on-prem setups.

**Why people like it**  
It squeezes far more output per GPU dollar than traditional inference systems. Especially useful when you run lots of concurrent requests—chatbots, agents, pipelines, RAG systems, or multiple users hitting one GPU.

**Where it shines for your work**  
For something like TARS, where multiple agents want to call LLMs concurrently and you want local control, vLLM is a natural backbone. It’s one of the few inference engines that doesn’t collapse under agent storms. It also pairs well with Ollama if you want a split setup—Ollama for coding, vLLM for batch/agent inference.

If you want, I can sketch how a TARS LLM layer could use both Ollama and vLLM with dynamic routing depending on request type.

## Prompt:
Sure, for F# please 

2/2

## Response:
Pensé pendant 4 secondes

Nice, let’s wire some LLM plumbing in F#. I’ll keep it practical and TARS-friendly.

Below is a **minimal but extensible F# LLM layer** that can:

*   Talk to **Ollama** (local HTTP API)
*   Talk to **vLLM** (OpenAI-compatible HTTP API)
*   **Route dynamically** based on a policy (e.g., “coding vs. reasoning”, “fast vs. smart”, etc.)

You can drop this into a `.fs` file in your TARS core and evolve it later.

* * *

1\. Shared domain types
-----------------------

First a tiny, engine-agnostic LLM model:

```fsharp
namespace Tars.Llm

open System

type Role =
    | System
    | User
    | Assistant

type LlmMessage = {
    Role: Role
    Content: string
}

/// High-level request from TARS – independent of specific backend.
type LlmRequest = {
    ModelHint: string option  // e.g. "code", "reasoning", "cheap", "deep"
    MaxTokens: int option
    Temperature: float option
    Messages: LlmMessage list
}

/// Unified response type
type LlmResponse = {
    Text: string
    FinishReason: string option
    Raw: string option // raw JSON if you want to log/debug
}

type LlmBackend =
    | Ollama of model: string
    | Vllm of model: string

/// Routing policy result
type RoutedBackend = {
    Backend: LlmBackend
    Endpoint: Uri
}
```

* * *

2\. Simple routing policy
-------------------------

This is where TARS gets opinionated: based on `ModelHint` (or later: task metadata), you pick Ollama vs vLLM.

```fsharp
module Tars.Llm.Routing =

    open System

    open Tars.Llm

    type RoutingConfig = {
        OllamaBaseUri: Uri      // e.g. http://localhost:11434/
        VllmBaseUri: Uri        // e.g. http://localhost:8000/ (OpenAI-compatible)
        DefaultOllamaModel: string
        DefaultVllmModel: string
    }

    /// Super simple, easy-to-evolve routing logic
    let chooseBackend (cfg: RoutingConfig) (req: LlmRequest) : RoutedBackend =
        match req.ModelHint |> Option.defaultValue "" with
        | hint when hint.Contains("code", StringComparison.OrdinalIgnoreCase) ->
            {
                Backend = Ollama cfg.DefaultOllamaModel
                Endpoint = cfg.OllamaBaseUri
            }
        | hint when hint.Contains("cheap", StringComparison.OrdinalIgnoreCase) ->
            {
                Backend = Ollama cfg.DefaultOllamaModel
                Endpoint = cfg.OllamaBaseUri
            }
        | hint when hint.Contains("reason", StringComparison.OrdinalIgnoreCase)
                || hint.Contains("analysis", StringComparison.OrdinalIgnoreCase) ->
            {
                Backend = Vllm cfg.DefaultVllmModel
                Endpoint = cfg.VllmBaseUri
            }
        | _ ->
            // Default: send to vLLM (better throughput for agent swarms)
            {
                Backend = Vllm cfg.DefaultVllmModel
                Endpoint = cfg.VllmBaseUri
            }
```

Later you can plug in a more complex heuristic (load, latency, cost model, etc.), but this is already usable.

* * *

3\. Ollama client (F#)
----------------------

Ollama has a very simple HTTP API, e.g. `POST /api/chat` or `POST /api/generate`.

Here’s a basic **chat-style** call (`/api/chat`) in F#:

```fsharp
namespace Tars.Llm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

module OllamaClient =

    open Tars.Llm

    [<CLIMutable>]
    type OllamaMessageDto = {
        role: string
        content: string
    }

    [<CLIMutable>]
    type OllamaRequestDto = {
        model: string
        messages: OllamaMessageDto list
        stream: bool
        temperature: float option
    }

    [<CLIMutable>]
    type OllamaResponseMessageDto = {
        role: string
        content: string
    }

    [<CLIMutable>]
    type OllamaResponseDto = {
        model: string
        message: OllamaResponseMessageDto
        done: bool
    }

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toOllamaRole = function
        | Role.System -> "system"
        | Role.User -> "user"
        | Role.Assistant -> "assistant"

    let private toOllamaMessages (msgs: LlmMessage list) =
        msgs
        |> List.map (fun m -> { role = toOllamaRole m.Role; content = m.Content })

    let sendChatAsync (http: HttpClient) (baseUri: Uri) (model: string) (req: LlmRequest) : Task<LlmResponse> =
        task {
            let dto: OllamaRequestDto = {
                model = model
                messages = toOllamaMessages req.Messages
                stream = false
                temperature = req.Temperature
            }

            let uri = Uri(baseUri, "/api/chat")
            use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
            resp.EnsureSuccessStatusCode() |> ignore

            let! raw = resp.Content.ReadAsStringAsync()
            let parsed = JsonSerializer.Deserialize<OllamaResponseDto>(raw, jsonOptions)

            match parsed with
            | null ->
                return {
                    Text = ""
                    FinishReason = Some "parse_error"
                    Raw = Some raw
                }
            | p ->
                return {
                    Text = p.message.content
                    FinishReason = Some (if p.done then "done" else "unknown")
                    Raw = Some raw
                }
        }
```

* * *

4\. vLLM (OpenAI-compatible) client
-----------------------------------

vLLM exposes an OpenAI-like API, typically:

*   `POST /v1/chat/completions`

We wrap that:

```fsharp
namespace Tars.Llm

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

module VllmClient =

    open Tars.Llm

    [<CLIMutable>]
    type OpenAiMessageDto = {
        role: string
        content: string
    }

    [<CLIMutable>]
    type OpenAiRequestDto = {
        model: string
        messages: OpenAiMessageDto list
        max_tokens: int option
        temperature: float option
        stream: bool option
    }

    [<CLIMutable>]
    type OpenAiChoiceMessageDto = {
        role: string
        content: string
    }

    [<CLIMutable>]
    type OpenAiChoiceDto = {
        index: int
        message: OpenAiChoiceMessageDto
        finish_reason: string
    }

    [<CLIMutable>]
    type OpenAiResponseDto = {
        id: string
        choices: OpenAiChoiceDto list
    }

    let private jsonOptions =
        JsonSerializerOptions(
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        )

    let private toOpenAiRole = function
        | Role.System -> "system"
        | Role.User -> "user"
        | Role.Assistant -> "assistant"

    let private toOpenAiMessages (msgs: LlmMessage list) =
        msgs
        |> List.map (fun m -> { role = toOpenAiRole m.Role; content = m.Content })

    let sendChatAsync (http: HttpClient) (baseUri: Uri) (model: string) (req: LlmRequest) : Task<LlmResponse> =
        task {
            let dto: OpenAiRequestDto = {
                model = model
                messages = toOpenAiMessages req.Messages
                max_tokens = req.MaxTokens
            temperature = req.Temperature
            stream = Some false
        }

        let uri = Uri(baseUri, "/v1/chat/completions")
        use! resp = http.PostAsJsonAsync(uri, dto, jsonOptions)
        resp.EnsureSuccessStatusCode() |> ignore

        let! raw = resp.Content.ReadAsStringAsync()
        let parsed = JsonSerializer.Deserialize<OpenAiResponseDto>(raw, jsonOptions)

        match parsed with
        | null ->
            return {
                Text = ""
                FinishReason = Some "parse_error"
                Raw = Some raw
            }
        | p ->
            let choice =
                p.choices
                |> List.sortBy (fun c -> c.index)
                |> List.tryHead

            match choice with
            | None ->
                return {
                    Text = ""
                    FinishReason = Some "no_choices"
                    Raw = Some raw
                }
            | Some c ->
                return {
                    Text = c.message.content
                    FinishReason = Some c.finish_reason
                    Raw = Some raw
                }
    }
```

* * *

5\. Unified LLM service
-----------------------

Now glue it together: one function TARS calls, which internally routes and calls the right backend.

```fsharp
namespace Tars.Llm

open System.Net.Http
open System.Threading.Tasks

module LlmService =

    open Tars.Llm
    open Tars.Llm.Routing

    type LlmServiceConfig = {
        Routing: RoutingConfig
    }

    type ILlmService =
        abstract member CompleteAsync : LlmRequest -> Task<LlmResponse>

    type DefaultLlmService(httpClient: HttpClient, cfg: LlmServiceConfig) =

        interface ILlmService with
            member _.CompleteAsync (req: LlmRequest) : Task<LlmResponse> =
                task {
                    let routed = chooseBackend cfg.Routing req

                    match routed.Backend with
                    | Ollama model ->
                        return! OllamaClient.sendChatAsync httpClient routed.Endpoint model req
                    | Vllm model ->
                        return! VllmClient.sendChatAsync httpClient routed.Endpoint model req
                }
```

Registering this in ASP.NET Core DI (if you host TARS that way):

```fsharp
// in Program.fs or Startup.fs
open System
open System.Net.Http
open Microsoft.Extensions.DependencyInjection
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

let configureServices (services: IServiceCollection) =
    services.AddHttpClient<ILlmService, DefaultLlmService>()
            .ConfigureHttpClient(fun client ->
                client.Timeout <- TimeSpan.FromSeconds(120.0)
            )
            |> ignore

    let routingCfg : RoutingConfig = {
        OllamaBaseUri = Uri("http://localhost:11434/")
        VllmBaseUri   = Uri("http://localhost:8000/") // or whatever your vLLM port is
        DefaultOllamaModel = "qwen2.5-coder:latest"
        DefaultVllmModel   = "qwen2.5-72b-instruct"
    }

    let svcCfg : LlmServiceConfig = { Routing = routingCfg }

    services.AddSingleton<LlmServiceConfig>(svcCfg) |> ignore
```

* * *

6\. Using the service from TARS
-------------------------------

Example of a TARS module asking “do code-y stuff, so probably Ollama”:

```fsharp
open System.Threading.Tasks
open Tars.Llm

let askForRefactor (llm: ILlmService) (code: string) : Task<string> =
    task {
        let req: LlmRequest = {
            ModelHint = Some "code"
            MaxTokens = Some 512
            Temperature = Some 0.2
            Messages = [
                { Role = Role.System; Content = "You are a senior F# and C# refactoring assistant." }
                { Role = Role.User; Content = $"Refactor this code for readability and safety:\n\n{code}" }
            ]
        }

        let! resp = llm.CompleteAsync req
        return resp.Text
    }
```

Or “deep reasoning, probably vLLM”:

```fsharp
let askForTarsDesign (llm: ILlmService) (question: string) : Task<string> =
    task {
        let req: LlmRequest = {
            ModelHint = Some "reasoning"
            MaxTokens = Some 2048
            Temperature = Some 0.4
            Messages = [
                { Role = Role.System; Content = "You are TARS' architectural co-designer. Be explicit, structured, and critical." }
                { Role = Role.User; Content = question }
            ]
        }

        let! resp = llm.CompleteAsync req
        return resp.Text
    }
```

* * *

This gives you:

*   A clean F# façade for **multiple LLM backends**.
*   A central place to tweak routing logic as TARS evolves.
*   Enough type structure to add logging, metrics, retries, or “backpressure for LLM calls” later.

Next natural step in TARS-land is to add:

*   A **metrics wrapper** (`ExecutionTime`, `Tokens`, `FailureCount`) around `ILlmService`.
*   A tiny **policy engine** where the router can use past metrics (or agent feedback) to choose models dynamically.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)