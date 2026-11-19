## Overview

Mistral AI provides high-performance, open-weights large language models (LLMs) designed for efficiency and versatility. The Mistral connector enables integration with Mistral AI's models, allowing you to build advanced natural language processing applications.

### Key Features

- Seamless integration with high-performance Mistral LLMs
- Support for chat completions and text generation
- Text embeddings with `mistral-embed` and `codestral-embed` models
- Efficient handling of model parameters and prompts
- Simplified access to Mistral AI's API endpoints
- Secure communication with API key authentication
- GraalVM compatible for native image builds

## Prerequisites

Before using this module in your Ballerina application, first you must obtain the nessary configuration to engage the LLM.

- Create a [Mistral account](https://console.mistral.ai/).
- Obtain an API key by following [these instructions](https://docs.mistral.ai/getting-started/quickstart/#account-setup)

## Quickstart

To use the `ai.mistral` module in your Ballerina application, update the `.bal` file as follows:

### Step 1: Import the module

Import the `ai.mistral` module.

```ballerina
import ballerinax/ai.mistral;
```

### Step 2: Intialize the Model Provider

Here's how to initialize the Model Provider:

```ballerina
import ballerina/ai;
import ballerinax/ai.mistral;

final ai:ModelProvider mistralModel = check new mistral:ModelProvider("mistralApiKey", mistral:MINISTRAL_3B_2410);
```

### Step 4: Invoke chat completion

```ballerina
ai:ChatMessage[] chatMessages = [{role: "user", content: "hi"}];
ai:ChatAssistantMessage response = check mistralModel->chat(chatMessages, tools = []);

chatMessages.push(response);
```

## Embedding Provider

The `mistral:EmbeddingProvider` implements `ai:EmbeddingProvider` and generates vector embeddings with Mistral AI embedding models.

### Initialize the Embedding Provider

```ballerina
import ballerina/ai;
import ballerinax/ai.mistral;

final ai:EmbeddingProvider embeddingProvider = check new mistral:EmbeddingProvider("mistralApiKey", mistral:MISTRAL_EMBED);
```

Optionally, set the output dimension and data type (supported by `codestral-embed`):

```ballerina
final ai:EmbeddingProvider codeEmbeddingProvider = check new mistral:EmbeddingProvider("mistralApiKey",
    mistral:CODESTRAL_EMBED, outputDimension = 256, outputDataType = mistral:EMBEDDING_INT8);
```

### Generate embeddings

```ballerina
ai:Embedding embedding = check embeddingProvider->embed(<ai:TextChunk>{content: "Hello, Mistral!"});

ai:Embedding[] embeddings = check embeddingProvider->batchEmbed([
    <ai:TextChunk>{content: "First chunk"},
    <ai:TextDocument>{content: "Second document"}
]);
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | - | The Mistral AI API key |
| `modelType` | `MISTRAL_AI_EMBEDDING_MODEL_NAMES` | - | `MISTRAL_EMBED`, `MISTRAL_EMBED_2312`, `CODESTRAL_EMBED` or `CODESTRAL_EMBED_2505` |
| `serviceUrl` | `string` | `https://api.mistral.ai/v1` | The base URL of the Mistral AI API (e.g., `https://api.eu.mistral.ai/v1`) |
| `outputDimension` | `int?` | `()` (model default) | Number of dimensions of the output embeddings; must be greater than 0 (`codestral-embed` supports up to 3072) |
| `outputDataType` | `EmbeddingDataType?` | `()` (`float`) | `EMBEDDING_FLOAT`, `EMBEDDING_INT8` or `EMBEDDING_UINT8` |
| `connectionConfig` | `*ConnectionConfig` | HTTP/2, 60 s timeout | Additional HTTP connection configuration, passed as named arguments (e.g., `timeout = 30`) |

### Notes

- Only `ai:TextChunk` and `ai:TextDocument` inputs are supported.
- `batchEmbed` sends all chunks in a single request and returns the embeddings in input order. An empty array returns `[]` without calling the API. Mistral does not publish a per-request input limit; split very large batches before calling `batchEmbed`.
- Model defaults: `mistral-embed` returns 1024 dimensions and `codestral-embed` returns 1536 dimensions.
- Integer data types (`int8`, `uint8`) are returned as `float` values with the same numbers. Keep `outputDimension` and `outputDataType` consistent with the vector store you use.
- The `binary`/`ubinary` data types, `encoding_format` and `metadata` request fields are not supported.
- HTTP failures (e.g., 400, 401, 422, 429, 5xx), timeouts and connection failures return an `ai:LlmConnectionError`. For HTTP status failures, its cause is an `http:ApplicationResponseError` (`http:ClientRequestError` for 4xx, `http:RemoteServerError` for 5xx) that holds the status code, headers (e.g., `Retry-After`) and response body. Use `retryConfig` in the connection configuration to retry on status codes such as 429, 500, 502, 503 and 504.
- Invalid responses return an `ai:LlmInvalidResponseError`.
