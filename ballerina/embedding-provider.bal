// Copyright (c) 2025 WSO2 LLC (http://www.wso2.com).
//
// WSO2 LLC. licenses this file to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import ballerina/ai;
import ballerina/ai.observe;
import ballerina/http;

# EmbeddingProvider provides an interface for interacting with Mistral AI embedding models.
@display {
    label: "Mistral Embedding Provider"
}
public distinct isolated client class EmbeddingProvider {
    *ai:EmbeddingProvider;
    private final http:Client embeddingClient;
    private final string modelType;
    private final int? outputDimension;
    private final EmbeddingDataType? outputDataType;

    # Initializes the Mistral AI embedding model with the given connection configuration.
    #
    # + apiKey - The Mistral AI API key
    # + modelType - The Mistral AI embedding model name
    # + serviceUrl - The base URL of Mistral AI API endpoint
    # + outputDimension - The number of dimensions of the output embeddings (e.g., up to 3072 for `codestral-embed`); uses the model default if not set
    # + outputDataType - The data type of the output embedding values; uses `float` if not set
    # + connectionConfig - Additional HTTP connection configuration
    # + return - `nil` on successful initialization; otherwise, returns an `ai:Error`
    public isolated function init(@display {label: "API Key"} string apiKey,
            @display {label: "Embedding Model Type"} MISTRAL_AI_EMBEDDING_MODEL_NAMES modelType,
            @display {label: "Service URL"} string serviceUrl = DEFAULT_MISTRAL_AI_SERVICE_URL,
            @display {label: "Output Dimension"} int? outputDimension = (),
            @display {label: "Output Data Type"} EmbeddingDataType? outputDataType = (),
            @display {label: "Connection Configuration"} *ConnectionConfig connectionConfig
    ) returns ai:Error? {
        if outputDimension is int && outputDimension < 1 {
            return error ai:Error(string `Invalid output dimension: ${outputDimension}. The value must be greater than 0.`);
        }
        http:Client|error embeddingClient = new (serviceUrl, getHttpClientConfig(apiKey, connectionConfig));
        if embeddingClient is error {
            return error ai:Error("Failed to initialize Mistral embedding provider", embeddingClient);
        }
        self.embeddingClient = embeddingClient;
        self.modelType = modelType;
        self.outputDimension = outputDimension;
        self.outputDataType = outputDataType;
    }

    # Generates an embedding vector for the provided chunk.
    #
    # + chunk - The `ai:Chunk` containing the content to embed
    # + return - The resulting `ai:Embedding` on success; otherwise, returns an `ai:Error`
    isolated remote function embed(ai:Chunk chunk) returns ai:Embedding|ai:Error {
        observe:EmbeddingSpan span = observe:createEmbeddingSpan(self.modelType);
        span.addProvider("mistral_ai");

        if chunk !is ai:TextDocument|ai:TextChunk {
            ai:Error err = error("Unsupported document type. only 'ai:TextDocument|ai:TextChunk' is supported");
            span.close(err);
            return err;
        }
        span.addInputContent(chunk.content);
        ai:Embedding[]|ai:Error embeddings = self.generateEmbeddings([chunk.content], span);
        if embeddings is ai:Error {
            span.close(embeddings);
            return embeddings;
        }
        span.close();
        return embeddings[0];
    }

    # Converts a batch of chunks into embeddings.
    #
    # + chunks - The array of chunks to be converted into embeddings
    # + return - An array of embeddings on success, or an `ai:Error`
    isolated remote function batchEmbed(ai:Chunk[] chunks) returns ai:Embedding[]|ai:Error {
        observe:EmbeddingSpan span = observe:createEmbeddingSpan(self.modelType);
        span.addProvider("mistral_ai");

        string[] inputs = [];
        foreach ai:Chunk chunk in chunks {
            if chunk !is ai:TextDocument|ai:TextChunk {
                ai:Error err = error("Unsupported chunk type. only 'ai:TextChunk[]|ai:TextDocument[]' is supported");
                span.close(err);
                return err;
            }
            inputs.push(chunk.content);
        }
        if inputs.length() == 0 {
            span.close();
            return [];
        }
        span.addInputContent(inputs);
        ai:Embedding[]|ai:Error embeddings = self.generateEmbeddings(inputs, span);
        if embeddings is ai:Error {
            span.close(embeddings);
            return embeddings;
        }
        span.close();
        return embeddings;
    }

    private isolated function generateEmbeddings(string[] inputs, observe:EmbeddingSpan span)
            returns ai:Embedding[]|ai:Error {
        EmbeddingRequest request = {model: self.modelType, input: inputs};
        int? outputDimension = self.outputDimension;
        if outputDimension is int {
            request.output_dimension = outputDimension;
        }
        EmbeddingDataType? outputDataType = self.outputDataType;
        if outputDataType is EmbeddingDataType {
            request.output_dtype = outputDataType;
        }

        json|http:ClientError payload = self.embeddingClient->/embeddings.post(request, {"Accept": "application/json"});
        if payload is http:PayloadBindingError {
            return error ai:LlmInvalidResponseError("Invalid response received from the embedding model", payload);
        }
        if payload is http:ClientError {
            return error ai:LlmConnectionError("Error while connecting to the embedding model", payload);
        }
        EmbeddingResponse|error response = payload.fromJsonWithType();
        if response is error {
            return error ai:LlmInvalidResponseError("Invalid response received from the embedding model", response);
        }

        string? responseModel = response.model;
        if responseModel is string {
            span.addResponseModel(responseModel);
        }
        int? promptTokens = response.usage?.prompt_tokens;
        if promptTokens is int {
            span.addInputTokenCount(promptTokens);
        }
        return getOrderedEmbeddings(response.data, inputs.length());
    }
}

isolated function getOrderedEmbeddings(EmbeddingData[] data, int expectedCount) returns ai:Embedding[]|ai:Error {
    if data.length() != expectedCount {
        return error ai:LlmInvalidResponseError(string `Expected ${expectedCount} embeddings ` +
            string `in the response, but received ${data.length()}`);
    }
    ai:Embedding?[] embeddings = [];
    embeddings.setLength(expectedCount);
    foreach int i in 0 ..< data.length() {
        EmbeddingData item = data[i];
        ai:Vector? embedding = item.embedding;
        if embedding is () {
            return error ai:LlmInvalidResponseError(string `Missing embedding at position ${i} in the response`);
        }
        int index = item.index ?: i;
        if index < 0 || index >= expectedCount || embeddings[index] !is () {
            return error ai:LlmInvalidResponseError(string `Invalid embedding index ${index} in the response`);
        }
        embeddings[index] = embedding;
    }
    return from ai:Embedding? embedding in embeddings
        where embedding is ai:Embedding
        select embedding;
}
