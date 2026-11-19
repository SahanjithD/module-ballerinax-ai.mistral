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
import ballerina/http;
import ballerina/test;

const EMBEDDING_API_KEY = "test-api-key";
const UNSUPPORTED_DOCUMENT_ERROR = "Unsupported document type. only 'ai:TextDocument|ai:TextChunk' is supported";
const UNSUPPORTED_CHUNK_ERROR = "Unsupported chunk type. only 'ai:TextChunk[]|ai:TextDocument[]' is supported";
const CONNECTION_ERROR = "Error while connecting to the embedding model";
const INVALID_RESPONSE_ERROR = "Invalid response received from the embedding model";

final EmbeddingProvider defaultEmbeddingProvider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL);

@test:BeforeEach
function resetEmbeddingMock() {
    resetEmbeddingRequests();
}

// Request shape and authentication

@test:Config
function testEmbedRequestShape() returns error? {
    ai:Embedding embedding = check defaultEmbeddingProvider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertEquals(getLastEmbeddingRequest(), {model: "mistral-embed", input: ["Hello"]});
    test:assertEquals(getLastRequestHeader("Authorization"), "Bearer test-api-key");
    test:assertEquals(getLastRequestHeader("Accept"), "application/json");
    string? contentType = getLastRequestHeader("Content-Type");
    test:assertTrue(contentType is string && contentType.startsWith("application/json"));
    test:assertEquals(getEmbeddingRequestCount(), 1);
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
}

@test:Config
function testBatchEmbedRequestShape() returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: "first"},
        <ai:TextDocument>{content: "second"}
    ]);
    test:assertEquals(getLastEmbeddingRequest(), {model: "mistral-embed", input: ["first", "second"]});
    test:assertEquals(getLastRequestHeader("Authorization"), "Bearer test-api-key");
    test:assertEquals(getEmbeddingRequestCount(), 1);
    test:assertEquals(embeddings.length(), 2);
    assertVector(embeddings[0], 1024, getMockFloatValue(0, 0));
    assertVector(embeddings[1], 1024, getMockFloatValue(1, 0));
}

@test:Config
function testEmbedWithCustomApiKey() returns error? {
    EmbeddingProvider provider = check new ("another-test-key", MISTRAL_EMBED, EMBEDDING_SERVICE_URL);
    _ = check provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertEquals(getLastRequestHeader("Authorization"), "Bearer another-test-key");
}

@test:Config
function testEmbedWithEmptyApiKey() {
    EmbeddingProvider|ai:Error provider = new ("", MISTRAL_EMBED, EMBEDDING_SERVICE_URL);
    if provider is ai:Error {
        return;
    }
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    if embedding is ai:Error {
        test:assertTrue(embedding is ai:LlmConnectionError);
        test:assertEquals(getEmbeddingRequestCount(), 0);
        return;
    }
    string? authorization = getLastRequestHeader("Authorization");
    test:assertTrue(authorization is () || authorization.trim() == "Bearer");
}

// modelType

@test:Config {dataProvider: embeddingModels}
function testModelType(MISTRAL_AI_EMBEDDING_MODEL_NAMES modelType, int expectedLength) returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, modelType, EMBEDDING_SERVICE_URL);
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertEquals(getLastEmbeddingRequest(), {model: modelType, input: ["Hello"]});
    assertVector(embedding, expectedLength, getMockFloatValue(0, 0));
}

function embeddingModels() returns [MISTRAL_AI_EMBEDDING_MODEL_NAMES, int][] => [
    [MISTRAL_EMBED, 1024],
    [MISTRAL_EMBED_2312, 1024],
    [CODESTRAL_EMBED, 1536],
    [CODESTRAL_EMBED_2505, 1536]
];

// serviceUrl

@test:Config
function testDefaultServiceUrl() {
    EmbeddingProvider|ai:Error provider = new (EMBEDDING_API_KEY, MISTRAL_EMBED);
    test:assertTrue(provider is EmbeddingProvider);
}

@test:Config
function testServiceUrlWithTrailingSlash() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL + "/");
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
    test:assertEquals(getEmbeddingRequestCount(), 1);
}

@test:Config {dataProvider: invalidServiceUrls}
function testInvalidServiceUrl(string serviceUrl) {
    EmbeddingProvider|ai:Error provider = new (EMBEDDING_API_KEY, MISTRAL_EMBED, serviceUrl);
    if provider is ai:Error {
        test:assertEquals(provider.message(), "Failed to initialize Mistral embedding provider");
        return;
    }
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertTrue(embedding is ai:LlmConnectionError);
}

function invalidServiceUrls() returns string[][] => [
    ["not a url"],
    [""]
];

@test:Config
function testUnreachableServiceUrl() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, "http://localhost:8199/mistral");
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertTrue(embedding is ai:LlmConnectionError);
    if embedding is ai:Error {
        test:assertEquals(embedding.message(), CONNECTION_ERROR);
    }
}

@test:Config
function testUnknownServicePath() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, "http://localhost:8095/unknown");
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    assertHttpError(embedding, 404);
}

// outputDimension

@test:Config
function testDefaultOutputDimensionIsOmitted() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL);
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    map<json> request = check getLastEmbeddingRequest().ensureType();
    test:assertFalse(request.hasKey("output_dimension"));
    test:assertFalse(request.hasKey("output_dtype"));
    assertVector(embedding, 1536, getMockFloatValue(0, 0));
}

@test:Config {dataProvider: validOutputDimensions}
function testValidOutputDimension(int outputDimension) returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL,
        outputDimension = outputDimension
    );
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertEquals(getLastEmbeddingRequest(),
            {model: "codestral-embed", input: ["Hello"], output_dimension: outputDimension});
    assertVector(embedding, outputDimension, getMockFloatValue(0, 0));
}

function validOutputDimensions() returns int[][] => [[1], [8], [10], [256], [512], [1024], [1536], [3072], [4096]];

@test:Config {dataProvider: invalidOutputDimensions}
function testInvalidOutputDimension(int outputDimension) {
    EmbeddingProvider|ai:Error provider = new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL,
        outputDimension = outputDimension
    );
    if provider !is ai:Error {
        test:assertFail("Expected an error for output dimension " + outputDimension.toString());
    }
    test:assertEquals(provider.message(),
            string `Invalid output dimension: ${outputDimension}. The value must be greater than 0.`);
    test:assertEquals(getEmbeddingRequestCount(), 0);
}

function invalidOutputDimensions() returns int[][] => [[0], [-1], [-1024], [int:MIN_VALUE]];

@test:Config
function testOutputDimensionRejectedByServer() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL,
        outputDimension = 4096
    );
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: STATUS_PREFIX + "422"});
    assertHttpError(embedding, 422);
    test:assertEquals(getLastEmbeddingRequest(),
            {model: "codestral-embed", input: [STATUS_PREFIX + "422"], output_dimension: 4096});
}

// outputDataType

@test:Config {dataProvider: outputDataTypes}
function testOutputDataType(EmbeddingDataType outputDataType) returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL,
        outputDataType = outputDataType
    );
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertEquals(getLastEmbeddingRequest(),
            {model: "codestral-embed", input: ["Hello"], output_dtype: outputDataType});
    assertVector(embedding, 1536, getExpectedFirstValue(outputDataType, 0));
}

function outputDataTypes() returns EmbeddingDataType[][] => [[EMBEDDING_FLOAT], [EMBEDDING_INT8], [EMBEDDING_UINT8]];

@test:Config
function testIntegerEmbeddingValuesAreConvertedToFloat() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL,
        outputDataType = EMBEDDING_INT8
    );
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: MALFORMED_PREFIX + "INT8_VALUES"});
    test:assertEquals(embedding, <ai:Vector>[-128.0, 0.0, 127.0]);
}

@test:Config
function testExponentNotationEmbeddingValues() returns error? {
    ai:Embedding embedding = check defaultEmbeddingProvider->embed(
        <ai:TextChunk>{content: MALFORMED_PREFIX + "EXPONENT_VALUES"});
    test:assertEquals(embedding, <ai:Vector>[0.0015, -200.0, 0.0, 3.0]);
}

// Cartesian product: modelType x outputDataType x outputDimension

@test:Config {dataProvider: cartesianEmbeddingCases}
function testEmbeddingParameterCartesianProduct(MISTRAL_AI_EMBEDDING_MODEL_NAMES modelType,
        EmbeddingDataType? outputDataType, int? outputDimension) returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, modelType, EMBEDDING_SERVICE_URL, outputDimension,
        outputDataType
    );
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertEquals(getLastEmbeddingRequest(), getExpectedRequest(modelType, ["Hello"], outputDimension,
                    outputDataType));
    assertVector(embedding, getExpectedEmbeddingLength(modelType, outputDimension),
            getExpectedFirstValue(outputDataType, 0));
}

function cartesianEmbeddingCases() returns map<[MISTRAL_AI_EMBEDDING_MODEL_NAMES, EmbeddingDataType?, int?]> {
    MISTRAL_AI_EMBEDDING_MODEL_NAMES[] models = [
        MISTRAL_EMBED,
        MISTRAL_EMBED_2312,
        CODESTRAL_EMBED,
        CODESTRAL_EMBED_2505
    ];
    EmbeddingDataType?[] dataTypes = [(), EMBEDDING_FLOAT, EMBEDDING_INT8, EMBEDDING_UINT8];
    int?[] dimensions = [(), 256, 512, 1024, 1536, 3072];
    map<[MISTRAL_AI_EMBEDDING_MODEL_NAMES, EmbeddingDataType?, int?]> cases = {};
    foreach MISTRAL_AI_EMBEDDING_MODEL_NAMES model in models {
        foreach EmbeddingDataType? dataType in dataTypes {
            foreach int? dimension in dimensions {
                cases[string `${model}_${dataType ?: "default"}_${dimension ?: "default"}`] =
                    [model, dataType, dimension];
            }
        }
    }
    return cases;
}

@test:Config
function testCartesianProductCaseCount() {
    test:assertEquals(cartesianEmbeddingCases().length(), 96);
}

// Pairwise: operation x chunk type x HTTP version x outputDataType x outputDimension x service URL

type PairwiseCase [string, string, http:HttpVersion, EmbeddingDataType?, int?, string,
        MISTRAL_AI_EMBEDDING_MODEL_NAMES];

@test:Config {dataProvider: pairwiseEmbeddingCases}
function testEmbeddingParameterPairwise(string operation, string chunkType, http:HttpVersion httpVersion,
        EmbeddingDataType? outputDataType, int? outputDimension, string serviceUrl,
        MISTRAL_AI_EMBEDDING_MODEL_NAMES modelType) returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, modelType, serviceUrl, outputDimension,
        outputDataType, httpVersion = httpVersion
    );
    ai:TextChunk|ai:TextDocument first = chunkType == "chunk" ? <ai:TextChunk>{content: "first"} :
        <ai:TextDocument>{content: "first"};
    ai:TextChunk|ai:TextDocument second = chunkType == "chunk" ? <ai:TextChunk>{content: "second"} :
        <ai:TextDocument>{content: "second"};
    int expectedLength = getExpectedEmbeddingLength(modelType, outputDimension);
    if operation == "embed" {
        ai:Embedding embedding = check provider->embed(first);
        test:assertEquals(getLastEmbeddingRequest(), getExpectedRequest(modelType, ["first"], outputDimension,
                        outputDataType));
        assertVector(embedding, expectedLength, getExpectedFirstValue(outputDataType, 0));
        return;
    }
    ai:Embedding[] embeddings = check provider->batchEmbed([first, second]);
    test:assertEquals(getLastEmbeddingRequest(), getExpectedRequest(modelType, ["first", "second"],
                    outputDimension, outputDataType));
    test:assertEquals(embeddings.length(), 2);
    assertVector(embeddings[0], expectedLength, getExpectedFirstValue(outputDataType, 0));
    assertVector(embeddings[1], expectedLength, getExpectedFirstValue(outputDataType, 1));
}

function pairwiseEmbeddingCases() returns PairwiseCase[] {
    string plain = EMBEDDING_SERVICE_URL;
    string slash = EMBEDDING_SERVICE_URL + "/";
    return [
        ["embed", "chunk", http:HTTP_1_1, (), (), plain, MISTRAL_EMBED],
        ["embed", "document", http:HTTP_2_0, EMBEDDING_FLOAT, 256, slash, MISTRAL_EMBED_2312],
        ["batch", "chunk", http:HTTP_1_1, EMBEDDING_INT8, 3072, slash, CODESTRAL_EMBED],
        ["batch", "document", http:HTTP_2_0, EMBEDDING_UINT8, (), plain, CODESTRAL_EMBED_2505],
        ["embed", "document", http:HTTP_2_0, EMBEDDING_INT8, 3072, plain, MISTRAL_EMBED],
        ["batch", "chunk", http:HTTP_1_1, EMBEDDING_FLOAT, 256, plain, CODESTRAL_EMBED],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_UINT8, (), slash, MISTRAL_EMBED_2312],
        ["batch", "chunk", http:HTTP_2_0, (), 256, slash, CODESTRAL_EMBED_2505],
        ["embed", "document", http:HTTP_1_1, (), 3072, plain, CODESTRAL_EMBED],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_FLOAT, (), plain, CODESTRAL_EMBED_2505],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_FLOAT, 3072, plain, MISTRAL_EMBED],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_INT8, (), plain, MISTRAL_EMBED_2312],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_INT8, 256, plain, CODESTRAL_EMBED_2505],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_UINT8, 256, plain, MISTRAL_EMBED],
        ["embed", "chunk", http:HTTP_1_1, EMBEDDING_UINT8, 3072, plain, CODESTRAL_EMBED]
    ];
}

// ConnectionConfig fields

@test:Config {dataProvider: connectionConfigs}
function testConnectionConfigField(ConnectionConfig connectionConfig) returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL, (), (),
        connectionConfig
    );
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: "Hello"});
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
    ai:Embedding[] embeddings = check provider->batchEmbed([<ai:TextChunk>{content: "Hello"}]);
    test:assertEquals(embeddings.length(), 1);
}

function connectionConfigs() returns map<[ConnectionConfig]> => {
    "default": [{}],
    "httpVersion1_0": [{httpVersion: http:HTTP_1_0}],
    "httpVersion1_1": [{httpVersion: http:HTTP_1_1}],
    "httpVersion2_0": [{httpVersion: http:HTTP_2_0}],
    "http1Settings": [
        {
            httpVersion: http:HTTP_1_1,
            http1Settings: {keepAlive: http:KEEPALIVE_NEVER, chunking: http:CHUNKING_NEVER}
        }
    ],
    "http2Settings": [{http2Settings: {http2PriorKnowledge: false, http2InitialWindowSize: 131070}}],
    "timeout": [{timeout: 5}],
    "forwardedEnable": [{forwarded: "enable"}],
    "forwardedTransition": [{forwarded: "transition"}],
    "poolConfig": [{poolConfig: {maxActiveConnections: 2, maxIdleConnections: 1, waitTime: 5}}],
    "cacheDisabled": [{cache: {enabled: false}}],
    "cacheEnabled": [{cache: {enabled: true, capacity: 8}}],
    "compressionNever": [{compression: http:COMPRESSION_NEVER}],
    "compressionAlways": [{compression: http:COMPRESSION_ALWAYS}],
    "circuitBreaker": [
        {
            circuitBreaker: {
                rollingWindow: {timeWindow: 10, bucketSize: 2, requestVolumeThreshold: 0},
                failureThreshold: 0.5,
                resetTime: 10,
                statusCodes: [500]
            }
        }
    ],
    "retryConfig": [{retryConfig: {count: 2, interval: 0.1, statusCodes: [503]}}],
    "responseLimits": [{responseLimits: {maxStatusLineLength: 8192, maxHeaderSize: 16384, maxEntityBodySize: -1}}],
    "validationDisabled": [{validation: false}],
    "validationEnabled": [{validation: true}]
};

@test:Config
function testTimeoutExceeded() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL, timeout = 1);
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: DELAY_PREFIX + "3"});
    test:assertTrue(embedding is ai:LlmConnectionError);
    if embedding is ai:Error {
        test:assertEquals(embedding.message(), CONNECTION_ERROR);
    }
}

@test:Config
function testDelayWithinTimeout() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL, timeout = 5);
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: DELAY_PREFIX + "0.5"});
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
}

@test:Config
function testRetryConfigRecoversFromTransientFailure() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL,
        retryConfig = {count: 3, interval: 0.1, statusCodes: [503]}
    );
    ai:Embedding embedding = check provider->embed(<ai:TextChunk>{content: FLAKY_PREFIX + "retry"});
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
    test:assertEquals(getEmbeddingRequestCount(), 3);
}

@test:Config
function testTransientFailureWithoutRetry() returns error? {
    ai:Embedding|ai:Error embedding = defaultEmbeddingProvider->embed(<ai:TextChunk>{content: FLAKY_PREFIX + "noretry"});
    assertHttpError(embedding, 503);
    test:assertEquals(getEmbeddingRequestCount(), 1);
}

@test:Config
function testRetryOnRateLimit() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL,
        retryConfig = {count: 2, interval: 0.1, statusCodes: [429]}
    );
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: STATUS_PREFIX + "429"});
    assertHttpError(embedding, 429);
    test:assertEquals(getEmbeddingRequestCount(), 3);
}

@test:Config
function testResponseLimitExceeded() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL,
        httpVersion = http:HTTP_1_1, timeout = 5, responseLimits = {maxEntityBodySize: 100}
    );
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertTrue(embedding is ai:LlmError);
}

@test:Config
function testProxyConfigIsApplied() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL,
        httpVersion = http:HTTP_1_1, proxy = {host: "localhost", port: 8199}
    );
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertTrue(embedding is ai:LlmConnectionError);
    test:assertEquals(getEmbeddingRequestCount(), 0);
}

@test:Config
function testSecureSocketConfigIsApplied() {
    EmbeddingProvider|ai:Error provider = new (EMBEDDING_API_KEY, MISTRAL_EMBED, "https://localhost:8095/mistral",
        secureSocket = {cert: "tests/resources/missing-cert.pem"}
    );
    if provider is ai:Error {
        test:assertEquals(provider.message(), "Failed to initialize Mistral embedding provider");
        return;
    }
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertTrue(embedding is ai:LlmConnectionError);
    test:assertEquals(getEmbeddingRequestCount(), 0);
}

@test:Config
function testCircuitBreakerOpensAfterFailures() returns error? {
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, MISTRAL_EMBED, EMBEDDING_SERVICE_URL,
        circuitBreaker = {
            rollingWindow: {timeWindow: 60, bucketSize: 10, requestVolumeThreshold: 0},
            failureThreshold: 0.1,
            resetTime: 60,
            statusCodes: [500]
        }
    );
    foreach int i in 0 ..< 2 {
        ai:Embedding|ai:Error failure = provider->embed(<ai:TextChunk>{content: STATUS_PREFIX + "500"});
        test:assertTrue(failure is ai:LlmConnectionError);
    }
    ai:Embedding|ai:Error embedding = provider->embed(<ai:TextChunk>{content: "Hello"});
    test:assertTrue(embedding is ai:LlmConnectionError);
    test:assertTrue(getEmbeddingRequestCount() < 3);
}

// Inputs

@test:Config
function testEmbedWithTextDocument() returns error? {
    ai:Embedding embedding = check defaultEmbeddingProvider->embed(<ai:TextDocument>{content: "A document"});
    test:assertEquals(getLastEmbeddingRequest(), {model: "mistral-embed", input: ["A document"]});
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
}

@test:Config {dataProvider: unsupportedChunks}
function testEmbedWithUnsupportedChunk(ai:Chunk chunk) {
    ai:Embedding|ai:Error embedding = defaultEmbeddingProvider->embed(chunk);
    if embedding !is ai:Error {
        test:assertFail("Expected an error for an unsupported chunk");
    }
    test:assertEquals(embedding.message(), UNSUPPORTED_DOCUMENT_ERROR);
    test:assertEquals(getEmbeddingRequestCount(), 0);
}

@test:Config {dataProvider: unsupportedChunks}
function testBatchEmbedWithUnsupportedChunk(ai:Chunk chunk) {
    ai:Embedding[]|ai:Error embeddings = defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: "supported"},
        chunk
    ]);
    if embeddings !is ai:Error {
        test:assertFail("Expected an error for an unsupported chunk");
    }
    test:assertEquals(embeddings.message(), UNSUPPORTED_CHUNK_ERROR);
    test:assertEquals(getEmbeddingRequestCount(), 0);
}

function unsupportedChunks() returns map<[ai:Chunk]> => {
    "image": [<ai:ImageDocument>{content: "https://example.com/image.png"}],
    "audio": [<ai:AudioDocument>{content: "https://example.com/audio.mp3"}],
    "file": [<ai:FileDocument>{content: "https://example.com/file.pdf"}],
    "genericChunk": [{'type: "text", content: "not a text chunk"}]
};

@test:Config
function testBatchEmbedWithEmptyArray() returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([]);
    test:assertEquals(embeddings, []);
    test:assertEquals(getEmbeddingRequestCount(), 0);
}

@test:Config
function testBatchEmbedWithSingleChunk() returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([<ai:TextChunk>{content: "only"}]);
    test:assertEquals(embeddings.length(), 1);
    test:assertEquals(getLastEmbeddingRequest(), {model: "mistral-embed", input: ["only"]});
    assertVector(embeddings[0], 1024, getMockFloatValue(0, 0));
}

@test:Config {dataProvider: largeBatchSizes}
function testBatchEmbedLargeBatchInSingleRequest(int batchSize) returns error? {
    resetEmbeddingRequests();
    EmbeddingProvider provider = check new (EMBEDDING_API_KEY, CODESTRAL_EMBED, EMBEDDING_SERVICE_URL,
        outputDimension = 8
    );
    ai:TextChunk[] chunks = from int i in 0 ..< batchSize
        select {content: string `chunk ${i}`};
    ai:Embedding[] embeddings = check provider->batchEmbed(chunks);
    test:assertEquals(getEmbeddingRequestCount(), 1);
    test:assertEquals(embeddings.length(), batchSize);
    foreach int i in 0 ..< batchSize {
        assertVector(embeddings[i], 8, getMockFloatValue(i, 0));
    }
    json[] input = check (check getLastEmbeddingRequest().input).ensureType();
    test:assertEquals(input.length(), batchSize);
    test:assertEquals(input[batchSize - 1], string `chunk ${batchSize - 1}`);
}

function largeBatchSizes() returns int[][] => [[50], [64], [128], [512]];

@test:Config
function testBatchEmbedOverServerLimit() {
    ai:TextChunk[] chunks = [{content: STATUS_PREFIX + "400 too many inputs"}];
    foreach int i in 0 ..< 1000 {
        chunks.push({content: "extra"});
    }
    ai:Embedding[]|ai:Error embeddings = defaultEmbeddingProvider->batchEmbed(chunks);
    assertHttpError(embeddings, 400);
    test:assertEquals(getEmbeddingRequestCount(), 1);
}

@test:Config {dataProvider: edgeCaseInputs}
function testEmbedEdgeCaseInput(string content) returns error? {
    ai:Embedding embedding = check defaultEmbeddingProvider->embed(<ai:TextChunk>{content});
    test:assertEquals(getLastEmbeddingRequest(), {model: "mistral-embed", input: [content]});
    assertVector(embedding, 1024, getMockFloatValue(0, 0));
}

@test:Config {dataProvider: edgeCaseInputs}
function testBatchEmbedEdgeCaseInput(string content) returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content},
        <ai:TextDocument>{content}
    ]);
    test:assertEquals(getLastEmbeddingRequest(), {model: "mistral-embed", input: [content, content]});
    test:assertEquals(embeddings.length(), 2);
}

function edgeCaseInputs() returns map<[string]> {
    string longInput = "";
    foreach int i in 0 ..< 10000 {
        longInput += "long text ";
    }
    return {
        "empty": [""],
        "whitespace": ["   \t\n  "],
        "unicode": ["Mistral 日本語 中文 한국어 Ελληνικά русский"],
        "emoji": ["Embeddings 🚀🌍👩🏽‍💻"],
        "rtl": ["مرحبا بالعالم שלום עולם"],
        "escapes": ["quote \" backslash \\ newline \n tab \t"],
        "long": [longInput]
    };
}

// HTTP errors

@test:Config {dataProvider: httpErrorStatusCodes}
function testEmbedHttpError(int statusCode) {
    ai:Embedding|ai:Error embedding = defaultEmbeddingProvider->embed(
        <ai:TextChunk>{content: STATUS_PREFIX + statusCode.toString()});
    assertHttpError(embedding, statusCode);
}

@test:Config {dataProvider: httpErrorStatusCodes}
function testBatchEmbedHttpError(int statusCode) {
    ai:Embedding[]|ai:Error embeddings = defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: STATUS_PREFIX + statusCode.toString()},
        <ai:TextChunk>{content: "second"}
    ]);
    assertHttpError(embeddings, statusCode);
}

function httpErrorStatusCodes() returns int[][] =>
    [[400], [401], [402], [403], [404], [422], [429], [500], [502], [503], [504]];

@test:Config
function testValidationErrorDetailIsAvailable() returns error? {
    ai:Embedding|ai:Error embedding = defaultEmbeddingProvider->embed(<ai:TextChunk>{content: STATUS_PREFIX + "422"});
    http:ApplicationResponseError cause = getHttpErrorCause(embedding);
    json body = check cause.detail().body.ensureType();
    json[] detail = check (check body.detail).ensureType();
    test:assertEquals(check detail[0].msg, "Input should be less than or equal to 3072");
}

@test:Config
function testRateLimitRetryAfterHeaderIsAvailable() returns error? {
    ai:Embedding|ai:Error embedding = defaultEmbeddingProvider->embed(<ai:TextChunk>{content: STATUS_PREFIX + "429"});
    http:ApplicationResponseError cause = getHttpErrorCause(embedding);
    map<string[]> headers = cause.detail().headers;
    string[]? retryAfter = headers["Retry-After"] ?: headers["retry-after"];
    test:assertEquals(retryAfter, ["1"]);
}

// Malformed and unusual responses

@test:Config {dataProvider: malformedResponses}
function testEmbedMalformedResponse(string kind) {
    ai:Embedding|ai:Error embedding = defaultEmbeddingProvider->embed(<ai:TextChunk>{content: MALFORMED_PREFIX + kind});
    test:assertTrue(embedding is ai:LlmInvalidResponseError, "Expected an invalid response error for " + kind);
}

@test:Config {dataProvider: malformedResponses}
function testBatchEmbedMalformedResponse(string kind) {
    ai:Embedding[]|ai:Error embeddings = defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: MALFORMED_PREFIX + kind},
        <ai:TextChunk>{content: "second"}
    ]);
    test:assertTrue(embeddings is ai:LlmInvalidResponseError, "Expected an invalid response error for " + kind);
}

function malformedResponses() returns string[][] => [
    ["NOT_JSON"],
    ["MISSING_DATA"],
    ["DATA_NOT_ARRAY"],
    ["FEWER_EMBEDDINGS"],
    ["MORE_EMBEDDINGS"],
    ["MISSING_EMBEDDING"],
    ["STRING_EMBEDDING"],
    ["DUPLICATE_INDEX"],
    ["OUT_OF_RANGE_INDEX"],
    ["NEGATIVE_INDEX"]
];

@test:Config
function testMalformedResponseErrorMessages() {
    ai:Embedding|ai:Error notJson = defaultEmbeddingProvider->embed(<ai:TextChunk>{content: MALFORMED_PREFIX + "NOT_JSON"});
    test:assertEquals(notJson is ai:Error ? notJson.message() : "", INVALID_RESPONSE_ERROR);
    ai:Embedding|ai:Error fewer = defaultEmbeddingProvider->embed(
        <ai:TextChunk>{content: MALFORMED_PREFIX + "FEWER_EMBEDDINGS"});
    test:assertEquals(fewer is ai:Error ? fewer.message() : "", "Expected 1 embeddings in the response, but received 0");
    ai:Embedding|ai:Error missing = defaultEmbeddingProvider->embed(
        <ai:TextChunk>{content: MALFORMED_PREFIX + "MISSING_EMBEDDING"});
    test:assertEquals(missing is ai:Error ? missing.message() : "", "Missing embedding at position 0 in the response");
    ai:Embedding|ai:Error outOfRange = defaultEmbeddingProvider->embed(
        <ai:TextChunk>{content: MALFORMED_PREFIX + "OUT_OF_RANGE_INDEX"});
    test:assertEquals(outOfRange is ai:Error ? outOfRange.message() : "", "Invalid embedding index 1 in the response");
}

@test:Config
function testBatchEmbedReordersByIndex() returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: REVERSED_ORDER},
        <ai:TextChunk>{content: "second"},
        <ai:TextChunk>{content: "third"}
    ]);
    test:assertEquals(embeddings.length(), 3);
    foreach int i in 0 ..< 3 {
        assertVector(embeddings[i], 1024, getMockFloatValue(i, 0));
    }
}

@test:Config
function testBatchEmbedWithoutIndexKeepsResponseOrder() returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: MALFORMED_PREFIX + "NO_INDEX"},
        <ai:TextChunk>{content: "second"}
    ]);
    test:assertEquals(embeddings, [<ai:Vector>[0.0], <ai:Vector>[1.0]]);
}

@test:Config
function testResponseWithExtraKeysIsAccepted() returns error? {
    ai:Embedding embedding = check defaultEmbeddingProvider->embed(
        <ai:TextChunk>{content: MALFORMED_PREFIX + "NO_USAGE_EXTRA_KEYS"});
    test:assertEquals(embedding, <ai:Vector>[0.5, -0.5]);
}

@test:Config
function testResponseWithoutUsageIsAccepted() returns error? {
    ai:Embedding[] embeddings = check defaultEmbeddingProvider->batchEmbed([
        <ai:TextChunk>{content: MALFORMED_PREFIX + "NO_USAGE"},
        <ai:TextChunk>{content: "second"}
    ]);
    test:assertEquals(embeddings, [<ai:Vector>[0.5, -0.5], <ai:Vector>[0.5, -0.5]]);
}

// Concurrency

@test:Config
function testConcurrentEmbedCalls() returns error? {
    future<ai:Embedding|ai:Error>[] futures = [];
    foreach int i in 0 ..< 8 {
        future<ai:Embedding|ai:Error> f = start defaultEmbeddingProvider->embed(<ai:TextChunk>{content: i.toString()});
        futures.push(f);
    }
    foreach future<ai:Embedding|ai:Error> f in futures {
        ai:Embedding embedding = check wait f;
        assertVector(embedding, 1024, getMockFloatValue(0, 0));
    }
    test:assertEquals(getEmbeddingRequestCount(), 8);
}

// Helpers

isolated function getExpectedRequest(string modelType, string[] input, int? outputDimension,
        EmbeddingDataType? outputDataType) returns map<json> {
    map<json> request = {model: modelType, input};
    if outputDimension is int {
        request["output_dimension"] = outputDimension;
    }
    if outputDataType is EmbeddingDataType {
        request["output_dtype"] = outputDataType;
    }
    return request;
}

isolated function getExpectedFirstValue(EmbeddingDataType? outputDataType, int index) returns float =>
    outputDataType is () || outputDataType == EMBEDDING_FLOAT ? getMockFloatValue(index, 0) :
    <float>getMockIntValue(index, 0);

isolated function assertVector(ai:Embedding embedding, int expectedLength, float expectedFirstValue) {
    if embedding !is ai:Vector {
        test:assertFail("Expected a dense vector");
    }
    test:assertEquals(embedding.length(), expectedLength);
    test:assertEquals(embedding[0], expectedFirstValue);
}

isolated function assertHttpError(ai:Embedding|ai:Embedding[]|ai:Error result, int expectedStatusCode) {
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected an ai:LlmConnectionError for status ${expectedStatusCode}`);
    }
    test:assertEquals(result.message(), CONNECTION_ERROR);
    error? cause = result.cause();
    if cause !is http:ApplicationResponseError {
        test:assertFail("Expected an http:ApplicationResponseError cause");
    }
    test:assertEquals(cause.detail().statusCode, expectedStatusCode);
}

isolated function getHttpErrorCause(ai:Embedding|ai:Error result) returns http:ApplicationResponseError {
    error? cause = result is ai:Error ? result.cause() : ();
    if cause !is http:ApplicationResponseError {
        test:assertFail("Expected an http:ApplicationResponseError cause");
    }
    return cause;
}
