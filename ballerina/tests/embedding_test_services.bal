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

import ballerina/http;
import ballerina/lang.runtime;

const EMBEDDING_SERVICE_PORT = 8095;
const EMBEDDING_SERVICE_URL = "http://localhost:8095/mistral";

const STATUS_PREFIX = "STATUS_";
const MALFORMED_PREFIX = "MALFORMED_";
const DELAY_PREFIX = "DELAY_";
const FLAKY_PREFIX = "FLAKY_";
const REVERSED_ORDER = "REVERSED_ORDER";

isolated json lastEmbeddingRequest = ();
isolated map<string> lastRequestHeaders = {};
isolated int embeddingRequestCount = 0;
isolated int flakyFailureCount = 0;

isolated function getLastEmbeddingRequest() returns json {
    lock {
        return lastEmbeddingRequest.clone();
    }
}

isolated function getLastRequestHeader(string name) returns string? {
    lock {
        return lastRequestHeaders[name];
    }
}

isolated function getEmbeddingRequestCount() returns int {
    lock {
        return embeddingRequestCount;
    }
}

isolated function resetEmbeddingRequests() {
    lock {
        lastEmbeddingRequest = ();
    }
    lock {
        lastRequestHeaders = {};
    }
    lock {
        embeddingRequestCount = 0;
    }
    lock {
        flakyFailureCount = 0;
    }
}

service /mistral on new http:Listener(EMBEDDING_SERVICE_PORT) {
    resource function post embeddings(http:Request req) returns http:Response|error {
        json payload = check req.getJsonPayload();
        map<string> headers = {};
        foreach string name in ["Authorization", "Content-Type", "Accept"] {
            string|http:HeaderNotFoundError value = req.getHeader(name);
            if value is string {
                headers[name] = value;
            }
        }
        lock {
            lastEmbeddingRequest = payload.clone();
        }
        lock {
            lastRequestHeaders = headers.clone();
        }
        lock {
            embeddingRequestCount += 1;
        }

        string[] inputs = check (check payload.input).cloneWithType();
        string firstInput = inputs.length() > 0 ? inputs[0] : "";
        http:Response res = new;

        if firstInput.startsWith(STATUS_PREFIX) {
            int statusCode = check int:fromString(firstInput.substring(STATUS_PREFIX.length(),
                    STATUS_PREFIX.length() + 3));
            res.statusCode = statusCode;
            if statusCode == 422 {
                res.setJsonPayload({
                    detail: [
                        {
                            'type: "less_than_equal",
                            loc: ["body", "output_dimension"],
                            msg: "Input should be less than or equal to 3072",
                            input: 4096,
                            ctx: {le: 3072}
                        }
                    ]
                });
            } else {
                if statusCode == 429 {
                    res.setHeader("Retry-After", "1");
                }
                res.setJsonPayload({
                    'object: "error",
                    message: string `Mock error with status ${statusCode}`,
                    'type: "invalid_request_error",
                    param: (),
                    code: statusCode.toString()
                });
            }
            return res;
        }

        if firstInput.startsWith(MALFORMED_PREFIX) {
            json|string malformed = getMalformedResponse(firstInput.substring(MALFORMED_PREFIX.length()), inputs);
            if malformed is string {
                res.setTextPayload(malformed, "application/json");
            } else {
                res.setJsonPayload(malformed);
            }
            return res;
        }

        if firstInput.startsWith(DELAY_PREFIX) {
            runtime:sleep(check decimal:fromString(firstInput.substring(DELAY_PREFIX.length())));
        }

        if firstInput.startsWith(FLAKY_PREFIX) {
            boolean shouldFail;
            lock {
                shouldFail = flakyFailureCount < 2;
                if shouldFail {
                    flakyFailureCount += 1;
                }
            }
            if shouldFail {
                res.statusCode = 503;
                res.setJsonPayload({'object: "error", message: "Service unavailable"});
                return res;
            }
        }

        res.setJsonPayload(getMockEmbeddingResponse(payload, inputs, firstInput == REVERSED_ORDER));
        return res;
    }
}

isolated function getMockEmbeddingResponse(json payload, string[] inputs, boolean reversed) returns json {
    string model = (payload.model is string) ? <string>checkpanic payload.model : "";
    json|error dimension = payload.output_dimension;
    json|error dataType = payload.output_dtype;
    int length = getExpectedEmbeddingLength(model, dimension is int ? dimension : ());
    boolean isFloat = dataType !is string || dataType == "float";

    json[] data = [];
    foreach int i in 0 ..< inputs.length() {
        int index = reversed ? inputs.length() - 1 - i : i;
        json[] embedding = [];
        foreach int j in 0 ..< length {
            embedding.push(isFloat ? getMockFloatValue(index, j) : getMockIntValue(index, j));
        }
        data.push({'object: "embedding", embedding, index});
    }
    return {
        id: "embd-mock",
        'object: "list",
        model,
        data,
        usage: {prompt_tokens: inputs.length() * 3, completion_tokens: 0, total_tokens: inputs.length() * 3}
    };
}

isolated function getMalformedResponse(string kind, string[] inputs) returns json|string {
    match kind {
        "NOT_JSON" => {
            return "this is not json";
        }
        "MISSING_DATA" => {
            return {id: "embd-mock", 'object: "list", model: "mistral-embed"};
        }
        "DATA_NOT_ARRAY" => {
            return {data: "invalid"};
        }
        "FEWER_EMBEDDINGS" => {
            return {data: []};
        }
        "MORE_EMBEDDINGS" => {
            json[] data = from int i in 0 ... inputs.length()
                select {embedding: [0.1], index: i};
            return {data};
        }
        "MISSING_EMBEDDING" => {
            return {data: from int i in 0 ..< inputs.length()
                    select {index: i}};
        }
        "STRING_EMBEDDING" => {
            return {data: from int i in 0 ..< inputs.length()
                    select {embedding: "AAAAAA==", index: i}};
        }
        "DUPLICATE_INDEX" => {
            int count = int:max(inputs.length(), 2);
            return {data: from int i in 0 ..< count
                    select {embedding: [0.1], index: 0}};
        }
        "OUT_OF_RANGE_INDEX" => {
            return {data: from int i in 0 ..< inputs.length()
                    select {embedding: [0.1], index: i + 1}};
        }
        "NEGATIVE_INDEX" => {
            return {data: from int i in 0 ..< inputs.length()
                    select {embedding: [0.1], index: -1}};
        }
        "NO_INDEX" => {
            return {data: from int i in 0 ..< inputs.length()
                    select {embedding: [<float>i]}};
        }
        "NO_USAGE_EXTRA_KEYS" => {
            return {
                id: "embd-mock",
                data: from int i in 0 ..< inputs.length()
                    select {embedding: [0.5, -0.5], index: i, 'object: "embedding", extra: true},
                usage: {prompt_tokens: 3, request_count: 1, prompt_token_details: {cached_tokens: 0}},
                unknown: "ignored"
            };
        }
        "NO_USAGE" => {
            return {data: from int i in 0 ..< inputs.length()
                    select {embedding: [0.5, -0.5], index: i}};
        }
        "INT8_VALUES" => {
            return {data: [{embedding: [-128, 0, 127], index: 0}]};
        }
        "EXPONENT_VALUES" => {
            return "{\"data\": [{\"embedding\": [1.5e-3, -2E2, 0.0, 3], \"index\": 0}]}";
        }
    }
    return {};
}

isolated function getExpectedEmbeddingLength(string model, int? dimension) returns int =>
    dimension ?: (model.startsWith("codestral") ? 1536 : 1024);

isolated function getMockFloatValue(int index, int position) returns float =>
    <float>index + <float>(position % 100) / 100.0;

isolated function getMockIntValue(int index, int position) returns int => (index + position) % 128;
