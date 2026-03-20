#include "api_utils.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>

namespace api_utils {

// Callback function for handling curl response
size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

/*** 
 * Curl get embeddings for a batch of queries from OpenAI API 
 * @param queries: Vector of input text queries
 * @param model: Model name for text embedding
 * @param api_key: OpenAI API key
 * @param dim: Dimension of the embedding
 * @param all_embeddings: Pre-allocated pointer to a float array to store all embeddings
 * @return: True if all embeddings were retrieved successfully, false otherwise
 */
bool get_batch_embeddings(const std::vector<std::string> &queries, 
                          const std::string &model, 
                          const std::string &api_key,
                          const int &dim,
                          float* all_embeddings) {
    CURL *curl;
    CURLcode res;
    bool success = true;

    curl = curl_easy_init();
    if (curl) {
        std::string url = "https://api.openai.com/v1/embeddings";
        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");

        try {
            nlohmann::json payload = {
                {"model", model},
                {"input", queries},
                {"dimensions", dim}  // Explicitly including dimension if needed
            };

            std::string postData = payload.dump();
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);

            std::string readBuffer;
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

            res = curl_easy_perform(curl);
            if (res != CURLE_OK) {
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
                success = false;
            } else {
                auto response_json = nlohmann::json::parse(readBuffer, nullptr, false);
                if (response_json.is_discarded() || !response_json.contains("data")) {
                    throw std::runtime_error("JSON parsing error or missing data field in response.");
                }
                int index = 0;
                for (const auto &item : response_json["data"]) {
                    auto embedding = item["embedding"];
                    if (embedding.size() != dim) {
                        std::cerr << "Expected dimension: " << dim << ", Actual dimension: " << embedding.size() << std::endl;
                        throw std::runtime_error("Embedding dimension mismatch.");
                    }

                    for (int j = 0; j < dim; ++j) {
                        all_embeddings[index * dim + j] = embedding[j].get<float>();
                    }
                    index++;
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            success = false;
        }

        // Cleanup
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    } else {
        success = false;
    }

    return success;
}

// Function to run GPT model with query and top documents
std::string run_gpt4o_mini(const std::string &query, const std::vector<std::string> &top_docs, const std::string &model, const std::string &api_key) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        // Combine the query and top documents into a single conversation-like input
        std::string prompt = query + "\nTop documents:\n";
        for (const auto& doc : top_docs) {
            prompt += "- " + doc + "\n";
        }

        // Use the chat/completions endpoint with properly formatted messages
        nlohmann::json payload = {
            {"model", model},
            {"messages", {
                {{"role", "system"}, {"content", "You are a helpful assistant."}},
                {{"role", "user"}, {"content", prompt}}
            }}
        };

        std::string url = "https://api.openai.com/v1/chat/completions";
        std::string postData = payload.dump();

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    // Parse the response to extract only the assistant's message content
    try {
        auto json_response = nlohmann::json::parse(readBuffer);
        if (json_response.contains("error")) {
            std::cerr << "API error: " << json_response["error"]["message"] << std::endl;
            return "Error: " + json_response["error"]["message"].get<std::string>();
        }
        std::string answer = json_response["choices"][0]["message"]["content"];
        return answer;
    } catch (const std::exception &e) {
        std::cerr << "Error parsing JSON response: " << e.what() << std::endl;
        return "Error: Unable to parse response.";
    }
}


}  // namespace api_utils
