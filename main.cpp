#include "llama.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <iostream>


int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    llama_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        if (!llama_model_load(params.model, model, vocab, params.n_ctx)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "\n\n");


    httplib::Server svr;

    svr.Get("/", [](const httplib::Request &request, httplib::Response &res) {
        std::ifstream f("../html/index.html");
        std::string html;
        if(f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            html = ss.str();
        }

        res.status = 200;
        res.set_content(html, "text/html");
    });

    svr.Post("/completions", [&params, &vocab, &model, &rng](const httplib::Request &request, httplib::Response &res) {
        nlohmann::json request_json = nullptr;
        try {
            request_json = nlohmann::json::parse(request.body);
        } catch(...) {
            std::string response_string = R"~~~(
                {
                    "error": {
                        "message": "We could not parse the JSON body of your request.",
                        "type": "invalid_request_error",
                        "param": null,
                        "code": null
                    }
                }
            )~~~";
            res.status = 200;
            res.set_content(response_string, "application/json");
            return;
        }

        std::string input_prompt = request_json.at("prompt").get<std::string>();
        std::cout << "Prompt: " << input_prompt << std::endl;
        params.prompt = input_prompt;
        std::string output_text = "";
        if(!do_eval(params, vocab, model, rng, &output_text)) {
            ggml_free(model.ctx);

            std::string response_string = R"~~~(
                {
                    "error": {
                        "message": "Error evaluating results.",
                        "type": "eval_error",
                        "param": null,
                        "code": null
                    }
                }
            )~~~";
            res.status = 200;
            res.set_content(response_string, "application/json");
            return;
        }

        // Let response structure be ChatGPT-like
        std::string created = std::to_string(std::time(nullptr));
        std::string response_id = random_string(24);
        std::string escaped_output = escape_json(output_text);
        std::string response_string = R"({"id": ")" + response_id + R"(", "object": "text_completion", "created": )" + created + R"(, "choices": [ {"text": ")" + escaped_output + R"(", "index": 0 } ]})";

        res.status = 200;
        res.set_content(response_string, "text/plain");
    });

    std::cout << "Alpaca Server is listening on port " << std::to_string(params.port) << std::endl;
    svr.listen("0.0.0.0", params.port);

    ggml_free(model.ctx);

    return 0;
}
