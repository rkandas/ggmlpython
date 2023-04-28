#ifndef GPT_BINDINGS_H
#define GPT_BINDINGS_H

#include <string>

extern "C" {
    const char* generate_text(const char* prompt, int seed, int n_predict, int n_threads, int top_k, int top_p, float temp);
    bool load_model(const char* path);
}

#endif // GPT_BINDINGS_H
