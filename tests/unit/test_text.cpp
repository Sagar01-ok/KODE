#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/text/text_encoder.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;
using namespace kode::text;
using namespace kode::autodiff;

void test_vocabulary() {
    KODE_LOG_INFO("Running test_vocabulary...");

    Vocabulary vocab;

    // 1. Initial special tokens verification
    KODE_TEST_ASSERT(vocab.size() == 5);
    KODE_TEST_ASSERT(vocab.token_to_id("[PAD]") == TOKEN_PAD);
    KODE_TEST_ASSERT(vocab.token_to_id("[UNK]") == TOKEN_UNK);
    KODE_TEST_ASSERT(vocab.token_to_id("[BOS]") == TOKEN_BOS);
    KODE_TEST_ASSERT(vocab.token_to_id("[EOS]") == TOKEN_EOS);
    KODE_TEST_ASSERT(vocab.token_to_id("[EMPTY]") == TOKEN_EMPTY);

    KODE_TEST_ASSERT(vocab.id_to_token(TOKEN_PAD) == "[PAD]");
    KODE_TEST_ASSERT(vocab.id_to_token(TOKEN_UNK) == "[UNK]");
    KODE_TEST_ASSERT(vocab.id_to_token(TOKEN_BOS) == "[BOS]");
    KODE_TEST_ASSERT(vocab.id_to_token(TOKEN_EOS) == "[EOS]");
    KODE_TEST_ASSERT(vocab.id_to_token(TOKEN_EMPTY) == "[EMPTY]");

    // 2. Add tokens
    int64_t id_cat = vocab.add_token("cat");
    int64_t id_dog = vocab.add_token("dog");
    KODE_TEST_ASSERT(id_cat == 5);
    KODE_TEST_ASSERT(id_dog == 6);
    KODE_TEST_ASSERT(vocab.size() == 7);

    // Duplicate token returns existing ID
    KODE_TEST_ASSERT(vocab.add_token("cat") == id_cat);
    KODE_TEST_ASSERT(vocab.size() == 7);

    KODE_TEST_ASSERT(vocab.contains("cat"));
    KODE_TEST_ASSERT(vocab.contains("dog"));
    KODE_TEST_ASSERT(!vocab.contains("elephant"));

    // Unknown token resolves to TOKEN_UNK
    KODE_TEST_ASSERT(vocab.token_to_id("elephant") == TOKEN_UNK);

    // 3. Build from corpus
    Vocabulary corpus_vocab;
    std::vector<std::string> corpus = {
        "a red circle and a blue square",
        "a green triangle and a red square",
        "blue circle"
    };
    corpus_vocab.build_from_corpus(corpus, 15);
    KODE_TEST_ASSERT(corpus_vocab.contains("a"));
    KODE_TEST_ASSERT(corpus_vocab.contains("red"));
    KODE_TEST_ASSERT(corpus_vocab.contains("circle"));
    KODE_TEST_ASSERT(corpus_vocab.contains("and"));
    KODE_TEST_ASSERT(corpus_vocab.size() <= 15);

    // 4. Save and Load
    std::string tmp_vocab_path = "test_vocab_temp.txt";
    vocab.save(tmp_vocab_path);

    Vocabulary loaded_vocab;
    loaded_vocab.load(tmp_vocab_path);

    KODE_TEST_ASSERT(loaded_vocab.size() == vocab.size());
    for (int64_t i = 0; i < static_cast<int64_t>(vocab.size()); ++i) {
        KODE_TEST_ASSERT(loaded_vocab.id_to_token(i) == vocab.id_to_token(i));
        KODE_TEST_ASSERT(loaded_vocab.token_to_id(vocab.id_to_token(i)) == i);
    }

    std::filesystem::remove(tmp_vocab_path);
    KODE_LOG_INFO("test_vocabulary PASSED.");
}

void test_tokenizer_normalization_and_splitting() {
    KODE_LOG_INFO("Running test_tokenizer_normalization_and_splitting...");

    // 1. Lowercase conversion
    std::string norm1 = Tokenizer::normalize("Vibrant Red Sports Car");
    KODE_TEST_ASSERT(norm1 == "vibrant red sports car");

    // 2. Punctuation isolation
    std::string norm2 = Tokenizer::normalize("hello, world! How are you?");
    KODE_TEST_ASSERT(norm2 == "hello , world ! how are you ?");

    // 3. Whitespace collapsing
    std::string norm3 = Tokenizer::normalize("  lots   of   spaces \t and \n newlines  ");
    KODE_TEST_ASSERT(norm3 == "lots of spaces and newlines");

    // 4. Empty string
    std::string norm4 = Tokenizer::normalize("");
    KODE_TEST_ASSERT(norm4.empty());

    // 5. Splitting tokens
    std::vector<std::string> tokens = Tokenizer::split_tokens(norm2);
    KODE_TEST_ASSERT(tokens.size() == 8);
    KODE_TEST_ASSERT(tokens[0] == "hello");
    KODE_TEST_ASSERT(tokens[1] == ",");
    KODE_TEST_ASSERT(tokens[2] == "world");
    KODE_TEST_ASSERT(tokens[3] == "!");
    KODE_TEST_ASSERT(tokens[4] == "how");
    KODE_TEST_ASSERT(tokens[5] == "are");
    KODE_TEST_ASSERT(tokens[6] == "you");
    KODE_TEST_ASSERT(tokens[7] == "?");

    KODE_LOG_INFO("test_tokenizer_normalization_and_splitting PASSED.");
}

void test_tokenizer_encode_decode() {
    KODE_LOG_INFO("Running test_tokenizer_encode_decode...");

    auto vocab = std::make_shared<Vocabulary>();
    vocab->add_token("red");
    vocab->add_token("sports");
    vocab->add_token("car");

    Tokenizer tokenizer(vocab);

    // 1. Standard encode
    std::vector<int64_t> ids = tokenizer.encode("red sports car", 8, true);
    KODE_TEST_ASSERT(ids.size() == 8);
    KODE_TEST_ASSERT(ids[0] == TOKEN_BOS);
    KODE_TEST_ASSERT(ids[1] == vocab->token_to_id("red"));
    KODE_TEST_ASSERT(ids[2] == vocab->token_to_id("sports"));
    KODE_TEST_ASSERT(ids[3] == vocab->token_to_id("car"));
    KODE_TEST_ASSERT(ids[4] == TOKEN_EOS);
    KODE_TEST_ASSERT(ids[5] == TOKEN_PAD);
    KODE_TEST_ASSERT(ids[6] == TOKEN_PAD);
    KODE_TEST_ASSERT(ids[7] == TOKEN_PAD);

    // 2. Decode with skip_special = true
    std::string decoded = tokenizer.decode(ids, true);
    KODE_TEST_ASSERT(decoded == "red sports car");

    // 3. Decode with skip_special = false
    std::string decoded_all = tokenizer.decode(ids, false);
    KODE_TEST_ASSERT(decoded_all.rfind("[BOS]", 0) == 0);

    // 4. Empty string encode (CFG null conditioning token)
    std::vector<int64_t> empty_ids = tokenizer.encode("", 8, true);
    KODE_TEST_ASSERT(empty_ids.size() == 8);
    KODE_TEST_ASSERT(empty_ids[0] == TOKEN_EMPTY);
    for (size_t i = 1; i < 8; ++i) {
        KODE_TEST_ASSERT(empty_ids[i] == TOKEN_PAD);
    }

    // Whitespace string also encodes as empty
    std::vector<int64_t> ws_ids = tokenizer.encode("   \t  \n ", 8, true);
    KODE_TEST_ASSERT(ws_ids.size() == 8);
    KODE_TEST_ASSERT(ws_ids[0] == TOKEN_EMPTY);

    // 5. Sequence truncation
    std::vector<int64_t> trunc_ids = tokenizer.encode("red sports car red sports car red sports car", 6, true);
    KODE_TEST_ASSERT(trunc_ids.size() == 6);
    KODE_TEST_ASSERT(trunc_ids[0] == TOKEN_BOS);
    KODE_TEST_ASSERT(trunc_ids[5] == TOKEN_EOS);

    KODE_LOG_INFO("test_tokenizer_encode_decode PASSED.");
}

void test_text_encoder_architecture() {
    KODE_LOG_INFO("Running test_text_encoder_architecture...");

    TextEncoder encoder(1024, 64, 16, "text_enc");

    KODE_TEST_ASSERT(encoder.vocab_size() == 1024);
    KODE_TEST_ASSERT(encoder.embed_dim() == 64);
    KODE_TEST_ASSERT(encoder.max_seq_len() == 16);

    // Calculate parameter count:
    // token_emb: 1024 * 64 = 65,536
    // pos_emb: 16 * 64 = 1,024
    // fc1: (64 * 128) + 128 = 8,320
    // fc2: (128 * 64) + 64 = 8,256
    // ln: 64 + 64 = 128
    // Total = 83,264
    dim_t total_params = 0;
    for (const auto& p : encoder.parameters()) {
        total_params += p->numel();
    }

    KODE_LOG_INFO("TextEncoder Total Parameters: ", total_params);
    KODE_TEST_ASSERT(total_params == 83264);

    auto named_params = encoder.named_parameters();
    KODE_TEST_ASSERT(named_params.size() == 8); // token_emb.w, pos_emb.w, fc1.w, fc1.b, fc2.w, fc2.b, ln.w, ln.b

    KODE_LOG_INFO("test_text_encoder_architecture PASSED.");
}

void test_text_encoder_forward_backward() {
    KODE_LOG_INFO("Running test_text_encoder_forward_backward...");

    auto vocab = std::make_shared<Vocabulary>();
    vocab->add_token("red");
    vocab->add_token("circle");
    vocab->add_token("blue");
    vocab->add_token("square");

    Tokenizer tokenizer(vocab);
    TextEncoder encoder(1024, 64, 16, "text_enc");

    // 1. Single prompt forward
    TextEncoding enc1 = encoder.forward_prompt("red circle", tokenizer);
    KODE_TEST_ASSERT(enc1.sequence_tokens->shape() == Shape({1, 16, 64}));
    KODE_TEST_ASSERT(enc1.pooled_vector->shape() == Shape({1, 64}));

    // Verify values are finite
    const float_t* pool1_data = enc1.pooled_vector->data().data();
    for (dim_t i = 0; i < 64; ++i) {
        KODE_TEST_ASSERT(std::isfinite(pool1_data[i]));
    }

    // 2. Batch prompt forward
    std::vector<std::string> batch_prompts = {
        "red circle",
        "blue square",
        "" // Unconditional null prompt
    };
    TextEncoding batch_enc = encoder.forward_batch(batch_prompts, tokenizer);
    KODE_TEST_ASSERT(batch_enc.sequence_tokens->shape() == Shape({3, 16, 64}));
    KODE_TEST_ASSERT(batch_enc.pooled_vector->shape() == Shape({3, 64}));

    // Verify prompt semantic discrimination: "red circle" pooled vs "blue square" pooled
    const float_t* pool_data = batch_enc.pooled_vector->data().data();
    float_t diff_sum = 0.0f;
    for (dim_t i = 0; i < 64; ++i) {
        diff_sum += std::abs(pool_data[i] - pool_data[64 + i]);
    }
    KODE_TEST_ASSERT(diff_sum > 1e-4f);

    // 3. Autodiff Backward Verification
    encoder.zero_grad();
    Variable loss = autodiff::sum(autodiff::mul(batch_enc.pooled_vector, batch_enc.pooled_vector));
    loss->backward();

    KODE_TEST_ASSERT(!encoder.token_embedding()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!encoder.position_embedding()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!encoder.fc1()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!encoder.fc2()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!encoder.layer_norm()->weight()->grad().is_empty());

    // Verify zero_grad clears gradients
    encoder.zero_grad();
    const float_t* fc1_g = encoder.fc1()->weight()->grad().data();
    for (dim_t i = 0; i < encoder.fc1()->weight()->numel(); ++i) {
        KODE_TEST_ASSERT(fc1_g[i] == 0.0f);
    }

    KODE_LOG_INFO("test_text_encoder_forward_backward PASSED.");
}

int main() {
    KODE_LOG_INFO("==================================================");
    KODE_LOG_INFO("KODE Phase 5 — Text Processing & Conditioning Test Suite");
    KODE_LOG_INFO("==================================================");

    try {
        test_vocabulary();
        test_tokenizer_normalization_and_splitting();
        test_tokenizer_encode_decode();
        test_text_encoder_architecture();
        test_text_encoder_forward_backward();
    } catch (const std::exception& ex) {
        KODE_LOG_ERROR("EXCEPTION THROWN: ", ex.what());
        return 1;
    }

    KODE_LOG_INFO("==================================================");
    KODE_LOG_INFO("ALL PHASE 5 TEXT TESTS PASSED SUCCESSFULLY (5/5)");
    KODE_LOG_INFO("==================================================");
    return 0;
}
