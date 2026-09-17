#pragma once

#include "kode/core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace kode::text {

// Fixed Special Token IDs
constexpr int64_t TOKEN_PAD = 0;
constexpr int64_t TOKEN_UNK = 1;
constexpr int64_t TOKEN_BOS = 2;
constexpr int64_t TOKEN_EOS = 3;
constexpr int64_t TOKEN_EMPTY = 4; // Null conditioning token for Classifier-Free Guidance

class Vocabulary {
public:
    Vocabulary();

    int64_t add_token(const std::string& token);
    int64_t token_to_id(const std::string& token) const;
    const std::string& id_to_token(int64_t id) const;

    bool contains(const std::string& token) const;
    size_t size() const noexcept { return token2id_.size(); }

    void build_from_corpus(const std::vector<std::string>& texts, size_t max_vocab_size = 1024);

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    std::unordered_map<std::string, int64_t> token2id_;
    std::vector<std::string> id2token_;
};

class Tokenizer {
public:
    explicit Tokenizer(std::shared_ptr<Vocabulary> vocab = nullptr);

    // Text Normalization: lowercase, isolate punctuation, collapse whitespaces
    static std::string normalize(const std::string& text);

    // Splits normalized text into individual token strings
    static std::vector<std::string> split_tokens(const std::string& text);

    // Converts text into padded/truncated token IDs
    std::vector<int64_t> encode(const std::string& text, size_t max_seq_len = 16, bool add_special = true) const;

    // Decodes token IDs back to a reconstructed string
    std::string decode(const std::vector<int64_t>& ids, bool skip_special = true) const;

    std::shared_ptr<Vocabulary> vocabulary() const noexcept { return vocab_; }
    void set_vocabulary(std::shared_ptr<Vocabulary> vocab) noexcept { vocab_ = std::move(vocab); }

private:
    std::shared_ptr<Vocabulary> vocab_;
};

} // namespace kode::text
