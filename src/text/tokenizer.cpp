#include "kode/text/tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <unordered_set>

namespace kode::text {

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------
Vocabulary::Vocabulary() {
    add_token("[PAD]");   // ID 0
    add_token("[UNK]");   // ID 1
    add_token("[BOS]");   // ID 2
    add_token("[EOS]");   // ID 3
    add_token("[EMPTY]"); // ID 4 (Classifier-Free Guidance null condition)
}

int64_t Vocabulary::add_token(const std::string& token) {
    auto it = token2id_.find(token);
    if (it != token2id_.end()) {
        return it->second;
    }
    int64_t new_id = static_cast<int64_t>(id2token_.size());
    token2id_[token] = new_id;
    id2token_.push_back(token);
    return new_id;
}

int64_t Vocabulary::token_to_id(const std::string& token) const {
    auto it = token2id_.find(token);
    if (it != token2id_.end()) {
        return it->second;
    }
    return TOKEN_UNK;
}

const std::string& Vocabulary::id_to_token(int64_t id) const {
    if (id >= 0 && id < static_cast<int64_t>(id2token_.size())) {
        return id2token_[id];
    }
    static const std::string unk = "[UNK]";
    return unk;
}

bool Vocabulary::contains(const std::string& token) const {
    return token2id_.find(token) != token2id_.end();
}

void Vocabulary::build_from_corpus(const std::vector<std::string>& texts, size_t max_vocab_size) {
    std::unordered_map<std::string, size_t> counts;
    for (const auto& text : texts) {
        std::string norm = Tokenizer::normalize(text);
        std::vector<std::string> tokens = Tokenizer::split_tokens(norm);
        for (const auto& tok : tokens) {
            if (!tok.empty()) {
                counts[tok]++;
            }
        }
    }

    // Sort tokens by frequency descending
    std::vector<std::pair<std::string, size_t>> sorted_tokens(counts.begin(), counts.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    for (const auto& [token, count] : sorted_tokens) {
        if (size() >= max_vocab_size) break;
        add_token(token);
    }
}

void Vocabulary::save(const std::string& filepath) const {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file for writing: " + filepath);
    }
    for (const auto& tok : id2token_) {
        ofs << tok << "\n";
    }
}

void Vocabulary::load(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file for reading: " + filepath);
    }
    token2id_.clear();
    id2token_.clear();

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            add_token(line);
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
Tokenizer::Tokenizer(std::shared_ptr<Vocabulary> vocab)
    : vocab_(vocab ? std::move(vocab) : std::make_shared<Vocabulary>()) {}

std::string Tokenizer::normalize(const std::string& text) {
    std::string res;
    res.reserve(text.size() * 2);

    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::ispunct(uc)) {
            res.push_back(' ');
            res.push_back(static_cast<char>(std::tolower(uc)));
            res.push_back(' ');
        } else if (std::isspace(uc)) {
            res.push_back(' ');
        } else {
            res.push_back(static_cast<char>(std::tolower(uc)));
        }
    }

    // Collapse spaces
    std::string collapsed;
    bool last_was_space = true;
    for (char c : res) {
        if (c == ' ') {
            if (!last_was_space) {
                collapsed.push_back(' ');
                last_was_space = true;
            }
        } else {
            collapsed.push_back(c);
            last_was_space = false;
        }
    }
    if (!collapsed.empty() && collapsed.back() == ' ') {
        collapsed.pop_back();
    }
    return collapsed;
}

std::vector<std::string> Tokenizer::split_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

std::vector<int64_t> Tokenizer::encode(const std::string& text, size_t max_seq_len, bool add_special) const {
    std::vector<int64_t> ids;
    ids.reserve(max_seq_len);

    std::string norm = normalize(text);
    if (norm.empty()) {
        // Special empty prompt encoding for CFG
        ids.push_back(TOKEN_EMPTY);
        while (ids.size() < max_seq_len) {
            ids.push_back(TOKEN_PAD);
        }
        return ids;
    }

    if (add_special) {
        ids.push_back(TOKEN_BOS);
    }

    std::vector<std::string> tokens = split_tokens(norm);

    size_t content_limit = max_seq_len - (add_special ? 2 : 0);
    for (size_t i = 0; i < tokens.size() && ids.size() < content_limit + (add_special ? 1 : 0); ++i) {
        ids.push_back(vocab_->token_to_id(tokens[i]));
    }

    if (add_special && ids.size() < max_seq_len) {
        ids.push_back(TOKEN_EOS);
    }

    // Pad remaining positions with TOKEN_PAD
    while (ids.size() < max_seq_len) {
        ids.push_back(TOKEN_PAD);
    }

    if (ids.size() > max_seq_len) {
        ids.resize(max_seq_len);
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids, bool skip_special) const {
    std::ostringstream oss;
    bool first = true;

    for (int64_t id : ids) {
        if (skip_special && (id == TOKEN_PAD || id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_EMPTY)) {
            continue;
        }
        if (!first) {
            oss << " ";
        }
        oss << vocab_->id_to_token(id);
        first = false;
    }
    return oss.str();
}

} // namespace kode::text
