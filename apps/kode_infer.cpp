#include "kode/inference/inference.hpp"
#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

void print_usage(const char* prog) {
    std::cout << "KODE Standalone Text-to-Image CLI (kode_infer)\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --checkpoint <path>   Path to .kode model checkpoint (required)\n"
              << "  --prompt <string>       Text prompt for conditional generation (required)\n"
              << "  --sampler <ddim|ddpm>   Reverse sampler: 'ddim' (default) or 'ddpm'\n"
              << "  --steps <int>           Number of reverse sampling steps (default: 25)\n"
              << "  --guidance <float>      Classifier-Free Guidance strength (default: 5.0)\n"
              << "  --eta <float>           DDIM stochasticity eta (default: 0.0)\n"
              << "  --seed <int>            PRNG seed for latent noise (default: random)\n"
              << "  --output <path>         Output PNG image path (default: output.png)\n"
              << "  --width <int>           Image width in pixels (default: 32)\n"
              << "  --height <int>          Image height in pixels (default: 32)\n"
              << "  --help, -h              Display this help message\n";
}

int main(int argc, char* argv[]) {
    try {
        std::string checkpoint_path = "";
        std::string prompt = "";
        std::string sampler_str = "ddim";
        int steps = 25;
        float guidance = 5.0f;
        float eta = 0.0f;
        uint64_t seed = 0;
        bool seed_provided = false;
        std::string output_path = "output.png";
        int width = 32;
        int height = 32;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--checkpoint" && i + 1 < argc) {
                checkpoint_path = argv[++i];
            } else if (arg == "--prompt" && i + 1 < argc) {
                prompt = argv[++i];
            } else if (arg == "--sampler" && i + 1 < argc) {
                sampler_str = argv[++i];
            } else if (arg == "--steps" && i + 1 < argc) {
                steps = std::stoi(argv[++i]);
            } else if (arg == "--guidance" && i + 1 < argc) {
                guidance = std::stof(argv[++i]);
            } else if (arg == "--eta" && i + 1 < argc) {
                eta = std::stof(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                seed = std::stoull(argv[++i]);
                seed_provided = true;
            } else if (arg == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (arg == "--width" && i + 1 < argc) {
                width = std::stoi(argv[++i]);
            } else if (arg == "--height" && i + 1 < argc) {
                height = std::stoi(argv[++i]);
            }
        }

        if (prompt.empty()) {
            std::cerr << "Error: --prompt is required.\n\n";
            print_usage(argv[0]);
            return 1;
        }

        if (!seed_provided) {
            seed = std::random_device{}();
        }

        kode::inference::SamplingConfig config;
        config.sampler = (sampler_str == "ddpm" || sampler_str == "DDPM") 
            ? kode::inference::SamplerType::DDPM 
            : kode::inference::SamplerType::DDIM;
        config.steps = static_cast<kode::dim_t>(steps);
        config.guidance_scale = guidance;
        config.eta = eta;
        config.seed = seed;
        config.width = static_cast<kode::dim_t>(width);
        config.height = static_cast<kode::dim_t>(height);
        config.channels = 3;

        std::cout << "========================================================\n"
                  << "  KODE Text-to-Image Generation Engine (Phase 9)\n"
                  << "========================================================\n"
                  << " Prompt:       \"" << prompt << "\"\n"
                  << " Sampler:      " << (config.sampler == kode::inference::SamplerType::DDIM ? "DDIM" : "DDPM") << "\n"
                  << " Steps:        " << config.steps << "\n"
                  << " CFG Guidance: " << config.guidance_scale << "\n"
                  << " Seed:         " << config.seed << "\n"
                  << " Resolution:   " << config.width << "x" << config.height << " (3 channels)\n"
                  << " Output File:  " << output_path << "\n";

        if (!checkpoint_path.empty()) {
            std::cout << " Checkpoint:   " << checkpoint_path << "\n";
        }
        std::cout << "--------------------------------------------------------\n";

        std::cout << "[INFO] Initializing pipeline architecture...\n";
        auto pipeline = kode::inference::DiffusionPipeline::create_default();

        if (!checkpoint_path.empty()) {
            std::cout << "[INFO] Loading model weights from checkpoint...\n";
            auto meta = pipeline->load_checkpoint(checkpoint_path);
            std::cout << "[INFO] Checkpoint loaded (Step: " << meta.step 
                      << ", Epoch: " << meta.epoch 
                      << ", Loss: " << meta.loss << ")\n";
        } else {
            std::cout << "[WARN] No checkpoint provided; generating with default initialized weights.\n";
        }

        std::cout << "[INFO] Generating image...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = pipeline->generate_and_save(prompt, output_path, config);

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (success) {
            std::cout << "[SUCCESS] Image generated and saved to: " << output_path << "\n"
                      << " Generation Latency: " << elapsed_ms << " ms (" 
                      << (elapsed_ms / 1000.0) << " s)\n"
                      << "========================================================\n";
            return 0;
        } else {
            std::cerr << "[ERROR] Failed to save output image to: " << output_path << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }
}
