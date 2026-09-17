#include "kode/inference/inference.hpp"
#include "kode/evaluation/evaluation.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

void print_usage(const char* prog) {
    std::cout << "KODE Generative Evaluation CLI (kode_eval)\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --checkpoint <path>   Path to .kode model checkpoint\n"
              << "  --samples <int>       Number of benchmark evaluation samples (default: 16)\n"
              << "  --sampler <ddim|ddpm> Reverse sampler: 'ddim' (default) or 'ddpm'\n"
              << "  --steps <int>         Number of reverse sampling steps (default: 25)\n"
              << "  --guidance <float>    Classifier-Free Guidance scale (default: 5.0)\n"
              << "  --seed <int>          PRNG seed (default: 42)\n"
              << "  --output-dir <path>   Directory to save generated eval images (default: eval_output)\n"
              << "  --report <path>       Destination for JSON evaluation report (default: eval_report.json)\n"
              << "  --eval-val-loss       Compute held-out validation loss using DataLoader\n"
              << "  --val-samples <int>   Number of validation dataset samples (default: 50)\n"
              << "  --help, -h            Display this help message\n";
}

int main(int argc, char* argv[]) {
    try {
        std::string checkpoint_path = "";
        int num_samples = 16;
        std::string sampler_str = "ddim";
        int steps = 25;
        float guidance = 5.0f;
        uint64_t seed = 42;
        std::string output_dir = "eval_output";
        std::string report_path = "eval_report.json";
        bool compute_val_loss = false;
        int val_samples = 50;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--checkpoint" && i + 1 < argc) {
                checkpoint_path = argv[++i];
            } else if (arg == "--samples" && i + 1 < argc) {
                num_samples = std::stoi(argv[++i]);
            } else if (arg == "--sampler" && i + 1 < argc) {
                sampler_str = argv[++i];
            } else if (arg == "--steps" && i + 1 < argc) {
                steps = std::stoi(argv[++i]);
            } else if (arg == "--guidance" && i + 1 < argc) {
                guidance = std::stof(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else if (arg == "--output-dir" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--report" && i + 1 < argc) {
                report_path = argv[++i];
            } else if (arg == "--eval-val-loss") {
                compute_val_loss = true;
            } else if (arg == "--val-samples" && i + 1 < argc) {
                val_samples = std::stoi(argv[++i]);
            }
        }

        std::cout << "========================================================\n"
                  << "  KODE Automated Model Evaluation Engine (Phase 10)\n"
                  << "========================================================\n"
                  << " Checkpoint:     " << (checkpoint_path.empty() ? "(none - random init)" : checkpoint_path) << "\n"
                  << " Samples:        " << num_samples << "\n"
                  << " Sampler:        " << sampler_str << " (" << steps << " steps)\n"
                  << " CFG Guidance:   " << guidance << "\n"
                  << " Seed:           " << seed << "\n"
                  << " Output Dir:     " << output_dir << "\n"
                  << " Report File:    " << report_path << "\n"
                  << " Eval Val Loss:  " << (compute_val_loss ? "Enabled" : "Disabled") << "\n"
                  << "--------------------------------------------------------\n";

        // Create output directory
        std::filesystem::create_directories(output_dir);

        // Initialize pipeline
        std::cout << "[INFO] Initializing diffusion pipeline...\n";
        auto pipeline = kode::inference::DiffusionPipeline::create_default();

        if (!checkpoint_path.empty()) {
            std::cout << "[INFO] Loading weights from: " << checkpoint_path << " ...\n";
            auto meta = pipeline->load_checkpoint(checkpoint_path);
            std::cout << "[INFO] Checkpoint loaded (Step: " << meta.step 
                      << ", Epoch: " << meta.epoch 
                      << ", Loss: " << meta.loss << ")\n";
        }

        kode::inference::SamplingConfig config;
        config.sampler = (sampler_str == "ddpm" || sampler_str == "DDPM") 
            ? kode::inference::SamplerType::DDPM 
            : kode::inference::SamplerType::DDIM;
        config.steps = static_cast<kode::dim_t>(steps);
        config.guidance_scale = guidance;
        config.seed = seed;
        config.width = 32;
        config.height = 32;
        config.channels = 3;

        // Construct standardized procedural grounding dataset for reference evaluation
        std::cout << "[INFO] Synthesizing procedural test items for evaluation...\n";
        auto vocab = std::make_shared<kode::text::Vocabulary>();
        std::vector<std::string> words = {
            "a", "small", "large", "red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange",
            "circle", "square", "triangle", "cross", "diamond", "in", "the", "center", "top", "left",
            "right", "bottom", "on", "black", "dark", "gray", "light", "navy", "purple", "background"
        };
        for (const auto& w : words) vocab->add_token(w);
        auto tokenizer = std::make_shared<kode::text::Tokenizer>(vocab);

        auto eval_dataset = std::make_shared<kode::dataset::SyntheticGroundingDataset>(
            static_cast<size_t>(num_samples), tokenizer, seed + 1000, 32, 32
        );

        kode::evaluation::GroundingEvaluator evaluator;

        // Optionally compute held-out validation loss
        float val_loss = 0.0f;
        if (compute_val_loss) {
            std::cout << "[INFO] Computing held-out validation loss on " << val_samples << " procedural samples...\n";
            auto val_dataset = std::make_shared<kode::dataset::SyntheticGroundingDataset>(
                static_cast<size_t>(val_samples), tokenizer, seed + 5000, 32, 32
            );
            kode::dataset::DataLoader val_loader(val_dataset, 8, false, false, seed + 999);
            val_loss = evaluator.evaluate_validation_loss(
                *pipeline->unet(),
                *pipeline->text_encoder(),
                *pipeline->tokenizer(),
                *pipeline->diffusion(),
                val_loader,
                10
            );
            std::cout << "[INFO] Validation MSE Loss: " << val_loss << "\n";
        }

        // Run full evaluation with dataset reference pairs
        std::cout << "[INFO] Generating " << num_samples << " images and evaluating grounding metrics...\n";
        auto eval_t0 = std::chrono::high_resolution_clock::now();

        auto report = evaluator.evaluate_pipeline_with_dataset(
            *pipeline,
            *eval_dataset,
            static_cast<size_t>(num_samples),
            config
        );

        auto eval_t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(eval_t1 - eval_t0).count();

        // Save generated images
        for (size_t i = 0; i < report.sample_results.size(); ++i) {
            const auto& res = report.sample_results[i];
            char fname[128];
            std::snprintf(fname, sizeof(fname), "eval_%03zu_%s_%s.png", 
                          i + 1, res.expected_color.c_str(), res.expected_shape.c_str());
            std::string out_img_path = output_dir + "/" + fname;
            if (i < report.generated_images.size()) {
                kode::image::save_image_png(out_img_path, report.generated_images[i]);
            }
        }

        // Output summary report
        std::cout << "\n" << report.to_summary_string();
        if (compute_val_loss) {
            std::cout << " Validation MSE Loss:   " << val_loss << "\n";
            std::cout << "========================================================\n";
        }
        std::cout << " Total Evaluation Time: " << total_ms << " ms (" << (total_ms / 1000.0) << " s)\n"
                  << " Per-Sample Latency:    " << (total_ms / num_samples) << " ms\n\n";

        // Save JSON report
        std::ofstream ofs(report_path);
        if (ofs.is_open()) {
            ofs << report.to_json_string();
            std::cout << "[SUCCESS] JSON evaluation report written to: " << report_path << "\n";
        } else {
            std::cerr << "[WARN] Could not write report to: " << report_path << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }
}
