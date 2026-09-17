#include "kode/training/trainer.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/model/unet.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

void print_usage(const char* prog) {
    std::cout << "KODE Training Pipeline CLI (kode_train)\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --samples <int>         Synthetic procedural samples to generate (default: 200)\n"
              << "  --epochs <int>          Number of training epochs (default: 5)\n"
              << "  --batch-size <int>      Mini-batch size (default: 8)\n"
              << "  --lr <float>            Base learning rate (default: 0.0003)\n"
              << "  --warmup-steps <int>    Linear warmup steps (default: 20)\n"
              << "  --checkpoint-dir <path> Directory for periodic checkpoints (default: checkpoints)\n"
              << "  --output <path>         Final checkpoint destination (default: checkpoints/first_generation.kode)\n"
              << "  --seed <int>            PRNG seed (default: 42)\n"
              << "  --log-every <int>       Log step frequency (default: 5)\n"
              << "  --help, -h              Display this help message\n";
}

int main(int argc, char* argv[]) {
    try {
        int num_samples = 200;
        int epochs = 5;
        int batch_size = 8;
        float lr = 3e-4f;
        uint64_t warmup_steps = 20;
        std::string checkpoint_dir = "checkpoints";
        std::string output_checkpoint = "checkpoints/first_generation.kode";
        uint64_t seed = 42;
        int log_every = 5;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--samples" && i + 1 < argc) {
                num_samples = std::stoi(argv[++i]);
            } else if (arg == "--epochs" && i + 1 < argc) {
                epochs = std::stoi(argv[++i]);
            } else if (arg == "--batch-size" && i + 1 < argc) {
                batch_size = std::stoi(argv[++i]);
            } else if (arg == "--lr" && i + 1 < argc) {
                lr = std::stof(argv[++i]);
            } else if (arg == "--warmup-steps" && i + 1 < argc) {
                warmup_steps = std::stoull(argv[++i]);
            } else if (arg == "--checkpoint-dir" && i + 1 < argc) {
                checkpoint_dir = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_checkpoint = argv[++i];
            } else if (arg == "--seed" && i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else if (arg == "--log-every" && i + 1 < argc) {
                log_every = std::stoi(argv[++i]);
            }
        }

        std::cout << "========================================================\n"
                  << "  KODE Procedural Grounding Training Engine (Phase 9)\n"
                  << "========================================================\n"
                  << " Dataset:       SyntheticGroundingDataset (" << num_samples << " procedural items)\n"
                  << " Epochs:        " << epochs << "\n"
                  << " Batch Size:    " << batch_size << "\n"
                  << " Learning Rate: " << lr << "\n"
                  << " Warmup Steps:  " << warmup_steps << "\n"
                  << " Seed:          " << seed << "\n"
                  << " Output File:   " << output_checkpoint << "\n"
                  << "--------------------------------------------------------\n";

        // 1. Build Tokenizer with grounding vocabulary
        auto vocab = std::make_shared<kode::text::Vocabulary>();
        std::vector<std::string> words = {
            "a", "small", "large", "red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange",
            "circle", "square", "triangle", "cross", "diamond", "in", "the", "center", "top", "left",
            "right", "bottom", "on", "black", "dark", "gray", "light", "navy", "purple", "background"
        };
        for (const auto& w : words) {
            vocab->add_token(w);
        }
        auto tokenizer = std::make_shared<kode::text::Tokenizer>(vocab);

        // 2. Generate procedural dataset
        std::cout << "[INFO] Generating procedural grounding dataset (" << num_samples << " pairs)...\n";
        auto dataset = std::make_shared<kode::dataset::SyntheticGroundingDataset>(
            num_samples, tokenizer, seed, 32, 32
        );
        kode::dataset::DataLoader dataloader(dataset, batch_size, true, false, seed);
        std::cout << "[INFO] Dataset ready: " << dataset->size() << " samples, " 
                  << dataloader.num_batches() << " batches/epoch.\n";

        // 3. Assemble model components
        auto unet_cfg = kode::model::UNetConfig::default_config();
        auto unet = std::make_shared<kode::model::UNet>(unet_cfg);
        auto text_encoder = std::make_shared<kode::text::TextEncoder>(
            unet_cfg.text_vocab_size, unet_cfg.text_embed_dim, unet_cfg.text_max_seq_len
        );

        kode::diffusion::DiffusionConfig diff_cfg;
        diff_cfg.num_timesteps = 1000;
        diff_cfg.schedule_type = kode::diffusion::ScheduleType::Cosine;
        auto diffusion = std::make_shared<kode::diffusion::GaussianDiffusion>(diff_cfg);

        // 4. Configure Trainer
        kode::training::TrainingConfig train_cfg;
        train_cfg.learning_rate = lr;
        train_cfg.batch_size = static_cast<kode::dim_t>(batch_size);
        train_cfg.warmup_steps = warmup_steps;
        train_cfg.total_steps = static_cast<uint64_t>(epochs) * dataloader.num_batches();
        train_cfg.checkpoint_dir = checkpoint_dir;
        train_cfg.checkpoint_every_steps = 0; // Save at end
        train_cfg.log_every_steps = static_cast<uint64_t>(log_every);
        train_cfg.log_csv_path = "checkpoints/first_generation_loss.csv";
        train_cfg.seed = seed;

        kode::training::Trainer trainer(unet, text_encoder, tokenizer, diffusion, train_cfg);

        // 5. Training loop
        std::cout << "[INFO] Commencing training for " << train_cfg.total_steps << " steps...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        float_t last_loss = 0.0f;
        uint64_t total_samples_processed = 0;

        for (int ep = 0; ep < epochs; ++ep) {
            trainer.set_epoch(ep);
            dataloader.reset();

            while (dataloader.has_next()) {
                auto batch = dataloader.next_batch();
                last_loss = trainer.train_step(batch.images, batch.captions);
                total_samples_processed += batch.batch_size;

                if (trainer.step() % log_every == 0 || !dataloader.has_next()) {
                    std::cout << "  Epoch [" << (ep + 1) << "/" << epochs << "] "
                              << "Step " << trainer.step() << "/" << train_cfg.total_steps
                              << " - Loss: " << last_loss
                              << " - LR: " << trainer.optimizer()->lr() << "\n";
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        double throughput = total_samples_processed / (elapsed_sec > 0.0 ? elapsed_sec : 1.0);

        // 6. Save final checkpoint
        trainer.save_checkpoint(output_checkpoint, last_loss);

        std::cout << "========================================================\n"
                  << "[SUCCESS] Training Completed Successfully!\n"
                  << " Final Loss:    " << last_loss << "\n"
                  << " Total Time:    " << elapsed_sec << " s\n"
                  << " Throughput:    " << throughput << " samples/s\n"
                  << " Checkpoint:    " << output_checkpoint << "\n"
                  << "========================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }
}
