#include "kode/inference/inference.hpp"
#include "kode/training/trainer.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/model/unet.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/evaluation/evaluation.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace kode;

int main() {
    try {
        std::cout << "========================================================\n"
                  << "  KODE Phase 10 Official Benchmark Runner\n"
                  << "========================================================\n\n";

        // Initialize pipeline
        auto pipeline = inference::DiffusionPipeline::create_default();
        std::string prompt = "a small red circle on a black background";

        // -----------------------------------------------------------------
        // BENCH-INF-01: Single Image DDIM-25 (32x32) Latency
        // -----------------------------------------------------------------
        std::cout << "[BENCHMARK] Running BENCH-INF-01 (DDIM-25, 32x32)...\n";
        inference::SamplingConfig ddim_cfg;
        ddim_cfg.sampler = inference::SamplerType::DDIM;
        ddim_cfg.steps = 25;
        ddim_cfg.guidance_scale = 5.0f;
        ddim_cfg.seed = 42;

        // Warmup (1 pass)
        pipeline->generate(prompt, ddim_cfg);

        // Benchmark (3 passes for reliable average)
        std::vector<double> ddim_times_ms;
        for (int i = 0; i < 3; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            pipeline->generate(prompt, ddim_cfg);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ddim_times_ms.push_back(ms);
            std::cout << "  Pass " << (i + 1) << ": " << ms << " ms\n";
        }
        double ddim_mean = std::accumulate(ddim_times_ms.begin(), ddim_times_ms.end(), 0.0) / ddim_times_ms.size();
        std::cout << "  => BENCH-INF-01 Mean Latency: " << ddim_mean << " ms\n\n";

        // -----------------------------------------------------------------
        // BENCH-TRAIN-01: Training Throughput (Batch Size 16)
        // -----------------------------------------------------------------
        std::cout << "[BENCHMARK] Running BENCH-TRAIN-01 (Training Throughput B=16)...\n";
        auto unet = std::make_shared<model::UNet>();
        auto text_enc = std::make_shared<text::TextEncoder>();
        auto tokenizer = std::make_shared<text::Tokenizer>();
        auto diff = std::make_shared<diffusion::GaussianDiffusion>();
        training::TrainingConfig tr_cfg;
        tr_cfg.batch_size = 16;
        tr_cfg.checkpoint_every_steps = 0;
        tr_cfg.log_every_steps = 0;
        training::Trainer trainer(unet, text_enc, tokenizer, diff, tr_cfg);

        tensor::Tensor batch_imgs = tensor::Tensor::randn({16, 3, 32, 32}, 0.0f, 1.0f);
        std::vector<std::string> captions(16, "a small red circle on a black background");

        // Warmup (1 step)
        trainer.train_step(batch_imgs, captions);

        // Measure 3 training steps
        auto train_t0 = std::chrono::high_resolution_clock::now();
        int train_steps_count = 3;
        for (int i = 0; i < train_steps_count; ++i) {
            trainer.train_step(batch_imgs, captions);
        }
        auto train_t1 = std::chrono::high_resolution_clock::now();
        double train_sec = std::chrono::duration<double>(train_t1 - train_t0).count();
        double throughput = (train_steps_count * 16) / train_sec;
        std::cout << "  => BENCH-TRAIN-01 Throughput: " << throughput << " samples/s\n\n";

        // -----------------------------------------------------------------
        // BENCH-EVAL-01: Metric Evaluation Latency (Grounding + PSNR + SSIM)
        // -----------------------------------------------------------------
        std::cout << "[BENCHMARK] Running BENCH-EVAL-01 (Evaluation Metric Latency)...\n";
        evaluation::GroundingEvaluator evaluator;
        tensor::Tensor test_img = tensor::Tensor::randn({3, 32, 32}, 0.0f, 1.0f);
        tensor::Tensor ref_img = tensor::Tensor::randn({3, 32, 32}, 0.0f, 1.0f);
        std::string test_prompt = "a small red circle in the center on a black background";

        // Warmup (10 passes)
        for (int i = 0; i < 10; ++i) {
            evaluator.evaluate_sample_with_reference(test_img, ref_img, test_prompt);
        }

        // Measure 1000 passes
        auto eval_t0 = std::chrono::high_resolution_clock::now();
        int eval_passes = 1000;
        for (int i = 0; i < eval_passes; ++i) {
            evaluator.evaluate_sample_with_reference(test_img, ref_img, test_prompt);
        }
        auto eval_t1 = std::chrono::high_resolution_clock::now();
        double eval_total_us = std::chrono::duration<double, std::micro>(eval_t1 - eval_t0).count();
        double eval_per_sample_us = eval_total_us / static_cast<double>(eval_passes);
        std::cout << "  => BENCH-EVAL-01 Mean Metric Latency: " << eval_per_sample_us << " us (" << (eval_per_sample_us / 1000.0) << " ms)\n\n";

        std::cout << "========================================================\n"
                  << "SUMMARY OF MEASUREMENTS:\n"
                  << "  BENCH-INF-01 (DDIM-25):     " << ddim_mean << " ms\n"
                  << "  BENCH-TRAIN-01 (Throughput): " << throughput << " samples/s\n"
                  << "  BENCH-EVAL-01 (Eval Metric): " << eval_per_sample_us << " us (" << (eval_per_sample_us / 1000.0) << " ms)\n"
                  << "========================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
