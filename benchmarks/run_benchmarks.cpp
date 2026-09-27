#include "kode/inference/inference.hpp"
#include "kode/training/trainer.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/model/unet.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/evaluation/evaluation.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/nn/nn.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

using namespace kode;

int main() {
    try {
        std::cout << "========================================================\n"
                  << "  KODE Phase 11 Official Benchmark Runner\n"
                  << "========================================================\n\n";

        // -----------------------------------------------------------------
        // BENCH-TENS-01: GEMM 512x512x512 (FP32) GFLOP/s
        // -----------------------------------------------------------------
        std::cout << "[BENCHMARK] Running BENCH-TENS-01 (GEMM 512x512x512 FP32)...\n";
        dim_t gemm_dim = 512;
        tensor::Tensor mat_a = tensor::Tensor::randn({gemm_dim, gemm_dim}, 0.0f, 1.0f, 42);
        tensor::Tensor mat_b = tensor::Tensor::randn({gemm_dim, gemm_dim}, 0.0f, 1.0f, 43);
        tensor::Tensor mat_c({gemm_dim, gemm_dim}, 0.0f);

        // Warmup (3 passes)
        for (int i = 0; i < 3; ++i) {
            tensor::gemm_cpu(mat_a.data(), mat_b.data(), mat_c.data(), gemm_dim, gemm_dim, gemm_dim, false);
        }

        // Measure (10 passes)
        int gemm_passes = 10;
        auto t0_gemm = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < gemm_passes; ++i) {
            tensor::gemm_cpu(mat_a.data(), mat_b.data(), mat_c.data(), gemm_dim, gemm_dim, gemm_dim, false);
        }
        auto t1_gemm = std::chrono::high_resolution_clock::now();
        double gemm_total_sec = std::chrono::duration<double>(t1_gemm - t0_gemm).count();
        double gemm_mean_sec = gemm_total_sec / static_cast<double>(gemm_passes);
        double gemm_flops = 2.0 * static_cast<double>(gemm_dim) * static_cast<double>(gemm_dim) * static_cast<double>(gemm_dim);
        double gemm_gflops = (gemm_flops / gemm_mean_sec) / 1e9;
        double gemm_ms = gemm_mean_sec * 1000.0;
        std::cout << "  Mean Time: " << gemm_ms << " ms\n";
        std::cout << "  => BENCH-TENS-01 Performance: " << std::fixed << std::setprecision(2) << gemm_gflops << " GFLOP/s\n\n";

        // -----------------------------------------------------------------
        // BENCH-TENS-02: Conv2D 32x32x32 -> 64 (3x3) Latency
        // -----------------------------------------------------------------
        std::cout << "[BENCHMARK] Running BENCH-TENS-02 (Conv2D 32x32x32 -> 64, 3x3)...\n";
        nn::Conv2d bench_conv(32, 64, 3, 1, 1, 1, true, "bench_conv");
        tensor::Tensor conv_in = tensor::Tensor::randn({1, 32, 32, 32}, 0.0f, 1.0f, 101);
        autodiff::Variable conv_in_var = autodiff::make_variable(conv_in, false, "conv_in");

        // Warmup (5 passes)
        for (int i = 0; i < 5; ++i) {
            bench_conv.forward(conv_in_var);
        }

        // Measure (50 passes)
        int conv_passes = 50;
        auto t0_conv = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < conv_passes; ++i) {
            bench_conv.forward(conv_in_var);
        }
        auto t1_conv = std::chrono::high_resolution_clock::now();
        double conv_total_ms = std::chrono::duration<double, std::milli>(t1_conv - t0_conv).count();
        double conv_mean_ms = conv_total_ms / static_cast<double>(conv_passes);
        std::cout << "  => BENCH-TENS-02 Mean Latency: " << std::fixed << std::setprecision(3) << conv_mean_ms << " ms\n\n";

        // -----------------------------------------------------------------
        // BENCH-INF-01: Single Image DDIM-25 (32x32) Latency
        // -----------------------------------------------------------------
        std::cout << "[BENCHMARK] Running BENCH-INF-01 (DDIM-25, 32x32)...\n";
        auto pipeline = inference::DiffusionPipeline::create_default();
        std::string prompt = "a small red circle on a black background";

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
            std::cout << "  Pass " << (i + 1) << ": " << std::fixed << std::setprecision(2) << ms << " ms\n";
        }
        double ddim_mean = std::accumulate(ddim_times_ms.begin(), ddim_times_ms.end(), 0.0) / ddim_times_ms.size();
        std::cout << "  => BENCH-INF-01 Mean Latency: " << std::fixed << std::setprecision(2) << ddim_mean << " ms\n\n";

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
        std::cout << "  => BENCH-TRAIN-01 Throughput: " << std::fixed << std::setprecision(2) << throughput << " samples/s\n\n";

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
        std::cout << "  => BENCH-EVAL-01 Mean Metric Latency: " << std::fixed << std::setprecision(2) << eval_per_sample_us << " us (" << (eval_per_sample_us / 1000.0) << " ms)\n\n";

        std::cout << "========================================================\n"
                  << "OFFICIAL BENCHMARK REGISTRY SUMMARY:\n"
                  << "  BENCH-TENS-01 (GEMM 512^3):  " << std::fixed << std::setprecision(2) << gemm_gflops << " GFLOP/s (" << gemm_ms << " ms)\n"
                  << "  BENCH-TENS-02 (Conv2D 32->64): " << std::fixed << std::setprecision(3) << conv_mean_ms << " ms\n"
                  << "  BENCH-INF-01  (DDIM-25):     " << std::fixed << std::setprecision(2) << ddim_mean << " ms\n"
                  << "  BENCH-TRAIN-01(Throughput):  " << std::fixed << std::setprecision(2) << throughput << " samples/s\n"
                  << "  BENCH-EVAL-01 (Eval Metric): " << std::fixed << std::setprecision(2) << eval_per_sample_us << " us\n"
                  << "========================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
