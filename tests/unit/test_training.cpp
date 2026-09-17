#include "kode/tensor/tensor.hpp"
#include "kode/autodiff/autodiff.hpp"
#include "kode/nn/nn.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/model/unet.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/training/optimizer.hpp"
#include "kode/training/checkpoint.hpp"
#include "kode/training/logger.hpp"
#include "kode/training/trainer.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <filesystem>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_diffusion_schedules() {
    std::cout << "[TEST] Running Gaussian Diffusion schedules and noising..." << std::endl;

    diffusion::DiffusionConfig lin_cfg;
    lin_cfg.num_timesteps = 100;
    lin_cfg.schedule_type = diffusion::ScheduleType::Linear;
    diffusion::GaussianDiffusion lin_diff(lin_cfg);

    KODE_TEST_ASSERT(lin_diff.num_timesteps() == 100);
    KODE_TEST_ASSERT(lin_diff.beta(0) < lin_diff.beta(99));
    KODE_TEST_ASSERT(lin_diff.alpha_bar(0) > lin_diff.alpha_bar(99));
    KODE_TEST_ASSERT(lin_diff.alpha_bar(0) > 0.95f);
    KODE_TEST_ASSERT(lin_diff.alpha_bar(99) < 0.5f);

    diffusion::DiffusionConfig cos_cfg;
    cos_cfg.num_timesteps = 1000;
    cos_cfg.schedule_type = diffusion::ScheduleType::Cosine;
    diffusion::GaussianDiffusion cos_diff(cos_cfg);

    KODE_TEST_ASSERT(cos_diff.num_timesteps() == 1000);
    KODE_TEST_ASSERT(cos_diff.alpha_bar(0) > 0.99f);
    KODE_TEST_ASSERT(cos_diff.alpha_bar(999) < 0.01f);

    // Test forward noising q_sample
    tensor::Tensor x0({2, 3, 32, 32}, 1.0f);
    std::vector<dim_t> t = {0, 999};
    tensor::Tensor noise({2, 3, 32, 32}, 0.0f); // Zero noise to check mean scaling
    tensor::Tensor xt = cos_diff.q_sample(x0, t, noise);

    // At t=0, xt should be sqrt(alpha_bar(0)) * 1.0 ≈ 0.999 * 1.0
    float_t val_t0 = xt.at({0, 0, 0, 0});
    KODE_TEST_ASSERT(std::abs(val_t0 - cos_diff.sqrt_alpha_bar(0)) < 1e-4f);

    // At t=999, xt should be sqrt(alpha_bar(999)) * 1.0 ≈ close to 0
    float_t val_t999 = xt.at({1, 0, 0, 0});
    KODE_TEST_ASSERT(val_t999 < 0.1f);

    // Single step reverse sampling checks
    tensor::Tensor eps_pred({2, 3, 32, 32}, 0.1f);
    tensor::Tensor x_prev_ddpm = cos_diff.p_sample_step(eps_pred, xt, 50);
    KODE_TEST_ASSERT(x_prev_ddpm.shape() == xt.shape());

    tensor::Tensor x_prev_ddim = cos_diff.ddim_step(eps_pred, xt, 50, 40);
    KODE_TEST_ASSERT(x_prev_ddim.shape() == xt.shape());

    std::cout << "  -> Gaussian Diffusion schedules and sampling PASSED" << std::endl;
}

void test_adamw_and_lr_schedule() {
    std::cout << "[TEST] Running AdamW and LR schedule verification..." << std::endl;

    // 1. CosineAnnealingLR
    training::CosineAnnealingLR sched(1e-3f, 1000, 100, 1e-5f);
    float_t lr_0 = sched.get_lr(0);
    float_t lr_warmup = sched.get_lr(100);
    float_t lr_mid = sched.get_lr(550);
    float_t lr_end = sched.get_lr(1000);

    KODE_TEST_ASSERT(lr_0 < lr_warmup);
    KODE_TEST_ASSERT(std::abs(lr_warmup - 1e-3f) < 1e-5f);
    KODE_TEST_ASSERT(lr_mid < lr_warmup && lr_mid > lr_end);
    KODE_TEST_ASSERT(std::abs(lr_end - 1e-5f) < 1e-6f);

    // 2. Gradient Clipping
    autodiff::Variable p1 = autodiff::make_variable(tensor::Tensor({10}, 0.0f), true);
    autodiff::Variable p2 = autodiff::make_variable(tensor::Tensor({10}, 0.0f), true);
    p1->grad() = tensor::Tensor({10}, 3.0f);
    p2->grad() = tensor::Tensor({10}, 4.0f);
    // Norm = sqrt(10 * 9 + 10 * 16) = sqrt(90 + 160) = sqrt(250) ≈ 15.811
    float_t orig_norm = training::clip_grad_norm({p1, p2}, 5.0f);
    KODE_TEST_ASSERT(std::abs(orig_norm - std::sqrt(250.0f)) < 1e-3f);

    // Recompute norm after clipping - should be <= 5.0 + eps
    float_t clipped_norm = training::clip_grad_norm({p1, p2}, 100.0f);
    KODE_TEST_ASSERT(clipped_norm <= 5.01f);

    // 3. AdamW step
    training::AdamWConfig opt_cfg;
    opt_cfg.lr = 0.1f;
    opt_cfg.weight_decay = 0.0f;
    autodiff::Variable w = autodiff::make_variable(tensor::Tensor({2}, 1.0f), true);
    training::AdamW opt({w}, opt_cfg);

    w->grad() = tensor::Tensor({2}, 0.5f);
    opt.step();

    // After 1 step with positive gradient, weight should decrease
    KODE_TEST_ASSERT(w->data().at({0}) < 1.0f);
    KODE_TEST_ASSERT(opt.step_count() == 1);
    KODE_TEST_ASSERT(!opt.m_moments()[0].is_empty());
    KODE_TEST_ASSERT(!opt.v_moments()[0].is_empty());

    std::cout << "  -> AdamW & LR Schedule PASSED" << std::endl;
}

void test_checkpoint_binary_serialization() {
    std::cout << "[TEST] Running Checkpoint binary serialization (.kode)..." << std::endl;

    nn::Linear model(4, 2, true, "test_layer");
    training::AdamW opt(model.parameters());

    // Perform a dummy step so moments are populated
    for (auto& p : model.parameters()) {
        p->grad() = tensor::Tensor::ones(p->shape());
    }
    opt.step();

    std::string test_file = "test_ckpt.kode";
    training::CheckpointMetadata meta_in;
    meta_in.step = 42;
    meta_in.epoch = 2;
    meta_in.loss = 0.12345f;
    meta_in.config_json = "{\"test\": true}";

    training::Checkpoint::save(test_file, model, &opt, meta_in);

    // Verify metadata reading
    training::CheckpointMetadata meta_read = training::Checkpoint::read_metadata(test_file);
    KODE_TEST_ASSERT(meta_read.step == 42);
    KODE_TEST_ASSERT(meta_read.epoch == 2);
    KODE_TEST_ASSERT(std::abs(meta_read.loss - 0.12345f) < 1e-5f);
    KODE_TEST_ASSERT(meta_read.config_json == "{\"test\": true}");

    // Load into fresh model & optimizer
    nn::Linear model2(4, 2, true, "test_layer");
    training::AdamW opt2(model2.parameters());

    training::CheckpointMetadata meta_out = training::Checkpoint::load(test_file, model2, &opt2);
    KODE_TEST_ASSERT(meta_out.step == 42);
    KODE_TEST_ASSERT(opt2.step_count() == 1);

    // Bitwise check weights
    for (dim_t i = 0; i < model.weight()->numel(); ++i) {
        KODE_TEST_ASSERT(model.weight()->data().data()[i] == model2.weight()->data().data()[i]);
    }
    for (dim_t i = 0; i < model.bias()->numel(); ++i) {
        KODE_TEST_ASSERT(model.bias()->data().data()[i] == model2.bias()->data().data()[i]);
    }

    // Clean up file
    std::filesystem::remove(test_file);

    std::cout << "  -> Checkpoint Binary Serialization (.kode) PASSED" << std::endl;
}

void test_trainer_pipeline_step() {
    std::cout << "[TEST] Running full Trainer pipeline on mini-batch..." << std::endl;

    model::UNetConfig unet_cfg = model::UNetConfig::default_config();
    auto unet = std::make_shared<model::UNet>(unet_cfg);
    auto text_enc = std::make_shared<text::TextEncoder>(1024, 64, 16);
    auto tokenizer = std::make_shared<text::Tokenizer>();

    diffusion::DiffusionConfig diff_cfg;
    diff_cfg.num_timesteps = 1000;
    diff_cfg.schedule_type = diffusion::ScheduleType::Cosine;
    auto diffusion = std::make_shared<diffusion::GaussianDiffusion>(diff_cfg);

    training::TrainingConfig train_cfg;
    train_cfg.batch_size = 2;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.warmup_steps = 10;
    train_cfg.total_steps = 100;
    train_cfg.checkpoint_every_steps = 0; // Don't write during test
    train_cfg.log_every_steps = 1;
    train_cfg.log_csv_path = ""; // No CSV in test

    training::Trainer trainer(unet, text_enc, tokenizer, diffusion, train_cfg);

    tensor::Tensor dummy_images = tensor::Tensor::randn({2, 3, 32, 32}, 0.0f, 1.0f);
    std::vector<std::string> captions = {"a red square", "a blue circle"};

    // Execute 2 training steps
    float_t loss1 = trainer.train_step(dummy_images, captions);
    KODE_TEST_ASSERT(loss1 > 0.0f && !std::isnan(loss1) && !std::isinf(loss1));
    KODE_TEST_ASSERT(trainer.step() == 1);

    float_t loss2 = trainer.train_step(dummy_images, captions);
    KODE_TEST_ASSERT(loss2 > 0.0f && !std::isnan(loss2) && !std::isinf(loss2));
    KODE_TEST_ASSERT(trainer.step() == 2);

    std::cout << "  -> Trainer Pipeline Steps PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "KODE Phase 8: Training Engine Test Suite" << std::endl;
        std::cout << "========================================" << std::endl;

        test_diffusion_schedules();
        test_adamw_and_lr_schedule();
        test_checkpoint_binary_serialization();
        test_trainer_pipeline_step();

        std::cout << "========================================" << std::endl;
        std::cout << "ALL PHASE 8 TESTS PASSED SUCCESSFULLY!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_training: " << e.what() << std::endl;
        return 1;
    }
}
