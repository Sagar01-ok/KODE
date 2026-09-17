#include "kode/evaluation/metrics.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace kode::evaluation {

float_t compute_mse(const tensor::Tensor& a, const tensor::Tensor& b) {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("Tensor shape mismatch in compute_mse: element count mismatch.");
    }
    dim_t n = a.numel();
    if (n == 0) return 0.0f;

    const float_t* a_ptr = a.data();
    const float_t* b_ptr = b.data();
    double sum_sq = 0.0;
    for (dim_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(a_ptr[i]) - static_cast<double>(b_ptr[i]);
        sum_sq += diff * diff;
    }
    return static_cast<float_t>(sum_sq / static_cast<double>(n));
}

float_t compute_psnr(const tensor::Tensor& a, const tensor::Tensor& b, float_t max_val) {
    float_t mse = compute_mse(a, b);
    if (mse <= 1e-12f) {
        return 100.0f; // Bitwise or near-bitwise identical
    }
    double max_d = static_cast<double>(max_val);
    double mse_d = static_cast<double>(mse);
    double psnr = 10.0 * std::log10((max_d * max_d) / mse_d);
    return static_cast<float_t>(psnr);
}

float_t compute_ssim(const tensor::Tensor& a, const tensor::Tensor& b, float_t max_val) {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("Tensor shape mismatch in compute_ssim: element count mismatch.");
    }

    const Shape& shp = a.shape();
    dim_t c_dim = 1;
    dim_t spatial = 1;

    if (shp.size() == 3) {
        c_dim = shp[0];
        spatial = shp[1] * shp[2];
    } else if (shp.size() == 4) {
        c_dim = shp[1];
        spatial = shp[2] * shp[3];
    } else {
        spatial = a.numel();
    }

    if (spatial == 0) return 1.0f;

    double c1 = (0.01 * max_val) * (0.01 * max_val);
    double c2 = (0.03 * max_val) * (0.03 * max_val);

    const float_t* a_ptr = a.data();
    const float_t* b_ptr = b.data();

    double total_ssim = 0.0;

    for (dim_t c = 0; c < c_dim; ++c) {
        const float_t* a_chan = a_ptr + (c * spatial);
        const float_t* b_chan = b_ptr + (c * spatial);

        double mean_a = 0.0;
        double mean_b = 0.0;
        for (dim_t i = 0; i < spatial; ++i) {
            mean_a += a_chan[i];
            mean_b += b_chan[i];
        }
        mean_a /= static_cast<double>(spatial);
        mean_b /= static_cast<double>(spatial);

        double var_a = 0.0;
        double var_b = 0.0;
        double cov_ab = 0.0;
        for (dim_t i = 0; i < spatial; ++i) {
            double da = a_chan[i] - mean_a;
            double db = b_chan[i] - mean_b;
            var_a += da * da;
            var_b += db * db;
            cov_ab += da * db;
        }
        var_a /= static_cast<double>(spatial);
        var_b /= static_cast<double>(spatial);
        cov_ab /= static_cast<double>(spatial);

        double num = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2);
        double den = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);

        double ssim_c = (den > 0.0) ? (num / den) : 1.0;
        total_ssim += ssim_c;
    }

    return static_cast<float_t>(total_ssim / static_cast<double>(c_dim));
}

ColorHistogram compute_color_histogram(const tensor::Tensor& image, dim_t num_bins) {
    if (num_bins <= 0) {
        throw std::invalid_argument("num_bins must be positive.");
    }

    ColorHistogram hist;
    hist.num_bins = num_bins;
    hist.r_bins.assign(num_bins, 0.0f);
    hist.g_bins.assign(num_bins, 0.0f);
    hist.b_bins.assign(num_bins, 0.0f);

    const Shape& shp = image.shape();
    dim_t c_dim = 0, spatial = 0;
    const float_t* data = image.data();

    if (shp.size() == 3) {
        c_dim = shp[0];
        spatial = shp[1] * shp[2];
    } else if (shp.size() == 4) {
        c_dim = shp[1];
        spatial = shp[2] * shp[3];
    } else {
        throw std::invalid_argument("Image tensor must be 3D (C, H, W) or 4D (1, C, H, W).");
    }

    if (c_dim < 3 || spatial == 0) return hist;

    const float_t* r_plane = data;
    const float_t* g_plane = data + spatial;
    const float_t* b_plane = data + (2 * spatial);

    for (dim_t i = 0; i < spatial; ++i) {
        // Map [-1.0, 1.0] -> [0.0, 1.0]
        float_t r_norm = std::clamp((r_plane[i] + 1.0f) * 0.5f, 0.0f, 1.0f);
        float_t g_norm = std::clamp((g_plane[i] + 1.0f) * 0.5f, 0.0f, 1.0f);
        float_t b_norm = std::clamp((b_plane[i] + 1.0f) * 0.5f, 0.0f, 1.0f);

        dim_t r_idx = std::clamp(static_cast<dim_t>(r_norm * static_cast<float_t>(num_bins)), static_cast<dim_t>(0), num_bins - 1);
        dim_t g_idx = std::clamp(static_cast<dim_t>(g_norm * static_cast<float_t>(num_bins)), static_cast<dim_t>(0), num_bins - 1);
        dim_t b_idx = std::clamp(static_cast<dim_t>(b_norm * static_cast<float_t>(num_bins)), static_cast<dim_t>(0), num_bins - 1);

        hist.r_bins[r_idx] += 1.0f;
        hist.g_bins[g_idx] += 1.0f;
        hist.b_bins[b_idx] += 1.0f;
    }

    // Normalize probabilities
    float_t norm_factor = 1.0f / static_cast<float_t>(spatial);
    for (dim_t b = 0; b < num_bins; ++b) {
        hist.r_bins[b] *= norm_factor;
        hist.g_bins[b] *= norm_factor;
        hist.b_bins[b] *= norm_factor;
    }

    return hist;
}

float_t compute_histogram_intersection(const ColorHistogram& h1, const ColorHistogram& h2) {
    if (h1.num_bins != h2.num_bins || h1.num_bins == 0) return 0.0f;

    dim_t n = h1.num_bins;
    double r_sim = 0.0, g_sim = 0.0, b_sim = 0.0;

    for (dim_t i = 0; i < n; ++i) {
        r_sim += std::min(h1.r_bins[i], h2.r_bins[i]);
        g_sim += std::min(h1.g_bins[i], h2.g_bins[i]);
        b_sim += std::min(h1.b_bins[i], h2.b_bins[i]);
    }

    return static_cast<float_t>((r_sim + g_sim + b_sim) / 3.0);
}

float_t compute_bhattacharyya_distance(const ColorHistogram& h1, const ColorHistogram& h2) {
    if (h1.num_bins != h2.num_bins || h1.num_bins == 0) return 10.0f;

    dim_t n = h1.num_bins;
    double r_bc = 0.0, g_bc = 0.0, b_bc = 0.0;

    for (dim_t i = 0; i < n; ++i) {
        r_bc += std::sqrt(std::max(0.0f, h1.r_bins[i]) * std::max(0.0f, h2.r_bins[i]));
        g_bc += std::sqrt(std::max(0.0f, h1.g_bins[i]) * std::max(0.0f, h2.g_bins[i]));
        b_bc += std::sqrt(std::max(0.0f, h1.b_bins[i]) * std::max(0.0f, h2.b_bins[i]));
    }

    double avg_bc = std::clamp((r_bc + g_bc + b_bc) / 3.0, 1e-12, 1.0);
    return static_cast<float_t>(-std::log(avg_bc));
}

DistributionStats compute_image_distribution_stats(const std::vector<tensor::Tensor>& images) {
    DistributionStats stats;
    stats.mean.assign(3, 0.0f);
    stats.std.assign(3, 0.0f);
    stats.cov = tensor::Tensor({3, 3}, 0.0f);

    if (images.empty()) return stats;

    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;
    size_t total_pixels = 0;

    for (const auto& img : images) {
        const Shape& shp = img.shape();
        dim_t spatial = (shp.size() == 3) ? (shp[1] * shp[2]) : (shp[2] * shp[3]);
        const float_t* ptr = img.data();

        for (dim_t i = 0; i < spatial; ++i) {
            r_sum += ptr[i];
            g_sum += ptr[spatial + i];
            b_sum += ptr[2 * spatial + i];
        }
        total_pixels += spatial;
    }

    if (total_pixels == 0) return stats;

    double mean_r = r_sum / static_cast<double>(total_pixels);
    double mean_g = g_sum / static_cast<double>(total_pixels);
    double mean_b = b_sum / static_cast<double>(total_pixels);

    stats.mean[0] = static_cast<float_t>(mean_r);
    stats.mean[1] = static_cast<float_t>(mean_g);
    stats.mean[2] = static_cast<float_t>(mean_b);

    double c00 = 0.0, c01 = 0.0, c02 = 0.0;
    double c11 = 0.0, c12 = 0.0, c22 = 0.0;

    for (const auto& img : images) {
        const Shape& shp = img.shape();
        dim_t spatial = (shp.size() == 3) ? (shp[1] * shp[2]) : (shp[2] * shp[3]);
        const float_t* ptr = img.data();

        for (dim_t i = 0; i < spatial; ++i) {
            double dr = ptr[i] - mean_r;
            double dg = ptr[spatial + i] - mean_g;
            double db = ptr[2 * spatial + i] - mean_b;

            c00 += dr * dr;
            c01 += dr * dg;
            c02 += dr * db;
            c11 += dg * dg;
            c12 += dg * db;
            c22 += db * db;
        }
    }

    double norm = static_cast<double>(total_pixels);
    c00 /= norm; c01 /= norm; c02 /= norm;
    c11 /= norm; c12 /= norm; c22 /= norm;

    stats.std[0] = static_cast<float_t>(std::sqrt(c00));
    stats.std[1] = static_cast<float_t>(std::sqrt(c11));
    stats.std[2] = static_cast<float_t>(std::sqrt(c22));

    float_t* cov_ptr = stats.cov.data();
    cov_ptr[0] = static_cast<float_t>(c00); cov_ptr[1] = static_cast<float_t>(c01); cov_ptr[2] = static_cast<float_t>(c02);
    cov_ptr[3] = static_cast<float_t>(c01); cov_ptr[4] = static_cast<float_t>(c11); cov_ptr[5] = static_cast<float_t>(c12);
    cov_ptr[6] = static_cast<float_t>(c02); cov_ptr[7] = static_cast<float_t>(c12); cov_ptr[8] = static_cast<float_t>(c22);

    return stats;
}

float_t compute_frechet_distance(const DistributionStats& s1, const DistributionStats& s2) {
    if (s1.mean.size() < 3 || s2.mean.size() < 3) return 0.0f;

    // 2-Wasserstein Gaussian distance on color feature representations:
    // W_2^2 = ||mu_1 - mu_2||^2 + sum_{c=0}^2 (sigma_{1,c} - sigma_{2,c})^2
    double mean_diff_sq = 0.0;
    double std_diff_sq = 0.0;

    for (int c = 0; c < 3; ++c) {
        double dm = static_cast<double>(s1.mean[c]) - static_cast<double>(s2.mean[c]);
        double ds = static_cast<double>(s1.std[c]) - static_cast<double>(s2.std[c]);
        mean_diff_sq += dm * dm;
        std_diff_sq += ds * ds;
    }

    double w2 = std::sqrt(std::max(0.0, mean_diff_sq + std_diff_sq));
    return static_cast<float_t>(w2);
}

} // namespace kode::evaluation
