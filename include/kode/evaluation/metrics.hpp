#pragma once

#include "kode/core/types.hpp"
#include "kode/tensor/tensor.hpp"
#include <vector>
#include <cmath>

namespace kode::evaluation {

// Image reconstruction and fidelity metrics
float_t compute_mse(const tensor::Tensor& a, const tensor::Tensor& b);
float_t compute_psnr(const tensor::Tensor& a, const tensor::Tensor& b, float_t max_val = 2.0f);
float_t compute_ssim(const tensor::Tensor& a, const tensor::Tensor& b, float_t max_val = 2.0f);

// Color histogram statistics
struct ColorHistogram {
    std::vector<float_t> r_bins;
    std::vector<float_t> g_bins;
    std::vector<float_t> b_bins;
    dim_t num_bins{16};
};

ColorHistogram compute_color_histogram(const tensor::Tensor& image, dim_t num_bins = 16);
float_t compute_histogram_intersection(const ColorHistogram& h1, const ColorHistogram& h2);
float_t compute_bhattacharyya_distance(const ColorHistogram& h1, const ColorHistogram& h2);

// Pixel Fréchet Distance (PFD) across feature distributions
struct DistributionStats {
    std::vector<float_t> mean; // Mean per channel (3,)
    std::vector<float_t> std;  // Std dev per channel (3,)
    tensor::Tensor cov;        // Full covariance matrix (3, 3)
};

DistributionStats compute_image_distribution_stats(const std::vector<tensor::Tensor>& images);
float_t compute_frechet_distance(const DistributionStats& s1, const DistributionStats& s2);

} // namespace kode::evaluation
