#include "kode/evaluation/grounding_evaluator.hpp"
#include "kode/evaluation/metrics.hpp"
#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace kode::evaluation {

GroundingEvaluator::GroundingEvaluator() {
    fg_palette_ = {
        {"red", 1.0f, 0.0f, 0.0f},
        {"green", 0.0f, 0.85f, 0.0f},
        {"blue", 0.0f, 0.25f, 1.0f},
        {"yellow", 1.0f, 0.95f, 0.0f},
        {"cyan", 0.0f, 0.95f, 0.95f},
        {"magenta", 1.0f, 0.0f, 0.95f},
        {"white", 0.95f, 0.95f, 0.95f},
        {"orange", 1.0f, 0.5f, 0.0f}
    };

    bg_palette_ = {
        {"black", 0.05f, 0.05f, 0.05f},
        {"dark gray", 0.25f, 0.25f, 0.25f},
        {"light gray", 0.75f, 0.75f, 0.75f},
        {"white", 0.95f, 0.95f, 0.95f},
        {"navy", 0.0f, 0.05f, 0.35f},
        {"dark green", 0.05f, 0.3f, 0.05f},
        {"dark red", 0.35f, 0.05f, 0.05f},
        {"purple", 0.3f, 0.05f, 0.35f}
    };
}

std::string GroundingEvaluator::detect_background_color(const tensor::Tensor& image, float_t& out_dist) const {
    const Shape& shp = image.shape();
    dim_t h = (shp.size() == 3) ? shp[1] : shp[2];
    dim_t w = (shp.size() == 3) ? shp[2] : shp[3];
    dim_t spatial = h * w;
    const float_t* data = image.data();

    // Estimate background color from border/perimeter pixels
    double bg_r = 0.0, bg_g = 0.0, bg_b = 0.0;
    size_t bg_count = 0;

    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            if (y == 0 || y == h - 1 || x == 0 || x == w - 1) {
                size_t idx = static_cast<size_t>(y * w + x);
                bg_r += (data[idx] + 1.0f) * 0.5f;
                bg_g += (data[spatial + idx] + 1.0f) * 0.5f;
                bg_b += (data[2 * spatial + idx] + 1.0f) * 0.5f;
                bg_count++;
            }
        }
    }

    if (bg_count > 0) {
        bg_r /= static_cast<double>(bg_count);
        bg_g /= static_cast<double>(bg_count);
        bg_b /= static_cast<double>(bg_count);
    }

    float_t mean_r = static_cast<float_t>(std::clamp(bg_r, 0.0, 1.0));
    float_t mean_g = static_cast<float_t>(std::clamp(bg_g, 0.0, 1.0));
    float_t mean_b = static_cast<float_t>(std::clamp(bg_b, 0.0, 1.0));

    std::string best_color = "black";
    float_t min_dist = 1e9f;

    for (const auto& p : bg_palette_) {
        float_t dr = mean_r - p.r;
        float_t dg = mean_g - p.g;
        float_t db = mean_b - p.b;
        float_t dist = std::sqrt(dr * dr + dg * dg + db * db);
        if (dist < min_dist) {
            min_dist = dist;
            best_color = p.name;
        }
    }

    out_dist = min_dist;
    return best_color;
}

std::string GroundingEvaluator::detect_foreground_color(const tensor::Tensor& image, float_t& out_dist) const {
    const Shape& shp = image.shape();
    dim_t h = (shp.size() == 3) ? shp[1] : shp[2];
    dim_t w = (shp.size() == 3) ? shp[2] : shp[3];
    dim_t spatial = h * w;
    const float_t* data = image.data();

    // 1. Estimate background color from perimeter
    double bg_r = 0.0, bg_g = 0.0, bg_b = 0.0;
    size_t bg_count = 0;

    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            if (y == 0 || y == h - 1 || x == 0 || x == w - 1) {
                size_t idx = static_cast<size_t>(y * w + x);
                bg_r += (data[idx] + 1.0f) * 0.5f;
                bg_g += (data[spatial + idx] + 1.0f) * 0.5f;
                bg_b += (data[2 * spatial + idx] + 1.0f) * 0.5f;
                bg_count++;
            }
        }
    }

    if (bg_count > 0) {
        bg_r /= static_cast<double>(bg_count);
        bg_g /= static_cast<double>(bg_count);
        bg_b /= static_cast<double>(bg_count);
    }

    // 2. Find maximum contrast and isolate foreground
    double max_diff = 0.0;
    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y * w + x);
            double r = (data[idx] + 1.0f) * 0.5f;
            double g = (data[spatial + idx] + 1.0f) * 0.5f;
            double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
            double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                    (g - bg_g) * (g - bg_g) + 
                                    (b - bg_b) * (b - bg_b));
            if (diff > max_diff) max_diff = diff;
        }
    }

    double threshold = std::max(0.12, max_diff * 0.45);
    double fg_r = 0.0, fg_g = 0.0, fg_b = 0.0;
    size_t fg_count = 0;

    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y * w + x);
            double r = (data[idx] + 1.0f) * 0.5f;
            double g = (data[spatial + idx] + 1.0f) * 0.5f;
            double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
            double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                    (g - bg_g) * (g - bg_g) + 
                                    (b - bg_b) * (b - bg_b));

            if (diff >= threshold) {
                fg_r += r;
                fg_g += g;
                fg_b += b;
                fg_count++;
            }
        }
    }

    if (fg_count < 4) {
        // Fallback to central region
        dim_t cy = h / 2, cx = w / 2;
        size_t c_idx = static_cast<size_t>(cy * w + cx);
        fg_r = (data[c_idx] + 1.0f) * 0.5f;
        fg_g = (data[spatial + c_idx] + 1.0f) * 0.5f;
        fg_b = (data[2 * spatial + c_idx] + 1.0f) * 0.5f;
        fg_count = 1;
    }

    float_t mean_r = static_cast<float_t>(fg_r / static_cast<double>(fg_count));
    float_t mean_g = static_cast<float_t>(fg_g / static_cast<double>(fg_count));
    float_t mean_b = static_cast<float_t>(fg_b / static_cast<double>(fg_count));

    std::string best_color = "unknown";
    float_t min_dist = 1e9f;

    for (const auto& p : fg_palette_) {
        float_t dr = mean_r - p.r;
        float_t dg = mean_g - p.g;
        float_t db = mean_b - p.b;
        float_t dist = std::sqrt(dr * dr + dg * dg + db * db);
        if (dist < min_dist) {
            min_dist = dist;
            best_color = p.name;
        }
    }

    out_dist = min_dist;
    return best_color;
}

std::string GroundingEvaluator::detect_foreground_shape(const tensor::Tensor& image, float_t& out_confidence) const {
    const Shape& shp = image.shape();
    dim_t h = (shp.size() == 3) ? shp[1] : shp[2];
    dim_t w = (shp.size() == 3) ? shp[2] : shp[3];
    dim_t spatial = h * w;
    const float_t* data = image.data();

    // 1. Background color from border
    double bg_r = 0.0, bg_g = 0.0, bg_b = 0.0;
    size_t bg_count = 0;
    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            if (y == 0 || y == h - 1 || x == 0 || x == w - 1) {
                size_t idx = static_cast<size_t>(y * w + x);
                bg_r += (data[idx] + 1.0f) * 0.5f;
                bg_g += (data[spatial + idx] + 1.0f) * 0.5f;
                bg_b += (data[2 * spatial + idx] + 1.0f) * 0.5f;
                bg_count++;
            }
        }
    }
    if (bg_count > 0) {
        bg_r /= static_cast<double>(bg_count);
        bg_g /= static_cast<double>(bg_count);
        bg_b /= static_cast<double>(bg_count);
    }

    // 2. Find max difference
    double max_diff = 0.0;
    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y * w + x);
            double r = (data[idx] + 1.0f) * 0.5f;
            double g = (data[spatial + idx] + 1.0f) * 0.5f;
            double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
            double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                    (g - bg_g) * (g - bg_g) + 
                                    (b - bg_b) * (b - bg_b));
            if (diff > max_diff) max_diff = diff;
        }
    }

    double threshold = std::max(0.12, max_diff * 0.45);

    // 3. Extract foreground pixel coordinates
    std::vector<std::pair<dim_t, dim_t>> fg_pixels;
    dim_t min_x = w, max_x = 0, min_y = h, max_y = 0;
    double sum_x = 0.0, sum_y = 0.0;

    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y * w + x);
            double r = (data[idx] + 1.0f) * 0.5f;
            double g = (data[spatial + idx] + 1.0f) * 0.5f;
            double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
            double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                    (g - bg_g) * (g - bg_g) + 
                                    (b - bg_b) * (b - bg_b));

            if (diff >= threshold) {
                fg_pixels.emplace_back(x, y);
                sum_x += static_cast<double>(x);
                sum_y += static_cast<double>(y);
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (fg_pixels.size() < 6) {
        out_confidence = 0.0f;
        return "circle"; // fallback default
    }

    size_t n_fg = fg_pixels.size();
    double cx = sum_x / static_cast<double>(n_fg);
    (void)cx;
    double cy = sum_y / static_cast<double>(n_fg);

    double box_w = static_cast<double>(max_x - min_x + 1);
    double box_h = static_cast<double>(max_y - min_y + 1);
    double box_area = box_w * box_h;
    double fill_ratio = static_cast<double>(n_fg) / box_area;

    double box_mid_y = (static_cast<double>(min_y) + static_cast<double>(max_y)) / 2.0;
    double vert_asymmetry = (cy - box_mid_y) / std::max(1.0, box_h);

    // Check how many of the 4 bounding box corners are filled
    auto is_fg_pixel = [&](dim_t px, dim_t py) -> bool {
        size_t idx = static_cast<size_t>(py * w + px);
        double r = (data[idx] + 1.0f) * 0.5f;
        double g = (data[spatial + idx] + 1.0f) * 0.5f;
        double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
        double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                (g - bg_g) * (g - bg_g) + 
                                (b - bg_b) * (b - bg_b));
        return diff >= threshold;
    };

    int corners_filled = 0;
    if (is_fg_pixel(min_x, min_y)) corners_filled++;
    if (is_fg_pixel(max_x, min_y)) corners_filled++;
    if (is_fg_pixel(min_x, max_y)) corners_filled++;
    if (is_fg_pixel(max_x, max_y)) corners_filled++;

    // Width analysis across top rows (constant for cross arm, expanding for diamond/triangle)
    auto count_row_width = [&](dim_t py) -> dim_t {
        dim_t count = 0;
        for (dim_t px = min_x; px <= max_x; ++px) {
            if (is_fg_pixel(px, py)) count++;
        }
        return count;
    };

    dim_t w_top = count_row_width(min_y);
    dim_t y_check = std::min(max_y, static_cast<dim_t>(min_y + std::max(1.0, box_h / 4.0)));
    dim_t w_lower = count_row_width(y_check);
    dim_t width_growth = (w_lower > w_top) ? (w_lower - w_top) : 0;

    // Decision Logic
    if (vert_asymmetry > 0.065) {
        out_confidence = static_cast<float_t>(std::clamp(vert_asymmetry / 0.16, 0.5, 1.0));
        return "triangle";
    }

    if (corners_filled >= 3 && fill_ratio >= 0.85) {
        out_confidence = static_cast<float_t>(std::clamp(fill_ratio, 0.7, 1.0));
        return "square";
    }

    if (fill_ratio >= 0.65) {
        out_confidence = static_cast<float_t>(std::clamp(1.0 - std::abs(fill_ratio - 0.785), 0.5, 1.0));
        return "circle";
    }

    if (width_growth <= 1) {
        out_confidence = 0.9f;
        return "cross";
    }

    out_confidence = 0.9f;
    return "diamond";
}

std::string GroundingEvaluator::detect_position(const tensor::Tensor& image) const {
    const Shape& shp = image.shape();
    dim_t h = (shp.size() == 3) ? shp[1] : shp[2];
    dim_t w = (shp.size() == 3) ? shp[2] : shp[3];
    dim_t spatial = h * w;
    const float_t* data = image.data();

    // Background estimate
    double bg_r = 0.0, bg_g = 0.0, bg_b = 0.0;
    size_t bg_count = 0;
    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            if (y == 0 || y == h - 1 || x == 0 || x == w - 1) {
                size_t idx = static_cast<size_t>(y * w + x);
                bg_r += (data[idx] + 1.0f) * 0.5f;
                bg_g += (data[spatial + idx] + 1.0f) * 0.5f;
                bg_b += (data[2 * spatial + idx] + 1.0f) * 0.5f;
                bg_count++;
            }
        }
    }
    if (bg_count > 0) {
        bg_r /= static_cast<double>(bg_count);
        bg_g /= static_cast<double>(bg_count);
        bg_b /= static_cast<double>(bg_count);
    }

    double max_diff = 0.0;
    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y * w + x);
            double r = (data[idx] + 1.0f) * 0.5f;
            double g = (data[spatial + idx] + 1.0f) * 0.5f;
            double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
            double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                    (g - bg_g) * (g - bg_g) + 
                                    (b - bg_b) * (b - bg_b));
            if (diff > max_diff) max_diff = diff;
        }
    }

    double threshold = std::max(0.12, max_diff * 0.45);
    double sum_x = 0.0, sum_y = 0.0;
    size_t fg_count = 0;

    for (dim_t y = 0; y < h; ++y) {
        for (dim_t x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y * w + x);
            double r = (data[idx] + 1.0f) * 0.5f;
            double g = (data[spatial + idx] + 1.0f) * 0.5f;
            double b = (data[2 * spatial + idx] + 1.0f) * 0.5f;
            double diff = std::sqrt((r - bg_r) * (r - bg_r) + 
                                    (g - bg_g) * (g - bg_g) + 
                                    (b - bg_b) * (b - bg_b));
            if (diff >= threshold) {
                sum_x += static_cast<double>(x);
                sum_y += static_cast<double>(y);
                fg_count++;
            }
        }
    }

    if (fg_count == 0) return "center";

    double cx = sum_x / static_cast<double>(fg_count);
    double cy = sum_y / static_cast<double>(fg_count);

    // Anchor positions in SyntheticGroundingDataset
    struct PosAnchor {
        const char* name;
        double ax, ay;
    };
    PosAnchor anchors[] = {
        {"center", 16.0, 16.0},
        {"top-left", 10.0, 10.0},
        {"top-right", 22.0, 10.0},
        {"bottom-left", 10.0, 22.0},
        {"bottom-right", 22.0, 22.0}
    };

    const char* best_pos = "center";
    double min_dist_sq = 1e9;
    for (const auto& a : anchors) {
        double dx = cx - a.ax;
        double dy = cy - a.ay;
        double d2 = dx * dx + dy * dy;
        if (d2 < min_dist_sq) {
            min_dist_sq = d2;
            best_pos = a.name;
        }
    }
    return best_pos;
}

GroundingResult GroundingEvaluator::evaluate_sample(
    const tensor::Tensor& image,
    const std::string& prompt
) {
    GroundingResult res;
    res.prompt = prompt;

    // Split prompt into foreground description and background description
    // Example: "a small red circle in the top-left on a black background"
    std::string fg_part = prompt;
    std::string bg_part = "";
    size_t on_pos = prompt.find(" on a ");
    if (on_pos != std::string::npos) {
        fg_part = prompt.substr(0, on_pos);
        bg_part = prompt.substr(on_pos + 6);
    }

    // 1. Parse expectations from prompt
    std::vector<std::string> known_colors = {
        "red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange"
    };
    for (const auto& c : known_colors) {
        std::string token = " " + c + " ";
        std::string prefix = c + " ";
        if (fg_part.find(token) != std::string::npos || fg_part.rfind(prefix, 0) == 0) {
            res.expected_color = c;
            break;
        }
    }

    std::vector<std::string> known_shapes = {
        "circle", "square", "triangle", "cross", "diamond"
    };
    for (const auto& s : known_shapes) {
        if (fg_part.find(s) != std::string::npos) {
            res.expected_shape = s;
            break;
        }
    }

    std::vector<std::string> known_bgs = {
        "dark gray", "light gray", "dark green", "dark red",
        "black", "white", "navy", "purple"
    };
    for (const auto& b : known_bgs) {
        if (!bg_part.empty() && bg_part.find(b) != std::string::npos) {
            res.expected_background = b;
            break;
        } else if (prompt.find(b) != std::string::npos) {
            res.expected_background = b;
            break;
        }
    }

    std::vector<std::string> known_pos = {
        "top-left", "top-right", "bottom-left", "bottom-right"
    };
    for (const auto& p : known_pos) {
        if (prompt.find(p) != std::string::npos) {
            res.expected_position = p;
            break;
        }
    }
    if (res.expected_position.empty()) {
        res.expected_position = "center";
    }

    // 2. Attribute and geometry detections
    float_t color_dist = 0.0f;
    res.detected_color = detect_foreground_color(image, color_dist);
    res.color_distance = color_dist;
    res.color_matched = (!res.expected_color.empty() && res.detected_color == res.expected_color);

    float_t shape_conf = 0.0f;
    res.detected_shape = detect_foreground_shape(image, shape_conf);
    res.shape_confidence = shape_conf;
    res.shape_matched = (!res.expected_shape.empty() && res.detected_shape == res.expected_shape);

    float_t bg_dist = 0.0f;
    res.detected_background = detect_background_color(image, bg_dist);
    res.background_matched = (!res.expected_background.empty() && res.detected_background == res.expected_background);

    res.detected_position = detect_position(image);
    res.position_matched = (!res.expected_position.empty() && res.detected_position == res.expected_position);

    return res;
}

GroundingResult GroundingEvaluator::evaluate_sample_with_reference(
    const tensor::Tensor& image,
    const tensor::Tensor& reference,
    const std::string& prompt
) {
    GroundingResult res = evaluate_sample(image, prompt);
    res.psnr = compute_psnr(image, reference);
    res.ssim = compute_ssim(image, reference);
    return res;
}

EvaluationReport GroundingEvaluator::evaluate_pipeline(
    inference::DiffusionPipeline& pipeline,
    const std::vector<std::string>& prompts,
    const inference::SamplingConfig& config
) {
    EvaluationReport report;
    report.total_samples = prompts.size();
    if (prompts.empty()) return report;

    std::vector<tensor::Tensor> generated_images;
    generated_images.reserve(prompts.size());

    for (const auto& prompt : prompts) {
        tensor::Tensor img = pipeline.generate(prompt, config);
        generated_images.push_back(img);

        GroundingResult r = evaluate_sample(img, prompt);
        if (r.color_matched) report.color_correct++;
        if (r.shape_matched) report.shape_correct++;
        if (r.background_matched) report.background_correct++;
        if (r.position_matched) report.position_correct++;

        if (!r.expected_color.empty()) {
            report.color_confusion_matrix[r.expected_color][r.detected_color]++;
        }
        if (!r.expected_shape.empty()) {
            report.shape_confusion_matrix[r.expected_shape][r.detected_shape]++;
        }

        report.sample_results.push_back(r);
    }

    report.color_accuracy = static_cast<float_t>(report.color_correct) / static_cast<float_t>(report.total_samples);
    report.shape_accuracy = static_cast<float_t>(report.shape_correct) / static_cast<float_t>(report.total_samples);
    report.background_accuracy = static_cast<float_t>(report.background_correct) / static_cast<float_t>(report.total_samples);
    report.position_accuracy = static_cast<float_t>(report.position_correct) / static_cast<float_t>(report.total_samples);

    // Compute distribution metrics
    auto gen_stats = compute_image_distribution_stats(generated_images);
    report.avg_pfd = std::sqrt(gen_stats.std[0] * gen_stats.std[0] + 
                               gen_stats.std[1] * gen_stats.std[1] + 
                               gen_stats.std[2] * gen_stats.std[2]);

    return report;
}

EvaluationReport GroundingEvaluator::evaluate_pipeline_with_dataset(
    inference::DiffusionPipeline& pipeline,
    const dataset::Dataset& dataset,
    size_t max_samples,
    const inference::SamplingConfig& config
) {
    EvaluationReport report;
    size_t n = std::min(dataset.size(), max_samples);
    report.total_samples = n;
    if (n == 0) return report;

    std::vector<tensor::Tensor> gen_images;
    std::vector<tensor::Tensor> ref_images;
    gen_images.reserve(n);
    ref_images.reserve(n);

    double total_psnr = 0.0;
    double total_ssim = 0.0;
    double total_hist_sim = 0.0;

    for (size_t i = 0; i < n; ++i) {
        auto item = dataset.get(i);
        tensor::Tensor gen = pipeline.generate(item.caption, config);

        GroundingResult r = evaluate_sample_with_reference(gen, item.image, item.caption);
        if (r.color_matched) report.color_correct++;
        if (r.shape_matched) report.shape_correct++;
        if (r.background_matched) report.background_correct++;
        if (r.position_matched) report.position_correct++;

        if (!r.expected_color.empty()) {
            report.color_confusion_matrix[r.expected_color][r.detected_color]++;
        }
        if (!r.expected_shape.empty()) {
            report.shape_confusion_matrix[r.expected_shape][r.detected_shape]++;
        }

        total_psnr += static_cast<double>(r.psnr);
        total_ssim += static_cast<double>(r.ssim);

        auto h_gen = compute_color_histogram(gen, 16);
        auto h_ref = compute_color_histogram(item.image, 16);
        total_hist_sim += static_cast<double>(compute_histogram_intersection(h_gen, h_ref));

        report.sample_results.push_back(r);
        gen_images.push_back(std::move(gen));
        ref_images.push_back(std::move(item.image));
    }

    report.color_accuracy = static_cast<float_t>(report.color_correct) / static_cast<float_t>(report.total_samples);
    report.shape_accuracy = static_cast<float_t>(report.shape_correct) / static_cast<float_t>(report.total_samples);
    report.background_accuracy = static_cast<float_t>(report.background_correct) / static_cast<float_t>(report.total_samples);
    report.position_accuracy = static_cast<float_t>(report.position_correct) / static_cast<float_t>(report.total_samples);

    report.avg_psnr = static_cast<float_t>(total_psnr / static_cast<double>(n));
    report.avg_ssim = static_cast<float_t>(total_ssim / static_cast<double>(n));
    report.avg_histogram_intersection = static_cast<float_t>(total_hist_sim / static_cast<double>(n));

    // Fréchet distance between generated distribution and reference distribution
    auto gen_stats = compute_image_distribution_stats(gen_images);
    auto ref_stats = compute_image_distribution_stats(ref_images);
    report.avg_pfd = compute_frechet_distance(gen_stats, ref_stats);
    report.generated_images = std::move(gen_images);

    return report;
}

float_t GroundingEvaluator::evaluate_validation_loss(
    model::UNet& unet,
    text::TextEncoder& text_encoder,
    text::Tokenizer& tokenizer,
    diffusion::GaussianDiffusion& diffusion,
    dataset::DataLoader& dataloader,
    size_t max_batches
) {
    autodiff::NoGradGuard no_grad;
    unet.eval();
    text_encoder.eval();

    dataloader.reset();
    size_t batch_count = 0;
    double total_loss = 0.0;

    while (dataloader.has_next() && batch_count < max_batches) {
        auto batch = dataloader.next_batch();
        dim_t b = batch.batch_size;

        std::vector<dim_t> t_steps = diffusion.sample_timesteps(b, 1337 + batch_count);
        std::vector<float_t> t_floats(b);
        for (dim_t i = 0; i < b; ++i) t_floats[i] = static_cast<float_t>(t_steps[i]);

        tensor::Tensor noise = tensor::Tensor::randn(batch.images.shape(), 0.0f, 1.0f);
        tensor::Tensor xt = diffusion.q_sample(batch.images, t_steps, noise);

        auto text_enc = text_encoder.forward_batch(batch.captions, tokenizer);
        autodiff::Variable xt_var = autodiff::make_variable(std::move(xt), false, "xt");

        autodiff::Variable pred_noise = unet.forward(xt_var, t_floats, text_enc);

        // MSE
        float_t mse = compute_mse(pred_noise->data(), noise);
        total_loss += mse;
        batch_count++;
    }

    return (batch_count > 0) ? static_cast<float_t>(total_loss / static_cast<double>(batch_count)) : 0.0f;
}

std::string EvaluationReport::to_json_string() const {
    nlohmann::json j;
    j["total_samples"] = total_samples;
    j["color_correct"] = color_correct;
    j["color_accuracy"] = color_accuracy;
    j["shape_correct"] = shape_correct;
    j["shape_accuracy"] = shape_accuracy;
    j["background_correct"] = background_correct;
    j["background_accuracy"] = background_accuracy;
    j["position_correct"] = position_correct;
    j["position_accuracy"] = position_accuracy;
    j["avg_pfd"] = avg_pfd;
    j["avg_histogram_intersection"] = avg_histogram_intersection;
    j["avg_psnr"] = avg_psnr;
    j["avg_ssim"] = avg_ssim;

    nlohmann::json cm_color;
    for (const auto& [expected, detected_map] : color_confusion_matrix) {
        for (const auto& [detected, count] : detected_map) {
            cm_color[expected][detected] = count;
        }
    }
    j["color_confusion_matrix"] = cm_color;

    nlohmann::json cm_shape;
    for (const auto& [expected, detected_map] : shape_confusion_matrix) {
        for (const auto& [detected, count] : detected_map) {
            cm_shape[expected][detected] = count;
        }
    }
    j["shape_confusion_matrix"] = cm_shape;

    nlohmann::json samples = nlohmann::json::array();
    for (const auto& r : sample_results) {
        samples.push_back({
            {"prompt", r.prompt},
            {"expected_color", r.expected_color},
            {"detected_color", r.detected_color},
            {"color_matched", r.color_matched},
            {"expected_shape", r.expected_shape},
            {"detected_shape", r.detected_shape},
            {"shape_matched", r.shape_matched},
            {"expected_background", r.expected_background},
            {"detected_background", r.detected_background},
            {"background_matched", r.background_matched},
            {"expected_position", r.expected_position},
            {"detected_position", r.detected_position},
            {"position_matched", r.position_matched},
            {"psnr", r.psnr},
            {"ssim", r.ssim}
        });
    }
    j["samples"] = samples;

    return j.dump(2);
}

std::string EvaluationReport::to_summary_string() const {
    std::ostringstream oss;
    oss << "========================================================\n"
        << "  KODE Generative Evaluation Report\n"
        << "========================================================\n"
        << " Total Test Samples:    " << total_samples << "\n"
        << " Color Grounding Acc:   " << std::fixed << std::setprecision(1) 
        << (color_accuracy * 100.0f) << "% (" << color_correct << "/" << total_samples << ")\n"
        << " Shape Alignment Acc:   " << (shape_accuracy * 100.0f) << "% (" << shape_correct << "/" << total_samples << ")\n"
        << " Background Acc:        " << (background_accuracy * 100.0f) << "% (" << background_correct << "/" << total_samples << ")\n"
        << " Position Acc:          " << (position_accuracy * 100.0f) << "% (" << position_correct << "/" << total_samples << ")\n"
        << " Pixel Frechet Dist:    " << std::setprecision(4) << avg_pfd << "\n"
        << " Hist Intersection:     " << std::setprecision(4) << avg_histogram_intersection << "\n"
        << " Mean PSNR (dB):        " << std::setprecision(2) << avg_psnr << "\n"
        << " Mean SSIM:             " << std::setprecision(4) << avg_ssim << "\n"
        << "--------------------------------------------------------\n"
        << " Color Confusion Matrix:\n";

    for (const auto& [exp, det_map] : color_confusion_matrix) {
        oss << "  Target [" << std::setw(8) << exp << "] -> ";
        for (const auto& [det, cnt] : det_map) {
            oss << det << ":" << cnt << " ";
        }
        oss << "\n";
    }

    oss << "--------------------------------------------------------\n"
        << " Shape Confusion Matrix:\n";
    for (const auto& [exp, det_map] : shape_confusion_matrix) {
        oss << "  Target [" << std::setw(8) << exp << "] -> ";
        for (const auto& [det, cnt] : det_map) {
            oss << det << ":" << cnt << " ";
        }
        oss << "\n";
    }
    oss << "========================================================\n";
    return oss.str();
}

} // namespace kode::evaluation
