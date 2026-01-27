/**
 * Line2D Template Matching Implementation
 *
 * Based on meiqua/shape_based_matching with optimizations for Y-arrow detection.
 */

#include "line2Dup.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace line2Dup {

// Lookup table for orientation quantization (8 bins)
static const int QUANTIZATION_TABLE[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// Spread table: each bit in label expands to neighbors
static const unsigned char SPREAD_LUT[256] = {
    0, 3, 6, 7, 12, 15, 14, 15, 24, 27, 30, 31, 28, 31, 30, 31,
    48, 51, 54, 55, 60, 63, 62, 63, 56, 59, 62, 63, 60, 63, 62, 63,
    96, 99, 102, 103, 108, 111, 110, 111, 120, 123, 126, 127, 124, 127, 126, 127,
    112, 115, 118, 119, 124, 127, 126, 127, 120, 123, 126, 127, 124, 127, 126, 127,
    192, 195, 198, 199, 204, 207, 206, 207, 216, 219, 222, 223, 220, 223, 222, 223,
    240, 243, 246, 247, 252, 255, 254, 255, 248, 251, 254, 255, 252, 255, 254, 255,
    224, 227, 230, 231, 236, 239, 238, 239, 248, 251, 254, 255, 252, 255, 254, 255,
    240, 243, 246, 247, 252, 255, 254, 255, 248, 251, 254, 255, 252, 255, 254, 255,
    0, 3, 6, 7, 12, 15, 14, 15, 24, 27, 30, 31, 28, 31, 30, 31,
    48, 51, 54, 55, 60, 63, 62, 63, 56, 59, 62, 63, 60, 63, 62, 63,
    96, 99, 102, 103, 108, 111, 110, 111, 120, 123, 126, 127, 124, 127, 126, 127,
    112, 115, 118, 119, 124, 127, 126, 127, 120, 123, 126, 127, 124, 127, 126, 127,
    192, 195, 198, 199, 204, 207, 206, 207, 216, 219, 222, 223, 220, 223, 222, 223,
    240, 243, 246, 247, 252, 255, 254, 255, 248, 251, 254, 255, 252, 255, 254, 255,
    224, 227, 230, 231, 236, 239, 238, 239, 248, 251, 254, 255, 252, 255, 254, 255,
    240, 243, 246, 247, 252, 255, 254, 255, 248, 251, 254, 255, 252, 255, 254, 255
};

// ============================================================================
// ColorGradient Implementation
// ============================================================================

ColorGradient::ColorGradient(float weak_thresh, int num_feat, float strong_thresh)
    : weak_threshold(weak_thresh), num_features(num_feat), strong_threshold(strong_thresh) {}

void ColorGradient::quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude,
                                          cv::Mat& angle, float weak_threshold) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    // Compute gradients using Sobel
    cv::Mat dx, dy;
    cv::Sobel(gray, dx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, dy, CV_32F, 0, 1, 3);

    // Compute magnitude and angle
    magnitude.create(gray.size(), CV_32F);
    angle.create(gray.size(), CV_8U);

    for (int y = 0; y < gray.rows; ++y) {
        const float* dx_row = dx.ptr<float>(y);
        const float* dy_row = dy.ptr<float>(y);
        float* mag_row = magnitude.ptr<float>(y);
        uchar* ang_row = angle.ptr<uchar>(y);

        for (int x = 0; x < gray.cols; ++x) {
            float gx = dx_row[x];
            float gy = dy_row[x];
            float mag = std::sqrt(gx * gx + gy * gy);
            mag_row[x] = mag;

            if (mag >= weak_threshold) {
                // Quantize angle to 8 bins (0-7)
                float theta = std::atan2(gy, gx);
                if (theta < 0) theta += static_cast<float>(M_PI * 2);

                // Map [0, 2*PI) to [0, 8)
                int bin = static_cast<int>(theta * 4.0f / static_cast<float>(M_PI)) % 8;
                ang_row[x] = static_cast<uchar>(1 << bin);  // Store as bit mask
            } else {
                ang_row[x] = 0;
            }
        }
    }
}

void ColorGradient::spread(const cv::Mat& src, cv::Mat& dst, int T) {
    dst = cv::Mat::zeros(src.size(), CV_8U);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            uchar val = src.at<uchar>(y, x);
            if (val == 0) continue;

            // Spread to T x T neighborhood
            int y_start = std::max(0, y - T);
            int y_end = std::min(src.rows, y + T + 1);
            int x_start = std::max(0, x - T);
            int x_end = std::min(src.cols, x + T + 1);

            for (int yy = y_start; yy < y_end; ++yy) {
                uchar* dst_row = dst.ptr<uchar>(yy);
                for (int xx = x_start; xx < x_end; ++xx) {
                    dst_row[xx] |= val;
                }
            }
        }
    }
}

bool ColorGradient::extractTemplate(const cv::Mat& src, const cv::Mat& mask,
                                    Template& templ) const {
    cv::Mat magnitude, angle;
    quantizedOrientations(src, magnitude, angle, weak_threshold);

    // Extract features above strong threshold
    std::vector<std::pair<float, Feature>> candidates;

    cv::Rect roi(0, 0, src.cols, src.rows);
    if (!mask.empty()) {
        // Find bounding box of mask
        std::vector<cv::Point> points;
        cv::findNonZero(mask, points);
        if (points.empty()) return false;
        roi = cv::boundingRect(points);
    }

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const float* mag_row = magnitude.ptr<float>(y);
        const uchar* ang_row = angle.ptr<uchar>(y);
        const uchar* mask_row = mask.empty() ? nullptr : mask.ptr<uchar>(y);

        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (mask_row && mask_row[x] == 0) continue;
            if (ang_row[x] == 0) continue;

            float mag = mag_row[x];
            if (mag >= strong_threshold) {
                // Determine label from bitmask
                int label = 0;
                for (int b = 0; b < 8; ++b) {
                    if (ang_row[x] & (1 << b)) {
                        label = b;
                        break;
                    }
                }
                candidates.push_back({mag, Feature(x, y, label)});
            }
        }
    }

    if (candidates.empty()) return false;

    // Sort by magnitude and keep top features
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int n_feat = std::min(num_features, static_cast<int>(candidates.size()));

    // Compute center of features
    float cx = 0, cy = 0;
    for (int i = 0; i < n_feat; ++i) {
        cx += candidates[i].second.x;
        cy += candidates[i].second.y;
    }
    cx /= n_feat;
    cy /= n_feat;

    // Store features relative to center
    templ.features.clear();
    templ.features.reserve(n_feat);
    for (int i = 0; i < n_feat; ++i) {
        Feature f = candidates[i].second;
        f.x -= static_cast<int>(cx);
        f.y -= static_cast<int>(cy);
        templ.features.push_back(f);
    }

    templ.width = src.cols;
    templ.height = src.rows;
    templ.pyramid_level = 0;

    return true;
}

void ColorGradient::computeResponseMaps(const cv::Mat& src,
                                        std::vector<cv::Mat>& response_maps) const {
    cv::Mat magnitude, angle;
    quantizedOrientations(src, magnitude, angle, weak_threshold);

    // Spread orientations
    cv::Mat spread_angle;
    spread(angle, spread_angle, 4);

    // Create 8 response maps (one per orientation bin)
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i) {
        response_maps[i] = cv::Mat::zeros(src.size(), CV_8U);
        uchar bit = 1 << i;

        for (int y = 0; y < src.rows; ++y) {
            const uchar* spread_row = spread_angle.ptr<uchar>(y);
            uchar* resp_row = response_maps[i].ptr<uchar>(y);

            for (int x = 0; x < src.cols; ++x) {
                if (spread_row[x] & bit) {
                    resp_row[x] = 255;
                }
            }
        }
    }
}

// ============================================================================
// Detector Implementation
// ============================================================================

Detector::Detector(int num_features, int pyramid_levels)
    : modality_(30.0f, num_features, 60.0f),
      num_features_(num_features),
      pyramid_levels_(pyramid_levels) {}

int Detector::addTemplate(const cv::Mat& source, const std::string& class_id,
                          const cv::Mat& mask, float angle, float scale) {
    if (source.empty()) {
        setError("Empty source image");
        return -1;
    }

    cv::Mat gray;
    if (source.channels() == 3) {
        cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = source.clone();
    }

    Template templ;
    if (!modality_.extractTemplate(gray, mask, templ)) {
        setError("Failed to extract template features");
        return -1;
    }

    templ.angle = angle;
    templ.scale = scale;
    templ.class_id = class_id;

    int template_id = static_cast<int>(templates_.size());
    templates_.push_back(templ);
    class_templates_[class_id].push_back(template_id);

    return template_id;
}

int Detector::addTemplatesWithRotations(const cv::Mat& source, const std::string& class_id,
                                        const cv::Mat& mask,
                                        float angle_start, float angle_end, float angle_step) {
    int count = 0;
    cv::Point2f center(source.cols / 2.0f, source.rows / 2.0f);

    for (float angle = angle_start; angle < angle_end; angle += angle_step) {
        cv::Mat rotated = util::rotateImage(source, angle, center);
        cv::Mat rotated_mask;
        if (!mask.empty()) {
            rotated_mask = util::rotateImage(mask, angle, center);
        }

        int tid = addTemplate(rotated, class_id, rotated_mask, angle, 1.0f);
        if (tid >= 0) {
            count++;
        }
    }

    return count;
}

std::vector<Match> Detector::match(const cv::Mat& source, float threshold,
                                   const std::vector<std::string>& class_ids,
                                   const cv::Mat& mask) const {
    std::vector<Match> matches;

    if (templates_.empty()) {
        return matches;
    }

    cv::Mat gray;
    if (source.channels() == 3) {
        cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = source;
    }

    // Compute response maps
    std::vector<cv::Mat> response_maps;
    modality_.computeResponseMaps(gray, response_maps);

    // Determine which templates to search
    std::vector<int> template_ids;
    if (class_ids.empty()) {
        template_ids.resize(templates_.size());
        std::iota(template_ids.begin(), template_ids.end(), 0);
    } else {
        for (const auto& cid : class_ids) {
            auto it = class_templates_.find(cid);
            if (it != class_templates_.end()) {
                template_ids.insert(template_ids.end(),
                                    it->second.begin(), it->second.end());
            }
        }
    }

    // Match each template
    for (int tid : template_ids) {
        const Template& templ = templates_[tid];
        matchLevel(response_maps, templ, matches, threshold);
    }

    // Sort by score
    std::sort(matches.begin(), matches.end());

    // NMS
    util::nms(matches, 0.5f);

    return matches;
}

void Detector::matchLevel(const std::vector<cv::Mat>& response_maps,
                          const Template& templ,
                          std::vector<Match>& matches,
                          float threshold) const {
    if (templ.features.empty() || response_maps.empty()) {
        return;
    }

    int rows = response_maps[0].rows;
    int cols = response_maps[0].cols;

    // Compute template bounding box
    int min_x = INT_MAX, min_y = INT_MAX;
    int max_x = INT_MIN, max_y = INT_MIN;
    for (const auto& f : templ.features) {
        min_x = std::min(min_x, f.x);
        min_y = std::min(min_y, f.y);
        max_x = std::max(max_x, f.x);
        max_y = std::max(max_y, f.y);
    }

    int start_x = -min_x;
    int start_y = -min_y;
    int end_x = cols - max_x;
    int end_y = rows - max_y;

    int num_features = static_cast<int>(templ.features.size());
    int threshold_count = static_cast<int>(threshold * num_features);

    // Slide template over image
    for (int cy = start_y; cy < end_y; cy += T_) {
        for (int cx = start_x; cx < end_x; cx += T_) {
            int count = 0;

            // Check all features
            for (const auto& f : templ.features) {
                int x = cx + f.x;
                int y = cy + f.y;

                if (x < 0 || x >= cols || y < 0 || y >= rows) continue;

                if (response_maps[f.label].at<uchar>(y, x)) {
                    count++;
                }
            }

            if (count >= threshold_count) {
                float similarity = static_cast<float>(count) / num_features;
                matches.emplace_back(
                    static_cast<float>(cx),
                    static_cast<float>(cy),
                    templ.angle,
                    templ.scale,
                    similarity,
                    templ.class_id,
                    -1  // Template ID filled later
                );
            }
        }
    }
}

Match Detector::matchInRing(const cv::Mat& source,
                            cv::Point2f center,
                            float inner_radius,
                            float outer_radius,
                            float threshold,
                            const std::string& class_id) const {
    Match best_match;

    if (templates_.empty()) {
        return best_match;
    }

    // Create ring mask
    cv::Mat ring_mask = createRingMask(source.cols, source.rows,
                                       center, inner_radius, outer_radius);

    // Match with mask (simplified - just check if matches are in ring)
    std::vector<std::string> class_filter;
    if (!class_id.empty()) {
        class_filter.push_back(class_id);
    }

    std::vector<Match> all_matches = match(source, threshold, class_filter, ring_mask);

    // Filter matches to ring region and find best
    for (const auto& m : all_matches) {
        float dx = m.x - center.x;
        float dy = m.y - center.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist >= inner_radius && dist <= outer_radius) {
            if (m.similarity > best_match.similarity) {
                best_match = m;
            }
        }
    }

    return best_match;
}

cv::Mat Detector::createRingMask(int width, int height,
                                 cv::Point2f center,
                                 float inner_radius,
                                 float outer_radius) const {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8U);
    cv::circle(mask, cv::Point(static_cast<int>(center.x), static_cast<int>(center.y)),
               static_cast<int>(outer_radius), cv::Scalar(255), -1);
    cv::circle(mask, cv::Point(static_cast<int>(center.x), static_cast<int>(center.y)),
               static_cast<int>(inner_radius), cv::Scalar(0), -1);
    return mask;
}

int Detector::numTemplates() const {
    return static_cast<int>(templates_.size());
}

int Detector::numTemplates(const std::string& class_id) const {
    auto it = class_templates_.find(class_id);
    return it != class_templates_.end() ? static_cast<int>(it->second.size()) : 0;
}

const Template& Detector::getTemplate(int template_id) const {
    static Template empty;
    if (template_id < 0 || template_id >= static_cast<int>(templates_.size())) {
        return empty;
    }
    return templates_[template_id];
}

void Detector::clear() {
    templates_.clear();
    class_templates_.clear();
}

std::vector<std::string> Detector::classIds() const {
    std::vector<std::string> ids;
    for (const auto& pair : class_templates_) {
        ids.push_back(pair.first);
    }
    return ids;
}

bool Detector::saveTemplates(const std::string& file_path) const {
    cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }

    fs << "num_templates" << static_cast<int>(templates_.size());

    for (size_t i = 0; i < templates_.size(); ++i) {
        const Template& t = templates_[i];
        std::string name = "template_" + std::to_string(i);

        fs << name << "{";
        fs << "width" << t.width;
        fs << "height" << t.height;
        fs << "angle" << t.angle;
        fs << "scale" << t.scale;
        fs << "class_id" << t.class_id;

        // Save features
        fs << "features" << "[";
        for (const auto& f : t.features) {
            fs << "{" << "x" << f.x << "y" << f.y << "label" << f.label << "}";
        }
        fs << "]";

        fs << "}";
    }

    return true;
}

int Detector::loadTemplates(const std::string& file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        setError("Failed to open template file: " + file_path);
        return -1;
    }

    int num_templates = 0;
    fs["num_templates"] >> num_templates;

    int loaded = 0;
    for (int i = 0; i < num_templates; ++i) {
        std::string name = "template_" + std::to_string(i);
        cv::FileNode node = fs[name];
        if (node.empty()) continue;

        Template t;
        node["width"] >> t.width;
        node["height"] >> t.height;
        node["angle"] >> t.angle;
        node["scale"] >> t.scale;
        node["class_id"] >> t.class_id;

        cv::FileNode features_node = node["features"];
        for (const auto& fn : features_node) {
            Feature f;
            fn["x"] >> f.x;
            fn["y"] >> f.y;
            fn["label"] >> f.label;
            t.features.push_back(f);
        }

        int tid = static_cast<int>(templates_.size());
        templates_.push_back(t);
        class_templates_[t.class_id].push_back(tid);
        loaded++;
    }

    return loaded;
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace util {

cv::Mat rotateImage(const cv::Mat& src, float angle, cv::Point2f center) {
    if (center.x < 0 || center.y < 0) {
        center = cv::Point2f(src.cols / 2.0f, src.rows / 2.0f);
    }

    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, src.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    return dst;
}

cv::Mat createCircularMask(int width, int height, float radius) {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8U);
    cv::Point center(width / 2, height / 2);
    cv::circle(mask, center, static_cast<int>(radius), cv::Scalar(255), -1);
    return mask;
}

void nms(std::vector<Match>& matches, float overlap_thresh) {
    if (matches.empty()) return;

    // Sort by score (already done)
    std::vector<bool> suppressed(matches.size(), false);

    for (size_t i = 0; i < matches.size(); ++i) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < matches.size(); ++j) {
            if (suppressed[j]) continue;

            float dx = matches[i].x - matches[j].x;
            float dy = matches[i].y - matches[j].y;
            float dist = std::sqrt(dx * dx + dy * dy);

            // Suppress if too close
            if (dist < overlap_thresh * 50) {  // 50 is approximate template size
                suppressed[j] = true;
            }
        }
    }

    // Remove suppressed matches
    std::vector<Match> filtered;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!suppressed[i]) {
            filtered.push_back(matches[i]);
        }
    }
    matches = std::move(filtered);
}

void drawMatches(cv::Mat& img, const std::vector<Match>& matches,
                 const Detector& detector, cv::Scalar color) {
    for (const auto& m : matches) {
        cv::circle(img, cv::Point(static_cast<int>(m.x), static_cast<int>(m.y)),
                   5, color, 2);

        // Draw orientation line
        float angle_rad = m.angle * static_cast<float>(M_PI) / 180.0f;
        int dx = static_cast<int>(30 * std::cos(angle_rad));
        int dy = static_cast<int>(30 * std::sin(angle_rad));
        cv::line(img, cv::Point(static_cast<int>(m.x), static_cast<int>(m.y)),
                 cv::Point(static_cast<int>(m.x) + dx, static_cast<int>(m.y) + dy),
                 color, 2);
    }
}

}  // namespace util

}  // namespace line2Dup
