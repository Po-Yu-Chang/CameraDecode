/**
 * Shape-Based Matcher C API Implementation
 */

#include "shape_matcher_c_api.h"
#include "line2Dup.h"

#include <string>
#include <cstring>
#include <memory>

#define SHAPEMATCHER_VERSION "1.0.0"

/**
 * Internal wrapper structure
 */
struct ShapeMatcherContext {
    std::unique_ptr<line2Dup::Detector> detector;
    std::string last_error;

    ShapeMatcherContext(int num_features, float weak_thresh, float strong_thresh)
        : detector(std::make_unique<line2Dup::Detector>(num_features, 2)) {
        // Note: ColorGradient thresholds are set internally
    }
};

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

SHAPEMATCHER_API ShapeMatcherHandle ShapeMatcher_Create(
    int num_features,
    float weak_thresh,
    float strong_thresh
) {
    try {
        auto* ctx = new ShapeMatcherContext(
            num_features > 0 ? num_features : 128,
            weak_thresh > 0 ? weak_thresh : 30.0f,
            strong_thresh > 0 ? strong_thresh : 60.0f
        );
        return static_cast<ShapeMatcherHandle>(ctx);
    } catch (...) {
        return nullptr;
    }
}

SHAPEMATCHER_API void ShapeMatcher_Destroy(ShapeMatcherHandle handle) {
    if (handle) {
        auto* ctx = static_cast<ShapeMatcherContext*>(handle);
        delete ctx;
    }
}

SHAPEMATCHER_API int ShapeMatcher_AddTemplateWithRotations(
    ShapeMatcherHandle handle,
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    const uint8_t* mask_data,
    const char* class_id,
    float angle_start,
    float angle_end,
    float angle_step
) {
    if (!handle || !image_data || width <= 0 || height <= 0) {
        return -1;
    }

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);

    try {
        // Create cv::Mat from raw data
        cv::Mat image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            std::memcpy(image.ptr(y), image_data + y * stride, width);
        }

        cv::Mat mask;
        if (mask_data) {
            mask.create(height, width, CV_8UC1);
            for (int y = 0; y < height; ++y) {
                std::memcpy(mask.ptr(y), mask_data + y * stride, width);
            }
        }

        std::string cid = class_id ? class_id : "default";

        int count = ctx->detector->addTemplatesWithRotations(
            image, cid, mask, angle_start, angle_end, angle_step
        );

        return count;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    } catch (...) {
        ctx->last_error = "Unknown error in AddTemplateWithRotations";
        return -1;
    }
}

SHAPEMATCHER_API int ShapeMatcher_AddTemplate(
    ShapeMatcherHandle handle,
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    const uint8_t* mask_data,
    const char* class_id,
    float angle
) {
    if (!handle || !image_data || width <= 0 || height <= 0) {
        return -1;
    }

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);

    try {
        cv::Mat image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            std::memcpy(image.ptr(y), image_data + y * stride, width);
        }

        cv::Mat mask;
        if (mask_data) {
            mask.create(height, width, CV_8UC1);
            for (int y = 0; y < height; ++y) {
                std::memcpy(mask.ptr(y), mask_data + y * stride, width);
            }
        }

        std::string cid = class_id ? class_id : "default";

        int tid = ctx->detector->addTemplate(image, cid, mask, angle, 1.0f);
        return tid;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    } catch (...) {
        ctx->last_error = "Unknown error in AddTemplate";
        return -1;
    }
}

SHAPEMATCHER_API int ShapeMatcher_Match(
    ShapeMatcherHandle handle,
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    const uint8_t* mask_data,
    float threshold,
    const char* class_id,
    ShapeMatchResult* results,
    int max_results
) {
    if (!handle || !image_data || width <= 0 || height <= 0 || !results || max_results <= 0) {
        return -1;
    }

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);

    try {
        cv::Mat image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            std::memcpy(image.ptr(y), image_data + y * stride, width);
        }

        cv::Mat mask;
        if (mask_data) {
            mask.create(height, width, CV_8UC1);
            for (int y = 0; y < height; ++y) {
                std::memcpy(mask.ptr(y), mask_data + y * stride, width);
            }
        }

        std::vector<std::string> class_filter;
        if (class_id && strlen(class_id) > 0) {
            class_filter.push_back(class_id);
        }

        auto matches = ctx->detector->match(image, threshold, class_filter, mask);

        int count = std::min(static_cast<int>(matches.size()), max_results);
        for (int i = 0; i < count; ++i) {
            const auto& m = matches[i];
            results[i].x = m.x;
            results[i].y = m.y;
            results[i].angle = m.angle;
            results[i].scale = m.scale;
            results[i].score = m.similarity;
            results[i].template_id = m.template_id;

            // Copy class_id
            strncpy(results[i].class_id, m.class_id.c_str(), sizeof(results[i].class_id) - 1);
            results[i].class_id[sizeof(results[i].class_id) - 1] = '\0';
            results[i].class_id_len = static_cast<int>(m.class_id.length());
        }

        return count;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    } catch (...) {
        ctx->last_error = "Unknown error in Match";
        return -1;
    }
}

SHAPEMATCHER_API int ShapeMatcher_MatchInRing(
    ShapeMatcherHandle handle,
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    float center_x,
    float center_y,
    float inner_radius,
    float outer_radius,
    float threshold,
    const char* class_id,
    ShapeMatchResult* result
) {
    if (!handle || !image_data || width <= 0 || height <= 0 || !result) {
        return -1;
    }

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);

    try {
        cv::Mat image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            std::memcpy(image.ptr(y), image_data + y * stride, width);
        }

        std::string cid = class_id ? class_id : "";

        auto match = ctx->detector->matchInRing(
            image,
            cv::Point2f(center_x, center_y),
            inner_radius,
            outer_radius,
            threshold,
            cid
        );

        if (match.similarity > 0) {
            result->x = match.x;
            result->y = match.y;
            result->angle = match.angle;
            result->scale = match.scale;
            result->score = match.similarity;
            result->template_id = match.template_id;

            strncpy(result->class_id, match.class_id.c_str(), sizeof(result->class_id) - 1);
            result->class_id[sizeof(result->class_id) - 1] = '\0';
            result->class_id_len = static_cast<int>(match.class_id.length());

            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    } catch (...) {
        ctx->last_error = "Unknown error in MatchInRing";
        return -1;
    }
}

SHAPEMATCHER_API int ShapeMatcher_GetTemplateCount(ShapeMatcherHandle handle) {
    if (!handle) return -1;

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);
    return ctx->detector->numTemplates();
}

SHAPEMATCHER_API int ShapeMatcher_GetClassTemplateCount(
    ShapeMatcherHandle handle,
    const char* class_id
) {
    if (!handle || !class_id) return -1;

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);
    return ctx->detector->numTemplates(class_id);
}

SHAPEMATCHER_API int ShapeMatcher_SaveTemplates(
    ShapeMatcherHandle handle,
    const char* file_path
) {
    if (!handle || !file_path) return 0;

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);
    return ctx->detector->saveTemplates(file_path) ? 1 : 0;
}

SHAPEMATCHER_API int ShapeMatcher_LoadTemplates(
    ShapeMatcherHandle handle,
    const char* file_path
) {
    if (!handle || !file_path) return -1;

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);
    return ctx->detector->loadTemplates(file_path);
}

SHAPEMATCHER_API void ShapeMatcher_ClearTemplates(ShapeMatcherHandle handle) {
    if (!handle) return;

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);
    ctx->detector->clear();
}

SHAPEMATCHER_API int ShapeMatcher_GetLastError(
    ShapeMatcherHandle handle,
    char* buffer,
    int buffer_size
) {
    if (!handle || !buffer || buffer_size <= 0) return 0;

    auto* ctx = static_cast<ShapeMatcherContext*>(handle);
    if (ctx->last_error.empty()) {
        buffer[0] = '\0';
        return 0;
    }

    int len = std::min(static_cast<int>(ctx->last_error.length()), buffer_size - 1);
    std::memcpy(buffer, ctx->last_error.c_str(), len);
    buffer[len] = '\0';
    return len;
}

SHAPEMATCHER_API const char* ShapeMatcher_GetVersion(void) {
    return SHAPEMATCHER_VERSION;
}

}  // extern "C"
