/**
 * Shape-Based Matcher C API
 *
 * C interface wrapper for the shape-based matching library.
 * Based on meiqua/shape_based_matching line2Dup algorithm.
 *
 * Usage:
 *   1. Create matcher: ShapeMatcher_Create()
 *   2. Add templates: ShapeMatcher_AddTemplateWithRotations()
 *   3. Match: ShapeMatcher_Match()
 *   4. Destroy: ShapeMatcher_Destroy()
 */

#ifndef SHAPE_MATCHER_C_API_H
#define SHAPE_MATCHER_C_API_H

#include <stdint.h>

#ifdef _WIN32
    #ifdef SHAPEMATCHER_EXPORTS
        #define SHAPEMATCHER_API __declspec(dllexport)
    #else
        #define SHAPEMATCHER_API __declspec(dllimport)
    #endif
#else
    #define SHAPEMATCHER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the shape matcher instance
typedef void* ShapeMatcherHandle;

/**
 * Match result structure
 */
typedef struct {
    float x;            // Match center X coordinate
    float y;            // Match center Y coordinate
    float angle;        // Match rotation angle in degrees (0-360)
    float scale;        // Match scale factor
    float score;        // Match score (0.0 - 1.0)
    int template_id;    // ID of matched template
    int class_id_len;   // Length of class_id string
    char class_id[64];  // Class identifier (null-terminated)
} ShapeMatchResult;

/**
 * Create a new shape matcher instance.
 *
 * @param num_features Number of gradient features to extract (default: 128)
 * @param weak_thresh Weak gradient threshold (default: 30.0)
 * @param strong_thresh Strong gradient threshold (default: 60.0)
 * @return Handle to the created matcher, or NULL on failure
 */
SHAPEMATCHER_API ShapeMatcherHandle ShapeMatcher_Create(
    int num_features,
    float weak_thresh,
    float strong_thresh
);

/**
 * Destroy a shape matcher instance and free resources.
 *
 * @param handle Handle to the matcher to destroy
 */
SHAPEMATCHER_API void ShapeMatcher_Destroy(ShapeMatcherHandle handle);

/**
 * Add a template image with multiple rotation variants.
 *
 * @param handle Matcher handle
 * @param image_data Grayscale image data (8-bit, row-major)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param stride Row stride in bytes (usually equals width for grayscale)
 * @param mask_data Optional binary mask (NULL for no mask, 255=valid, 0=ignore)
 * @param class_id Class identifier string (e.g., "y_arrow")
 * @param angle_start Start angle for rotation variants (degrees)
 * @param angle_end End angle for rotation variants (degrees)
 * @param angle_step Angle step between variants (degrees)
 * @return Number of templates added, or -1 on error
 */
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
);

/**
 * Add a single template at a specific angle.
 *
 * @param handle Matcher handle
 * @param image_data Grayscale image data
 * @param width Image width
 * @param height Image height
 * @param stride Row stride
 * @param mask_data Optional mask
 * @param class_id Class identifier
 * @param angle Template angle in degrees
 * @return Template ID on success, -1 on error
 */
SHAPEMATCHER_API int ShapeMatcher_AddTemplate(
    ShapeMatcherHandle handle,
    const uint8_t* image_data,
    int width,
    int height,
    int stride,
    const uint8_t* mask_data,
    const char* class_id,
    float angle
);

/**
 * Find matches in an image.
 *
 * @param handle Matcher handle
 * @param image_data Grayscale search image data
 * @param width Image width
 * @param height Image height
 * @param stride Row stride
 * @param mask_data Optional search mask (NULL for full image)
 * @param threshold Minimum match score threshold (0.0 - 1.0)
 * @param class_id Class to match (NULL or "" for all classes)
 * @param results Array to store match results
 * @param max_results Maximum number of results to return
 * @return Number of matches found, or -1 on error
 */
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
);

/**
 * Find best match in a ring region (optimized for Y-arrow detection).
 *
 * @param handle Matcher handle
 * @param image_data Grayscale image data
 * @param width Image width
 * @param height Image height
 * @param stride Row stride
 * @param center_x Ring center X coordinate
 * @param center_y Ring center Y coordinate
 * @param inner_radius Inner radius of search region
 * @param outer_radius Outer radius of search region
 * @param threshold Minimum match score threshold
 * @param class_id Class to match
 * @param result Single result output
 * @return 1 if match found, 0 if no match, -1 on error
 */
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
);

/**
 * Get the number of loaded templates.
 *
 * @param handle Matcher handle
 * @return Number of templates, or -1 on error
 */
SHAPEMATCHER_API int ShapeMatcher_GetTemplateCount(ShapeMatcherHandle handle);

/**
 * Get number of templates for a specific class.
 *
 * @param handle Matcher handle
 * @param class_id Class identifier
 * @return Number of templates for the class, or -1 on error
 */
SHAPEMATCHER_API int ShapeMatcher_GetClassTemplateCount(
    ShapeMatcherHandle handle,
    const char* class_id
);

/**
 * Save all templates to a file.
 *
 * @param handle Matcher handle
 * @param file_path Output file path
 * @return 1 on success, 0 on failure
 */
SHAPEMATCHER_API int ShapeMatcher_SaveTemplates(
    ShapeMatcherHandle handle,
    const char* file_path
);

/**
 * Load templates from a file.
 *
 * @param handle Matcher handle
 * @param file_path Input file path
 * @return Number of templates loaded, or -1 on error
 */
SHAPEMATCHER_API int ShapeMatcher_LoadTemplates(
    ShapeMatcherHandle handle,
    const char* file_path
);

/**
 * Clear all templates.
 *
 * @param handle Matcher handle
 */
SHAPEMATCHER_API void ShapeMatcher_ClearTemplates(ShapeMatcherHandle handle);

/**
 * Get the last error message.
 *
 * @param handle Matcher handle
 * @param buffer Output buffer for error message
 * @param buffer_size Size of buffer
 * @return Length of error message, or 0 if no error
 */
SHAPEMATCHER_API int ShapeMatcher_GetLastError(
    ShapeMatcherHandle handle,
    char* buffer,
    int buffer_size
);

/**
 * Get library version string.
 *
 * @return Version string (e.g., "1.0.0")
 */
SHAPEMATCHER_API const char* ShapeMatcher_GetVersion(void);

#ifdef __cplusplus
}
#endif

#endif // SHAPE_MATCHER_C_API_H
