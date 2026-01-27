/**
 * Line2D Template Matching
 *
 * Based on meiqua/shape_based_matching implementation.
 * Implements gradient-based template matching with SIMD acceleration.
 *
 * Reference: "Gradient Response Maps for Real-Time Detection of Textureless Objects"
 *            S. Hinterstoisser et al., IEEE TPAMI 2012
 */

#ifndef LINE2DUP_H
#define LINE2DUP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace line2Dup {

/**
 * Feature point with quantized gradient orientation
 */
struct Feature {
    int x;      // X coordinate
    int y;      // Y coordinate
    int label;  // Quantized gradient orientation (0-7 for 8 bins)

    Feature() : x(0), y(0), label(0) {}
    Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}
};

/**
 * Template containing features and metadata
 */
struct Template {
    int width;
    int height;
    int pyramid_level;
    std::vector<Feature> features;
    float angle;        // Template rotation angle
    float scale;        // Template scale
    std::string class_id;

    Template() : width(0), height(0), pyramid_level(0), angle(0), scale(1.0f) {}
};

/**
 * Match result with location, orientation, and score
 */
struct Match {
    float x;            // Match center X
    float y;            // Match center Y
    float angle;        // Match rotation angle
    float scale;        // Match scale
    float similarity;   // Match score (0.0 - 1.0)
    std::string class_id;
    int template_id;

    Match() : x(0), y(0), angle(0), scale(1.0f), similarity(0), template_id(-1) {}
    Match(float _x, float _y, float _angle, float _scale, float _sim,
          const std::string& _class_id, int _tid)
        : x(_x), y(_y), angle(_angle), scale(_scale), similarity(_sim),
          class_id(_class_id), template_id(_tid) {}

    bool operator<(const Match& other) const {
        return similarity > other.similarity;  // Descending order
    }
};

/**
 * Color gradient modality for template matching
 */
class ColorGradient {
public:
    ColorGradient(float weak_threshold = 30.0f, int num_features = 128,
                  float strong_threshold = 60.0f);

    /**
     * Extract features from a source image
     */
    bool extractTemplate(const cv::Mat& src, const cv::Mat& mask,
                         Template& templ) const;

    /**
     * Compute response maps for matching
     */
    void computeResponseMaps(const cv::Mat& src,
                             std::vector<cv::Mat>& response_maps) const;

    /**
     * Quantize gradient orientations
     */
    static void quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude,
                                      cv::Mat& angle, float weak_threshold);

    /**
     * Spread orientations for matching robustness
     */
    static void spread(const cv::Mat& src, cv::Mat& dst, int T);

    float weak_threshold;
    int num_features;
    float strong_threshold;

private:
    static const int GRADIENT_BINS = 8;
};

/**
 * Main detector class for shape-based matching
 */
class Detector {
public:
    /**
     * Constructor
     * @param num_features Number of gradient features per template
     * @param pyramid_levels Number of image pyramid levels (1 = no pyramid)
     */
    Detector(int num_features = 128, int pyramid_levels = 2);

    ~Detector() = default;

    /**
     * Add a template from an image
     * @param source Source grayscale image
     * @param class_id Class identifier for the template
     * @param mask Optional mask (255 = foreground)
     * @param angle Template angle in degrees
     * @param scale Template scale
     * @return Template ID, or -1 on failure
     */
    int addTemplate(const cv::Mat& source, const std::string& class_id,
                    const cv::Mat& mask = cv::Mat(),
                    float angle = 0.0f, float scale = 1.0f);

    /**
     * Add multiple rotated templates from a single source
     * @param source Source grayscale image
     * @param class_id Class identifier
     * @param mask Optional mask
     * @param angle_start Start angle (degrees)
     * @param angle_end End angle (degrees)
     * @param angle_step Angle step (degrees)
     * @return Number of templates added
     */
    int addTemplatesWithRotations(const cv::Mat& source, const std::string& class_id,
                                  const cv::Mat& mask,
                                  float angle_start, float angle_end, float angle_step);

    /**
     * Match templates in an image
     * @param source Source image to search
     * @param threshold Minimum similarity threshold (0.0 - 1.0)
     * @param class_ids Classes to match (empty for all)
     * @param mask Optional search mask
     * @return Vector of matches sorted by score
     */
    std::vector<Match> match(const cv::Mat& source, float threshold,
                             const std::vector<std::string>& class_ids = {},
                             const cv::Mat& mask = cv::Mat()) const;

    /**
     * Match in a ring region (optimized for arrow detection)
     * @param source Source image
     * @param center Ring center
     * @param inner_radius Inner search radius
     * @param outer_radius Outer search radius
     * @param threshold Minimum similarity threshold
     * @param class_id Class to match (empty for all)
     * @return Best match result
     */
    Match matchInRing(const cv::Mat& source,
                      cv::Point2f center,
                      float inner_radius,
                      float outer_radius,
                      float threshold,
                      const std::string& class_id = "") const;

    /**
     * Get number of templates
     */
    int numTemplates() const;

    /**
     * Get number of templates for a class
     */
    int numTemplates(const std::string& class_id) const;

    /**
     * Get template by ID
     */
    const Template& getTemplate(int template_id) const;

    /**
     * Clear all templates
     */
    void clear();

    /**
     * Save templates to file
     */
    bool saveTemplates(const std::string& file_path) const;

    /**
     * Load templates from file
     */
    int loadTemplates(const std::string& file_path);

    /**
     * Get class IDs
     */
    std::vector<std::string> classIds() const;

    /**
     * Set/get last error message
     */
    const std::string& lastError() const { return last_error_; }
    void setError(const std::string& msg) { last_error_ = msg; }

private:
    /**
     * Match templates at a single pyramid level
     */
    void matchLevel(const std::vector<cv::Mat>& response_maps,
                    const Template& templ,
                    std::vector<Match>& matches,
                    float threshold) const;

    /**
     * Create ring mask
     */
    cv::Mat createRingMask(int width, int height,
                           cv::Point2f center,
                           float inner_radius,
                           float outer_radius) const;

    ColorGradient modality_;
    int num_features_;
    int pyramid_levels_;
    std::vector<Template> templates_;
    std::map<std::string, std::vector<int>> class_templates_;
    mutable std::string last_error_;

    // T parameter for similarity computation (spread width)
    static const int T_ = 4;
};

/**
 * Utility functions
 */
namespace util {

/**
 * Rotate image around center
 */
cv::Mat rotateImage(const cv::Mat& src, float angle, cv::Point2f center = cv::Point2f(-1, -1));

/**
 * Create circular mask
 */
cv::Mat createCircularMask(int width, int height, float radius);

/**
 * Non-maximum suppression for matches
 */
void nms(std::vector<Match>& matches, float overlap_thresh);

/**
 * Draw matches on image
 */
void drawMatches(cv::Mat& img, const std::vector<Match>& matches,
                 const Detector& detector, cv::Scalar color = cv::Scalar(0, 255, 0));

}  // namespace util

}  // namespace line2Dup

#endif  // LINE2DUP_H
