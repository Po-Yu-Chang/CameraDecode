using System.Drawing;

namespace CameraMaui.ShapeMatcher
{
    /// <summary>
    /// Result from shape-based template matching
    /// </summary>
    public class ShapeMatcherResult
    {
        /// <summary>
        /// Whether a match was found
        /// </summary>
        public bool IsFound { get; set; }

        /// <summary>
        /// Match center X coordinate
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Match center Y coordinate
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// Match center as PointF
        /// </summary>
        public PointF Center => new PointF(X, Y);

        /// <summary>
        /// Geometric angle from ring center to match position (0-360)
        /// </summary>
        public float Angle { get; set; }

        /// <summary>
        /// Template rotation angle - how much the template was rotated to match (0-360)
        /// </summary>
        public float TemplateAngle { get; set; }

        /// <summary>
        /// Match scale factor
        /// </summary>
        public float Scale { get; set; } = 1.0f;

        /// <summary>
        /// Match confidence score (0.0 - 1.0)
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Template ID that matched
        /// </summary>
        public int TemplateId { get; set; } = -1;

        /// <summary>
        /// Class identifier of matched template
        /// </summary>
        public string ClassId { get; set; } = "";

        /// <summary>
        /// Error message if matching failed
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Create an empty/failed result
        /// </summary>
        public static ShapeMatcherResult NotFound(string? error = null) => new()
        {
            IsFound = false,
            ErrorMessage = error
        };
    }

    /// <summary>
    /// Interface for shape-based template matching
    /// </summary>
    public interface IShapeBasedMatcher : IDisposable
    {
        /// <summary>
        /// Number of templates loaded
        /// </summary>
        int TemplateCount { get; }

        /// <summary>
        /// Whether matcher is initialized and ready
        /// </summary>
        bool IsReady { get; }

        /// <summary>
        /// Add a template with rotation variants
        /// </summary>
        /// <param name="templateImage">Grayscale template image data</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="classId">Class identifier (e.g., "y_arrow")</param>
        /// <param name="angleStart">Start angle for variants (degrees)</param>
        /// <param name="angleEnd">End angle for variants (degrees)</param>
        /// <param name="angleStep">Angle step between variants (degrees)</param>
        /// <returns>Number of templates added, or -1 on error</returns>
        int AddTemplateWithRotations(byte[] templateImage, int width, int height,
            string classId, float angleStart = 0f, float angleEnd = 360f, float angleStep = 15f);

        /// <summary>
        /// Add a single template at a specific angle
        /// </summary>
        int AddTemplate(byte[] templateImage, int width, int height, string classId, float angle = 0f);

        /// <summary>
        /// Find matches in an image
        /// </summary>
        /// <param name="searchImage">Grayscale search image data</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="threshold">Minimum match score (0.0 - 1.0)</param>
        /// <param name="classId">Class to match (null for all)</param>
        /// <param name="maxResults">Maximum results to return</param>
        /// <returns>Array of match results</returns>
        ShapeMatcherResult[] Match(byte[] searchImage, int width, int height,
            float threshold = 0.5f, string? classId = null, int maxResults = 10);

        /// <summary>
        /// Find arrow in a ring region (optimized for Y-arrow detection)
        /// </summary>
        /// <param name="searchImage">Grayscale search image data</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="centerX">Ring center X</param>
        /// <param name="centerY">Ring center Y</param>
        /// <param name="innerRadius">Inner search radius</param>
        /// <param name="outerRadius">Outer search radius</param>
        /// <param name="threshold">Minimum match score</param>
        /// <param name="classId">Class to match</param>
        /// <returns>Best match result</returns>
        ShapeMatcherResult FindArrowInRing(byte[] searchImage, int width, int height,
            float centerX, float centerY, float innerRadius, float outerRadius,
            float threshold = 0.5f, string classId = "y_arrow");

        /// <summary>
        /// Get number of templates for a specific class
        /// </summary>
        int GetClassTemplateCount(string classId);

        /// <summary>
        /// Save all templates to file
        /// </summary>
        bool SaveTemplates(string filePath);

        /// <summary>
        /// Load templates from file
        /// </summary>
        int LoadTemplates(string filePath);

        /// <summary>
        /// Clear all templates
        /// </summary>
        void ClearTemplates();

        /// <summary>
        /// Get last error message
        /// </summary>
        string? GetLastError();
    }
}
