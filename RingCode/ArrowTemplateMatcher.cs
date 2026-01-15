#if ANDROID || WINDOWS
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using PointF = System.Drawing.PointF;
using Point = System.Drawing.Point;

namespace CameraMaui.RingCode
{
    /// <summary>
    /// Arrow template matching using Emgu.CV
    /// Similar to HALCON's find_scaled_shape_model
    /// </summary>
    public class ArrowTemplateMatcher
    {
        private Image<Gray, byte> _template;
        private VectorOfPoint _templateContour;
        private double _templateArea;
        private string _templatePath;

        public bool IsLoaded => _template != null;
        public string TemplatePath => _templatePath;
        public Image<Gray, byte> Template => _template;

        /// <summary>
        /// Load arrow template from file
        /// </summary>
        public bool LoadTemplate(string path)
        {
            try
            {
                if (!File.Exists(path))
                    return false;

                var img = new Image<Gray, byte>(path);
                return LoadTemplateFromImage(img, path);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Load arrow template from image
        /// </summary>
        public bool LoadTemplateFromImage(Image<Gray, byte> templateImage, string sourcePath = null)
        {
            try
            {
                // Threshold to binary
                var binary = new Image<Gray, byte>(templateImage.Size);
                CvInvoke.Threshold(templateImage, binary, 128, 255, ThresholdType.Binary | ThresholdType.Otsu);

                // Find contours
                using var contours = new VectorOfVectorOfPoint();
                using var hierarchy = new Mat();
                CvInvoke.FindContours(binary, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                if (contours.Size == 0)
                {
                    System.Diagnostics.Debug.WriteLine("[ArrowTemplate] No contours found in template");
                    return false;
                }

                // Find largest contour (the arrow)
                double maxArea = 0;
                int maxIdx = 0;
                for (int i = 0; i < contours.Size; i++)
                {
                    double area = CvInvoke.ContourArea(contours[i]);
                    if (area > maxArea)
                    {
                        maxArea = area;
                        maxIdx = i;
                    }
                }

                _templateContour = new VectorOfPoint(contours[maxIdx].ToArray());
                _templateArea = maxArea;
                _template = binary;
                _templatePath = sourcePath;

                // Save template for debugging
                if (sourcePath == null)
                {
                    var debugPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "template_debug.png");
                    binary.Save(debugPath);
                    _templatePath = debugPath;
                    System.Diagnostics.Debug.WriteLine($"[ArrowTemplate] Template saved to: {debugPath}, size: {binary.Width}x{binary.Height}, area: {maxArea:F0}");
                }

                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ArrowTemplate] Error loading template: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Save current template to file
        /// </summary>
        public bool SaveTemplate(string path)
        {
            if (_template == null)
                return false;

            try
            {
                _template.Save(path);
                _templatePath = path;
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Find arrow in image using multi-angle template matching
        /// Similar to HALCON's find_scaled_shape_model
        /// </summary>
        public ArrowMatchResult FindArrow(Image<Gray, byte> image, double minScore = 0.3)
        {
            var result = new ArrowMatchResult();

            if (_template == null || _templateContour == null)
            {
                result.ErrorMessage = "Template not loaded";
                return result;
            }

            try
            {
                // Threshold input image
                var binary = new Image<Gray, byte>(image.Size);
                CvInvoke.Threshold(image, binary, 128, 255, ThresholdType.Binary | ThresholdType.Otsu);

                // Find contours in image
                using var contours = new VectorOfVectorOfPoint();
                using var hierarchy = new Mat();
                CvInvoke.FindContours(binary, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                double bestScore = 0;
                int bestIdx = -1;
                double bestAngle = 0;

                // Match each contour against template
                for (int i = 0; i < contours.Size; i++)
                {
                    var contour = contours[i];
                    double area = CvInvoke.ContourArea(contour);

                    // Filter by area (should be similar to template)
                    if (area < _templateArea * 0.3 || area > _templateArea * 3.0)
                        continue;

                    // Use contour matching (Hu moments)
                    double matchScore = CvInvoke.MatchShapes(_templateContour, contour, ContoursMatchType.I1, 0);

                    // Convert to similarity score (lower matchScore = better match)
                    double score = 1.0 / (1.0 + matchScore);

                    if (score > bestScore && score >= minScore)
                    {
                        bestScore = score;
                        bestIdx = i;

                        // Calculate angle using PCA or moments
                        bestAngle = CalculateContourAngle(contour);
                    }
                }

                if (bestIdx >= 0)
                {
                    var matchedContour = contours[bestIdx];
                    var moments = CvInvoke.Moments(matchedContour);

                    result.IsFound = true;
                    result.Score = bestScore;
                    result.Angle = bestAngle;
                    result.Center = new PointF(
                        (float)(moments.M10 / moments.M00),
                        (float)(moments.M01 / moments.M00));

                    // Find tip point (furthest from center)
                    var points = matchedContour.ToArray();
                    double maxDist = 0;
                    foreach (var pt in points)
                    {
                        double dist = Math.Sqrt(Math.Pow(pt.X - result.Center.X, 2) +
                                                Math.Pow(pt.Y - result.Center.Y, 2));
                        if (dist > maxDist)
                        {
                            maxDist = dist;
                            result.TipPoint = new PointF(pt.X, pt.Y);
                        }
                    }

                    // Store matched contour for visualization
                    result.MatchedContour = new VectorOfPoint(matchedContour.ToArray());
                }
                else
                {
                    result.ErrorMessage = "No matching arrow found";
                }

                return result;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = $"Error: {ex.Message}";
                return result;
            }
        }

        /// <summary>
        /// Find arrow using optimized rotation-invariant template matching
        /// Uses image pyramids for speed + coarse-to-fine angle search
        /// Based on best practices from PyImageSearch and OpenCV documentation
        /// </summary>
        public ArrowMatchResult FindArrowMultiAngle(Image<Gray, byte> image, double minScore = 0.3,
            double angleStart = 0, double angleEnd = 360, double angleStep = 15)
        {
            var bestResult = new ArrowMatchResult();

            if (_template == null)
            {
                bestResult.ErrorMessage = "Template not loaded";
                return bestResult;
            }

            try
            {
                // Step 1: Preprocess - convert to grayscale and normalize (already grayscale)
                var processedImage = PreprocessForMatching(image);

                // Step 2: Image pyramid approach - start with downscaled image for speed
                // Then refine at full resolution
                var pyramidScales = new[] { 0.25, 0.5, 1.0 };  // Coarse to fine
                double coarseBestAngle = 0;
                double coarseBestScale = 1.0;
                bool foundCoarse = false;

                // Multi-scale template matching with optimized scales
                double[] templateScales = { 0.6, 0.8, 1.0, 1.2, 1.5 };

                foreach (double pyramidScale in pyramidScales)
                {
                    // Downsample image for coarse search
                    Image<Gray, byte> searchImage;
                    if (pyramidScale < 1.0)
                    {
                        int newWidth = Math.Max(50, (int)(processedImage.Width * pyramidScale));
                        int newHeight = Math.Max(50, (int)(processedImage.Height * pyramidScale));
                        searchImage = new Image<Gray, byte>(newWidth, newHeight);
                        CvInvoke.Resize(processedImage, searchImage, new System.Drawing.Size(newWidth, newHeight));
                    }
                    else
                    {
                        searchImage = processedImage;
                    }

                    // Use narrower angle range if we found a coarse match
                    double searchAngleStart = angleStart;
                    double searchAngleEnd = angleEnd;
                    double searchAngleStep = angleStep;

                    if (foundCoarse && pyramidScale > 0.25)
                    {
                        // Refine around the coarse match
                        searchAngleStart = coarseBestAngle - 30;
                        searchAngleEnd = coarseBestAngle + 30;
                        searchAngleStep = 5;  // Finer step for refinement
                    }

                    foreach (double scale in templateScales)
                    {
                        // Skip if we're refining and this isn't near the best scale
                        if (foundCoarse && pyramidScale > 0.25)
                        {
                            if (Math.Abs(scale - coarseBestScale) > 0.3)
                                continue;
                        }

                        // Calculate effective template size for this pyramid level
                        double effectiveScale = scale * pyramidScale;
                        int scaledWidth = Math.Max(8, (int)(_template.Width * effectiveScale));
                        int scaledHeight = Math.Max(8, (int)(_template.Height * effectiveScale));

                        // Skip if template is larger than search image
                        if (scaledWidth >= searchImage.Width - 2 || scaledHeight >= searchImage.Height - 2)
                            continue;

                        var scaledTemplate = new Image<Gray, byte>(scaledWidth, scaledHeight);
                        CvInvoke.Resize(_template, scaledTemplate, new System.Drawing.Size(scaledWidth, scaledHeight));

                        // Try rotation angles
                        for (double angle = searchAngleStart; angle < searchAngleEnd; angle += searchAngleStep)
                        {
                            // Rotate template around its center
                            using var rotationMat = new Mat();
                            CvInvoke.GetRotationMatrix2D(new PointF(scaledWidth / 2f, scaledHeight / 2f),
                                angle, 1.0, rotationMat);

                            var rotatedTemplate = new Image<Gray, byte>(scaledTemplate.Size);
                            CvInvoke.WarpAffine(scaledTemplate, rotatedTemplate, rotationMat, scaledTemplate.Size,
                                Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

                            // Template matching with TM_CCOEFF_NORMED (best for this use case)
                            using var matchResult = new Mat();
                            CvInvoke.MatchTemplate(searchImage, rotatedTemplate, matchResult, TemplateMatchingType.CcoeffNormed);

                            // Find best match location
                            double minVal = 0, maxVal = 0;
                            Point minLoc = new Point(), maxLoc = new Point();
                            CvInvoke.MinMaxLoc(matchResult, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                            if (maxVal > bestResult.Score && maxVal >= minScore)
                            {
                                bestResult.IsFound = true;
                                bestResult.Score = maxVal;
                                bestResult.Angle = angle;

                                // Convert coordinates back to full image scale
                                if (pyramidScale < 1.0)
                                {
                                    bestResult.Center = new PointF(
                                        (maxLoc.X + scaledWidth / 2f) / (float)pyramidScale,
                                        (maxLoc.Y + scaledHeight / 2f) / (float)pyramidScale);
                                }
                                else
                                {
                                    bestResult.Center = new PointF(
                                        maxLoc.X + scaledWidth / 2f,
                                        maxLoc.Y + scaledHeight / 2f);
                                }

                                coarseBestAngle = angle;
                                coarseBestScale = scale;
                                foundCoarse = true;
                            }

                            rotatedTemplate.Dispose();
                        }

                        scaledTemplate.Dispose();
                    }

                    if (searchImage != processedImage)
                        searchImage.Dispose();
                }

                if (!bestResult.IsFound)
                {
                    bestResult.ErrorMessage = $"No matching arrow found (best score: {bestResult.Score:F2})";
                }

                return bestResult;
            }
            catch (Exception ex)
            {
                bestResult.ErrorMessage = $"Error: {ex.Message}";
                return bestResult;
            }
        }

        /// <summary>
        /// Fast arrow search within a known ring region (ROI-based)
        /// Much faster than full image search when ring location is known
        /// </summary>
        public ArrowMatchResult FindArrowInRingRegion(Image<Gray, byte> image, PointF ringCenter,
            float innerRadius, float outerRadius, double minScore = 0.4)
        {
            var bestResult = new ArrowMatchResult();

            if (_template == null)
            {
                bestResult.ErrorMessage = "Template not loaded";
                return bestResult;
            }

            try
            {
                // Define ROI as the ring area with some margin
                int margin = (int)(outerRadius * 0.1f);
                int roiX = Math.Max(0, (int)(ringCenter.X - outerRadius - margin));
                int roiY = Math.Max(0, (int)(ringCenter.Y - outerRadius - margin));
                int roiW = Math.Min((int)(outerRadius * 2 + margin * 2), image.Width - roiX);
                int roiH = Math.Min((int)(outerRadius * 2 + margin * 2), image.Height - roiY);

                if (roiW < _template.Width || roiH < _template.Height)
                {
                    bestResult.ErrorMessage = "Ring region too small for template";
                    return bestResult;
                }

                // Extract ROI
                image.ROI = new System.Drawing.Rectangle(roiX, roiY, roiW, roiH);
                var roiImage = image.Clone();
                image.ROI = System.Drawing.Rectangle.Empty;

                // Preprocess ROI
                var processedROI = PreprocessForMatching(roiImage);

                // Calculate expected template scale based on ring size
                double expectedArrowSize = (outerRadius - innerRadius) * 0.8;
                double baseTemplateSize = Math.Max(_template.Width, _template.Height);
                double expectedScale = expectedArrowSize / baseTemplateSize;

                // Use limited scale range around expected scale
                double[] scales = {
                    expectedScale * 0.7,
                    expectedScale * 0.85,
                    expectedScale,
                    expectedScale * 1.15,
                    expectedScale * 1.3
                };

                // Search all angles (arrow can be at any position)
                for (double angle = 0; angle < 360; angle += 10)
                {
                    foreach (double scale in scales)
                    {
                        if (scale < 0.2 || scale > 3.0) continue;

                        int scaledWidth = Math.Max(8, (int)(_template.Width * scale));
                        int scaledHeight = Math.Max(8, (int)(_template.Height * scale));

                        if (scaledWidth >= processedROI.Width - 2 || scaledHeight >= processedROI.Height - 2)
                            continue;

                        var scaledTemplate = new Image<Gray, byte>(scaledWidth, scaledHeight);
                        CvInvoke.Resize(_template, scaledTemplate, new System.Drawing.Size(scaledWidth, scaledHeight));

                        // Rotate template
                        using var rotationMat = new Mat();
                        CvInvoke.GetRotationMatrix2D(new PointF(scaledWidth / 2f, scaledHeight / 2f),
                            angle, 1.0, rotationMat);

                        var rotatedTemplate = new Image<Gray, byte>(scaledTemplate.Size);
                        CvInvoke.WarpAffine(scaledTemplate, rotatedTemplate, rotationMat, scaledTemplate.Size,
                            Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

                        // Match
                        using var matchResult = new Mat();
                        CvInvoke.MatchTemplate(processedROI, rotatedTemplate, matchResult, TemplateMatchingType.CcoeffNormed);

                        double minVal = 0, maxVal = 0;
                        Point minLoc = new Point(), maxLoc = new Point();
                        CvInvoke.MinMaxLoc(matchResult, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                        if (maxVal > bestResult.Score && maxVal >= minScore)
                        {
                            // Verify match is in the ring area (not center hole)
                            float matchCenterX = maxLoc.X + scaledWidth / 2f + roiX;
                            float matchCenterY = maxLoc.Y + scaledHeight / 2f + roiY;
                            float distFromRingCenter = (float)Math.Sqrt(
                                Math.Pow(matchCenterX - ringCenter.X, 2) +
                                Math.Pow(matchCenterY - ringCenter.Y, 2));

                            // Match must be in ring area (between inner and outer radius)
                            if (distFromRingCenter >= innerRadius * 0.8f && distFromRingCenter <= outerRadius * 1.1f)
                            {
                                bestResult.IsFound = true;
                                bestResult.Score = maxVal;
                                bestResult.Angle = angle;
                                bestResult.Center = new PointF(matchCenterX, matchCenterY);
                            }
                        }

                        scaledTemplate.Dispose();
                        rotatedTemplate.Dispose();
                    }
                }

                roiImage.Dispose();
                processedROI.Dispose();

                if (!bestResult.IsFound)
                {
                    bestResult.ErrorMessage = $"No arrow found in ring region (best: {bestResult.Score:F2})";
                }

                return bestResult;
            }
            catch (Exception ex)
            {
                bestResult.ErrorMessage = $"Error: {ex.Message}";
                return bestResult;
            }
        }

        /// <summary>
        /// Preprocess image for template matching - normalize to similar format as template
        /// </summary>
        private Image<Gray, byte> PreprocessForMatching(Image<Gray, byte> image)
        {
            // Apply same processing as template creation
            var result = new Image<Gray, byte>(image.Size);

            // Normalize histogram for consistent intensity
            CvInvoke.Normalize(image, result, 0, 255, NormType.MinMax);

            // Apply same threshold as template
            CvInvoke.Threshold(result, result, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            return result;
        }

        /// <summary>
        /// Find arrow using edge-based template matching (more robust to lighting variations)
        /// </summary>
        public ArrowMatchResult FindArrowEdgeBased(Image<Gray, byte> image, double minScore = 0.3,
            double angleStart = 0, double angleEnd = 360, double angleStep = 15)
        {
            var bestResult = new ArrowMatchResult();

            if (_template == null)
            {
                bestResult.ErrorMessage = "Template not loaded";
                return bestResult;
            }

            try
            {
                // Extract edges from input image
                var imageEdges = new Image<Gray, byte>(image.Size);
                CvInvoke.Canny(image, imageEdges, 50, 150);

                // Extract edges from template
                var templateEdges = new Image<Gray, byte>(_template.Size);
                CvInvoke.Canny(_template, templateEdges, 50, 150);

                // Multi-scale edge-based matching
                double[] scales = { 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 };

                foreach (double scale in scales)
                {
                    int scaledWidth = Math.Max(10, (int)(templateEdges.Width * scale));
                    int scaledHeight = Math.Max(10, (int)(templateEdges.Height * scale));

                    if (scaledWidth >= imageEdges.Width || scaledHeight >= imageEdges.Height)
                        continue;

                    var scaledTemplateEdges = new Image<Gray, byte>(scaledWidth, scaledHeight);
                    CvInvoke.Resize(templateEdges, scaledTemplateEdges, new System.Drawing.Size(scaledWidth, scaledHeight));

                    for (double angle = angleStart; angle < angleEnd; angle += angleStep)
                    {
                        var rotationMat = new Mat();
                        CvInvoke.GetRotationMatrix2D(new PointF(scaledWidth / 2f, scaledHeight / 2f),
                            angle, 1.0, rotationMat);

                        var rotatedEdges = new Image<Gray, byte>(scaledTemplateEdges.Size);
                        CvInvoke.WarpAffine(scaledTemplateEdges, rotatedEdges, rotationMat, scaledTemplateEdges.Size);

                        var matchResult = new Mat();
                        CvInvoke.MatchTemplate(imageEdges, rotatedEdges, matchResult, TemplateMatchingType.CcoeffNormed);

                        double minVal = 0, maxVal = 0;
                        Point minLoc = new Point(), maxLoc = new Point();
                        CvInvoke.MinMaxLoc(matchResult, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                        if (maxVal > bestResult.Score && maxVal >= minScore)
                        {
                            bestResult.IsFound = true;
                            bestResult.Score = maxVal;
                            bestResult.Angle = angle;
                            bestResult.Center = new PointF(
                                maxLoc.X + scaledWidth / 2f,
                                maxLoc.Y + scaledHeight / 2f);
                        }
                    }
                }

                if (!bestResult.IsFound)
                {
                    bestResult.ErrorMessage = $"No matching arrow found (edge-based, best: {bestResult.Score:F2})";
                }

                return bestResult;
            }
            catch (Exception ex)
            {
                bestResult.ErrorMessage = $"Error: {ex.Message}";
                return bestResult;
            }
        }

        /// <summary>
        /// Calculate contour orientation angle using image moments
        /// </summary>
        private double CalculateContourAngle(VectorOfPoint contour)
        {
            var moments = CvInvoke.Moments(contour);

            // Calculate orientation from central moments
            double mu20 = moments.Mu20;
            double mu02 = moments.Mu02;
            double mu11 = moments.Mu11;

            // Angle in radians
            double theta = 0.5 * Math.Atan2(2 * mu11, mu20 - mu02);

            // Convert to degrees
            return theta * 180 / Math.PI;
        }

        /// <summary>
        /// Create template from a region in an image
        /// </summary>
        public static Image<Gray, byte> ExtractArrowTemplate(Image<Gray, byte> source, PointF center, float radius)
        {
            int size = (int)(radius * 2);
            int x = Math.Max(0, (int)(center.X - radius));
            int y = Math.Max(0, (int)(center.Y - radius));
            int w = Math.Min(size, source.Width - x);
            int h = Math.Min(size, source.Height - y);

            source.ROI = new System.Drawing.Rectangle(x, y, w, h);
            var cropped = source.Clone();
            source.ROI = System.Drawing.Rectangle.Empty;

            // Resize to standard size
            var resized = new Image<Gray, byte>(100, 100);
            CvInvoke.Resize(cropped, resized, new System.Drawing.Size(100, 100));

            return resized;
        }
    }

    /// <summary>
    /// Arrow match result
    /// </summary>
    public class ArrowMatchResult
    {
        public bool IsFound { get; set; }
        public double Score { get; set; }
        public double Angle { get; set; }
        public PointF Center { get; set; }
        public PointF TipPoint { get; set; }
        public VectorOfPoint MatchedContour { get; set; }
        public string ErrorMessage { get; set; } = "";
    }
}
#endif
