using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using PointF = System.Drawing.PointF;
using Point = System.Drawing.Point;
using Rectangle = System.Drawing.Rectangle;

namespace CameraMaui.RingCode
{
    /// <summary>
    /// Multi-ring code detection using Emgu.CV
    /// Uses CLAHE + Morphology + Multi-scale detection for robust results
    /// </summary>
    public class RingImageSegmentation
    {
        public class SegmentationResult
        {
            public List<RingRegion> DetectedRings { get; set; } = new();
            public bool Success { get; set; }
            public string Message { get; set; } = "";
            public Image<Gray, byte> ProcessedImage { get; set; }
        }

        public class RingRegion
        {
            public PointF Center { get; set; }
            public float OuterRadius { get; set; }
            public float InnerRadius { get; set; }
            public float MiddleRadius => (OuterRadius + InnerRadius) / 2;
            public Rectangle BoundingBox { get; set; }
            public Image<Gray, byte> CroppedImage { get; set; }
            public List<PointF> TrianglePoints { get; set; } = new();
            public double RotationAngle { get; set; }
            public int Index { get; set; }
            public double MatchScore { get; set; }
        }

        // Detection parameters - relaxed for better detection
        private readonly int _minRadius = 25;
        private readonly int _maxRadius = 500;  // Increased for single large ring
        private readonly double _circularityThreshold = 0.55; // More lenient

        // Logging
        public static Action<string> Log { get; set; } = (msg) => System.Diagnostics.Debug.WriteLine($"[Segmentation] {msg}");

        /// <summary>
        /// Main segmentation method
        /// </summary>
        public SegmentationResult SegmentImage(Image<Bgr, byte> sourceImage)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new SegmentationResult();

            try
            {
                var gray = sourceImage.Convert<Gray, byte>();
                Log($"Gray convert: {sw.ElapsedMilliseconds}ms");

                // Step 1: CLAHE for contrast normalization
                var normalized = ApplyCLAHE(gray);
                Log($"CLAHE: {sw.ElapsedMilliseconds}ms");

                // Step 2: Enhance edges with morphology
                var enhanced = EnhanceCircleEdges(normalized);
                Log($"Edge enhance: {sw.ElapsedMilliseconds}ms");

                // Step 3: Pyramid denoise
                var denoised = ApplyPyramidDenoise(enhanced);
                result.ProcessedImage = denoised;
                Log($"Pyramid denoise: {sw.ElapsedMilliseconds}ms");

                // Step 4: Multi-level detection
                result.DetectedRings = FindRingCodesMultiLevel(denoised, normalized, gray);
                Log($"Ring detection: {sw.ElapsedMilliseconds}ms, found {result.DetectedRings.Count}");

                // Step 5: Sort by position
                result.DetectedRings = SortRingsByPosition(result.DetectedRings);

                for (int i = 0; i < result.DetectedRings.Count; i++)
                {
                    result.DetectedRings[i].Index = i;
                }

                result.Success = result.DetectedRings.Count > 0;
                result.Message = result.Success ?
                    $"Found {result.DetectedRings.Count} ring code(s)" :
                    "No ring codes detected";

                Log($"Segmentation total: {sw.ElapsedMilliseconds}ms");
                return result;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message = $"Segmentation error: {ex.Message}";
                return result;
            }
        }

        /// <summary>
        /// Apply CLAHE for contrast normalization
        /// </summary>
        private Image<Gray, byte> ApplyCLAHE(Image<Gray, byte> source)
        {
            var result = new Image<Gray, byte>(source.Size);
            CvInvoke.CLAHE(source, 2.5, new System.Drawing.Size(8, 8), result);
            return result;
        }

        /// <summary>
        /// Enhance circle edges using morphological operations
        /// </summary>
        private Image<Gray, byte> EnhanceCircleEdges(Image<Gray, byte> source)
        {
            // Create circular structuring element for better circle edge detection
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));

            // Morphological gradient: dilation - erosion = edges
            var dilated = new Image<Gray, byte>(source.Size);
            var eroded = new Image<Gray, byte>(source.Size);

            CvInvoke.Dilate(source, dilated, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
            CvInvoke.Erode(source, eroded, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            // Combine: original + edge enhancement
            var edges = dilated.Sub(eroded);
            var enhanced = source.Add(edges.Mul(0.3));

            return enhanced.Convert<Gray, byte>();
        }

        /// <summary>
        /// Apply Image Pyramid denoising
        /// </summary>
        private Image<Gray, byte> ApplyPyramidDenoise(Image<Gray, byte> source)
        {
            var pyrDown = new Image<Gray, byte>(source.Width / 2, source.Height / 2);
            CvInvoke.PyrDown(source, pyrDown);

            var pyrUp = new Image<Gray, byte>(source.Size);
            CvInvoke.PyrUp(pyrDown, pyrUp);

            return pyrUp;
        }

        /// <summary>
        /// Fast detection with minimal methods
        /// </summary>
        private List<RingRegion> FindRingCodesMultiLevel(Image<Gray, byte> preprocessed,
            Image<Gray, byte> normalized, Image<Gray, byte> original)
        {
            var allRegions = new List<RingRegion>();

            // Method 1: Otsu threshold (fastest, most reliable)
            allRegions.AddRange(FindWithOtsuThreshold(preprocessed, original));

            // Method 2: Inverted Otsu for white rings with gray marks
            if (allRegions.Count == 0)
            {
                Log("Trying inverted threshold for white ring...");
                allRegions.AddRange(FindWithInvertedThreshold(preprocessed, original));
            }

            // Method 3: Single adaptive threshold (backup)
            if (allRegions.Count < 10)
            {
                allRegions.AddRange(FindWithAdaptiveThreshold(preprocessed, original,
                    AdaptiveThresholdType.GaussianC, 41, 4));
            }

            // Method 4: Single large ring detection using HoughCircles
            if (allRegions.Count == 0)
            {
                Log("Trying single large ring detection...");
                var singleRing = FindSingleLargeRing(preprocessed, original);
                if (singleRing != null)
                {
                    allRegions.Add(singleRing);
                }
            }

            // Merge with NMS
            var merged = ApplyNMS(allRegions, 0.5);

            return merged;
        }

        /// <summary>
        /// Find rings using inverted threshold (for white rings with gray marks)
        /// </summary>
        private List<RingRegion> FindWithInvertedThreshold(Image<Gray, byte> preprocessed, Image<Gray, byte> original)
        {
            var regions = new List<RingRegion>();

            // Invert the image
            var inverted = preprocessed.Not();

            var binary = new Image<Gray, byte>(inverted.Size);
            CvInvoke.Threshold(inverted, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            FindRingsInBinary(binary, original, regions);

            return regions;
        }

        /// <summary>
        /// Detect single large ring that fills most of the image
        /// Uses center-based analysis and HoughCircles
        /// </summary>
        private RingRegion FindSingleLargeRing(Image<Gray, byte> preprocessed, Image<Gray, byte> original)
        {
            try
            {
                int imgWidth = original.Width;
                int imgHeight = original.Height;
                int minDim = Math.Min(imgWidth, imgHeight);

                // Step 1: Apply Canny edge detection
                var edges = new Image<Gray, byte>(preprocessed.Size);
                CvInvoke.GaussianBlur(preprocessed, preprocessed, new System.Drawing.Size(5, 5), 1.5);
                CvInvoke.Canny(preprocessed, edges, 30, 90);

                // Step 2: Use HoughCircles to find circular patterns
                var circles = CvInvoke.HoughCircles(
                    preprocessed,
                    HoughModes.Gradient,
                    dp: 1.5,
                    minDist: minDim / 4,
                    param1: 100,
                    param2: 40,
                    minRadius: minDim / 8,
                    maxRadius: minDim / 2);

                if (circles.Length == 0)
                {
                    // Try with more lenient parameters
                    circles = CvInvoke.HoughCircles(
                        preprocessed,
                        HoughModes.Gradient,
                        dp: 2.0,
                        minDist: minDim / 4,
                        param1: 80,
                        param2: 30,
                        minRadius: minDim / 10,
                        maxRadius: (int)(minDim * 0.6));
                }

                if (circles.Length == 0)
                {
                    Log("HoughCircles found no circles");
                    return null;
                }

                // Find the largest circle near center
                float bestScore = 0;
                System.Drawing.PointF bestCenter = new System.Drawing.PointF(imgWidth / 2f, imgHeight / 2f);
                float bestRadius = 0;

                float imgCenterX = imgWidth / 2f;
                float imgCenterY = imgHeight / 2f;

                foreach (var circle in circles)
                {
                    float cx = circle.Center.X;
                    float cy = circle.Center.Y;
                    float r = circle.Radius;

                    // Score: prefer larger radius + closer to center
                    float distToCenter = (float)Math.Sqrt(Math.Pow(cx - imgCenterX, 2) + Math.Pow(cy - imgCenterY, 2));
                    float normalizedDist = distToCenter / minDim;
                    float score = r * (1.0f - normalizedDist * 0.5f);

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestCenter = new System.Drawing.PointF(cx, cy);
                        bestRadius = r;
                    }
                }

                if (bestRadius < minDim / 10)
                {
                    Log($"Best circle too small: {bestRadius}");
                    return null;
                }

                Log($"Found single ring: center=({bestCenter.X:F0},{bestCenter.Y:F0}), radius={bestRadius:F0}");

                // Validate pattern
                if (!ValidateRingCodePattern(original, bestCenter, bestRadius))
                {
                    Log("Pattern validation failed");
                    return null;
                }

                // Create region
                var region = new RingRegion
                {
                    Center = bestCenter,
                    OuterRadius = bestRadius,
                    InnerRadius = bestRadius * 0.35f,
                    BoundingBox = GetBoundingBox(bestCenter, bestRadius * 1.1f, original.Size),
                    MatchScore = CalculatePatternScore(original, bestCenter, bestRadius)
                };

                if (region.BoundingBox.Width > 0 && region.BoundingBox.Height > 0)
                {
                    original.ROI = region.BoundingBox;
                    region.CroppedImage = original.Clone();
                    original.ROI = Rectangle.Empty;

                    region.TrianglePoints = FindTriangleMarkers(original, region);
                    region.RotationAngle = CalculateRotationAngle(region);

                    return region;
                }
            }
            catch (Exception ex)
            {
                Log($"Single ring detection error: {ex.Message}");
            }

            return null;
        }

        /// <summary>
        /// Find rings using adaptive threshold
        /// </summary>
        private List<RingRegion> FindWithAdaptiveThreshold(Image<Gray, byte> preprocessed,
            Image<Gray, byte> original, AdaptiveThresholdType adaptiveType, int blockSize, double c)
        {
            var regions = new List<RingRegion>();

            var binary = new Image<Gray, byte>(preprocessed.Size);
            CvInvoke.AdaptiveThreshold(preprocessed, binary, 255, adaptiveType,
                ThresholdType.Binary, blockSize, c);

            // Apply morphological closing to connect broken circles
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Close, kernel,
                new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            FindRingsInBinary(binary, original, regions);

            return regions;
        }

        /// <summary>
        /// Find rings using Otsu threshold
        /// </summary>
        private List<RingRegion> FindWithOtsuThreshold(Image<Gray, byte> preprocessed, Image<Gray, byte> original)
        {
            var regions = new List<RingRegion>();

            var binary = new Image<Gray, byte>(preprocessed.Size);
            CvInvoke.Threshold(preprocessed, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            FindRingsInBinary(binary, original, regions);

            return regions;
        }

        /// <summary>
        /// Find rings using Canny edge detection
        /// </summary>
        private List<RingRegion> FindWithCannyEdge(Image<Gray, byte> preprocessed, Image<Gray, byte> original)
        {
            var regions = new List<RingRegion>();

            // Apply Canny edge detection
            var edges = new Image<Gray, byte>(preprocessed.Size);
            CvInvoke.Canny(preprocessed, edges, 50, 150);

            // Dilate to connect edges
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.Dilate(edges, edges, kernel, new Point(-1, -1), 2,
                BorderType.Default, new MCvScalar(0));

            FindRingsInBinary(edges, original, regions);

            return regions;
        }

        /// <summary>
        /// Find ring codes in binary image
        /// </summary>
        private void FindRingsInBinary(Image<Gray, byte> binary, Image<Gray, byte> original,
            List<RingRegion> regions)
        {
            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(binary, contours, hierarchy, RetrType.Tree,
                ChainApproxMethod.ChainApproxSimple);

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);
                double perimeter = CvInvoke.ArcLength(contour, true);

                if (perimeter < 1 || area < 400) continue;

                // Calculate circularity
                double circularity = 4 * Math.PI * area / (perimeter * perimeter);
                if (circularity < _circularityThreshold) continue;

                // Get center using moments
                var moments = CvInvoke.Moments(contour);
                if (moments.M00 < 1) continue;

                float cx = (float)(moments.M10 / moments.M00);
                float cy = (float)(moments.M01 / moments.M00);
                var center = new PointF(cx, cy);

                // Estimate radius from area
                float radius = (float)Math.Sqrt(area / Math.PI);

                // Check size constraints
                if (radius < _minRadius || radius > _maxRadius) continue;

                // Validate ring code pattern (more lenient)
                if (!ValidateRingCodePattern(original, center, radius))
                {
                    continue;
                }

                // Refine outer radius by analyzing radial profile
                // This fixes cases where inner circle is detected instead of outer boundary
                float refinedRadius = RefineOuterRadius(original, center, radius);

                // Calculate pattern score
                double patternScore = CalculatePatternScore(original, center, refinedRadius);

                var region = new RingRegion
                {
                    Center = center,
                    OuterRadius = refinedRadius,
                    InnerRadius = refinedRadius * 0.35f,
                    BoundingBox = GetBoundingBox(center, refinedRadius * 1.1f, original.Size),
                    MatchScore = patternScore
                };

                if (region.BoundingBox.Width > 0 && region.BoundingBox.Height > 0)
                {
                    original.ROI = region.BoundingBox;
                    region.CroppedImage = original.Clone();
                    original.ROI = Rectangle.Empty;

                    region.TrianglePoints = FindTriangleMarkers(original, region);
                    region.RotationAngle = CalculateRotationAngle(region);

                    regions.Add(region);
                }
            }
        }

        /// <summary>
        /// Validate ring code pattern - more lenient thresholds
        /// </summary>
        private bool ValidateRingCodePattern(Image<Gray, byte> source, PointF center, float radius)
        {
            float[] sampleRadii = { radius * 0.5f, radius * 0.65f, radius * 0.8f };
            int totalTransitions = 0;
            double totalVariance = 0;
            int validRadii = 0;

            foreach (float sampleRadius in sampleRadii)
            {
                var intensities = new List<int>();
                int sampleCount = 36;

                for (int i = 0; i < sampleCount; i++)
                {
                    double angle = i * (2 * Math.PI / sampleCount);
                    int x = (int)(center.X + sampleRadius * Math.Cos(angle));
                    int y = (int)(center.Y + sampleRadius * Math.Sin(angle));

                    if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                    {
                        intensities.Add(source.Data[y, x, 0]);
                    }
                }

                if (intensities.Count < 20) continue;

                double avg = intensities.Average();
                double variance = intensities.Sum(v => Math.Pow(v - avg, 2)) / intensities.Count;
                totalVariance += variance;

                // Count transitions
                double threshold = avg;
                bool lastWasHigh = intensities[0] > threshold;
                int transitions = 0;
                for (int j = 1; j < intensities.Count; j++)
                {
                    bool currentIsHigh = intensities[j] > threshold;
                    if (currentIsHigh != lastWasHigh)
                    {
                        transitions++;
                        lastWasHigh = currentIsHigh;
                    }
                }
                totalTransitions += transitions;
                validRadii++;
            }

            if (validRadii == 0) return false;

            double avgVariance = totalVariance / validRadii;
            double avgTransitions = totalTransitions / (double)validRadii;

            // More lenient criteria - just need some pattern
            // variance > 300 means there's black/white contrast
            // transitions > 3 means there are segments
            bool hasPattern = avgVariance > 300 && avgTransitions > 3;

            return hasPattern;
        }

        /// <summary>
        /// Calculate pattern score for ranking
        /// </summary>
        private double CalculatePatternScore(Image<Gray, byte> source, PointF center, float radius)
        {
            float sampleRadius = radius * 0.65f;
            var intensities = new List<int>();
            int sampleCount = 48;

            for (int i = 0; i < sampleCount; i++)
            {
                double angle = i * (2 * Math.PI / sampleCount);
                int x = (int)(center.X + sampleRadius * Math.Cos(angle));
                int y = (int)(center.Y + sampleRadius * Math.Sin(angle));

                if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                {
                    intensities.Add(source.Data[y, x, 0]);
                }
            }

            if (intensities.Count < 20) return 0;

            double avg = intensities.Average();
            double variance = intensities.Sum(v => Math.Pow(v - avg, 2)) / intensities.Count;

            int transitions = 0;
            bool lastWasHigh = intensities[0] > avg;
            for (int i = 1; i < intensities.Count; i++)
            {
                bool currentIsHigh = intensities[i] > avg;
                if (currentIsHigh != lastWasHigh)
                {
                    transitions++;
                    lastWasHigh = currentIsHigh;
                }
            }

            return Math.Sqrt(variance) * transitions;
        }

        /// <summary>
        /// Refine outer radius by finding the edge of data pattern
        /// Key insight: Find where pattern ENDS (transition from high variance to low variance)
        /// </summary>
        private float RefineOuterRadius(Image<Gray, byte> source, PointF center, float initialRadius)
        {
            int cx = (int)center.X;
            int cy = (int)center.Y;
            int numAngles = 24;

            // Step 1: Determine if initialRadius is likely the inner hole or data ring
            // Check if there's pattern at 1.0R, 1.5R, 2.0R
            float[] testMultiples = { 1.0f, 1.3f, 1.6f, 2.0f };
            float bestMultiple = 1.0f;
            double bestPatternScore = 0;

            foreach (float mult in testMultiples)
            {
                float testR = initialRadius * mult;
                if (testR > Math.Min(source.Width, source.Height) / 2f) continue;

                // Sample at 0.7 * testR (data ring area, not edge)
                float sampleR = testR * 0.70f;
                var intensities = new List<int>();

                for (int a = 0; a < 48; a++)
                {
                    double angle = a * (2 * Math.PI / 48);
                    int x = (int)(cx + sampleR * Math.Cos(angle));
                    int y = (int)(cy + sampleR * Math.Sin(angle));

                    if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                    {
                        intensities.Add(source.Data[y, x, 0]);
                    }
                }

                if (intensities.Count < 30) continue;

                // Calculate pattern score (variance * transitions)
                double mean = intensities.Average();
                double variance = intensities.Sum(v => Math.Pow(v - mean, 2)) / intensities.Count;

                int transitions = 0;
                bool lastHigh = intensities[0] > mean;
                for (int i = 1; i < intensities.Count; i++)
                {
                    bool curHigh = intensities[i] > mean;
                    if (curHigh != lastHigh)
                    {
                        transitions++;
                        lastHigh = curHigh;
                    }
                }

                double score = Math.Sqrt(variance) * transitions;
                Log($"    Test R={testR:F0} (mult={mult:F1}): variance={variance:F0}, transitions={transitions}, score={score:F0}");

                if (score > bestPatternScore)
                {
                    bestPatternScore = score;
                    bestMultiple = mult;
                }
            }

            float candidateRadius = initialRadius * bestMultiple;
            Log($"    Best multiple: {bestMultiple:F1} -> candidateR={candidateRadius:F0}");

            // Step 2: Fine-tune by finding outer edge (where pattern ends)
            // Scan from candidateR * 0.85 to candidateR * 1.15 to find exact boundary
            float[] outerEdgeEstimates = new float[numAngles];
            int validEstimates = 0;

            for (int a = 0; a < numAngles; a++)
            {
                double angle = a * (2 * Math.PI / numAngles);
                double cosA = Math.Cos(angle);
                double sinA = Math.Sin(angle);

                // Scan outward to find where pattern ends
                float lastPatternR = candidateRadius * 0.5f;

                for (float testR = candidateRadius * 0.6f; testR <= candidateRadius * 1.3f; testR += 3)
                {
                    // Check if there's pattern at this radius (high local variance)
                    var localIntensities = new List<int>();
                    for (float dr = -5; dr <= 5; dr += 2)
                    {
                        int x = (int)(cx + (testR + dr) * cosA);
                        int y = (int)(cy + (testR + dr) * sinA);
                        if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                        {
                            localIntensities.Add(source.Data[y, x, 0]);
                        }
                    }

                    if (localIntensities.Count < 3) continue;

                    double localMean = localIntensities.Average();
                    double localVariance = localIntensities.Sum(v => Math.Pow(v - localMean, 2)) / localIntensities.Count;

                    // Pattern exists if variance > 200 (not uniform)
                    if (localVariance > 200)
                    {
                        lastPatternR = testR;
                    }
                }

                // The outer edge is slightly beyond the last pattern location
                if (lastPatternR > candidateRadius * 0.6f)
                {
                    outerEdgeEstimates[a] = lastPatternR * 1.05f;  // Add 5% margin
                    validEstimates++;
                }
            }

            // Step 3: Use median of estimates
            if (validEstimates >= numAngles / 3)
            {
                var validRadii = outerEdgeEstimates.Where(r => r > 0).OrderBy(r => r).ToList();
                float medianRadius = validRadii[validRadii.Count / 2];

                // Sanity check
                if (medianRadius >= initialRadius * 0.9f && medianRadius <= initialRadius * 2.5f)
                {
                    Log($"    Radius refined: {initialRadius:F0} -> {medianRadius:F0} (from {validEstimates} edge samples)");
                    return medianRadius;
                }
            }

            Log($"    Using candidate radius: {candidateRadius:F0}");
            return candidateRadius;
        }

        /// <summary>
        /// Apply Non-Maximum Suppression + Remove nested rings
        /// </summary>
        private List<RingRegion> ApplyNMS(List<RingRegion> regions, double overlapThreshold)
        {
            if (regions.Count == 0) return regions;

            var sorted = regions.OrderByDescending(r => r.MatchScore).ToList();
            var result = new List<RingRegion>();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                result.Add(best);
                sorted.RemoveAt(0);

                sorted.RemoveAll(r =>
                {
                    double distance = Math.Sqrt(
                        Math.Pow(r.Center.X - best.Center.X, 2) +
                        Math.Pow(r.Center.Y - best.Center.Y, 2));
                    double maxDist = (r.OuterRadius + best.OuterRadius) * (1 - overlapThreshold);
                    return distance < maxDist;
                });
            }

            // Post-process: Remove smaller rings that are contained within larger rings
            result = RemoveNestedRings(result);

            return result;
        }

        /// <summary>
        /// Remove smaller rings that are completely contained within a larger ring
        /// This prevents false detections of code marks as separate rings
        /// </summary>
        private List<RingRegion> RemoveNestedRings(List<RingRegion> regions)
        {
            if (regions.Count <= 1) return regions;

            // Sort by radius descending (largest first)
            var sorted = regions.OrderByDescending(r => r.OuterRadius).ToList();
            var result = new List<RingRegion>();

            for (int i = 0; i < sorted.Count; i++)
            {
                var current = sorted[i];
                bool isNested = false;

                // Check if current ring is inside any larger ring in result
                foreach (var larger in result)
                {
                    double distance = Math.Sqrt(
                        Math.Pow(current.Center.X - larger.Center.X, 2) +
                        Math.Pow(current.Center.Y - larger.Center.Y, 2));

                    // If the smaller ring's center is within the larger ring's outer radius
                    // AND the smaller ring is significantly smaller (< 60% of larger)
                    // Then it's probably a false detection of a code mark
                    if (distance < larger.OuterRadius * 0.9 &&
                        current.OuterRadius < larger.OuterRadius * 0.6)
                    {
                        isNested = true;
                        Log($"Removed nested ring: r={current.OuterRadius:F0} inside r={larger.OuterRadius:F0}");
                        break;
                    }
                }

                if (!isNested)
                {
                    result.Add(current);
                }
            }

            return result;
        }

        /// <summary>
        /// Get bounding box
        /// </summary>
        private Rectangle GetBoundingBox(PointF center, float radius, System.Drawing.Size imageSize)
        {
            int x = Math.Max(0, (int)(center.X - radius));
            int y = Math.Max(0, (int)(center.Y - radius));
            int right = Math.Min(imageSize.Width, (int)(center.X + radius));
            int bottom = Math.Min(imageSize.Height, (int)(center.Y + radius));

            return new Rectangle(x, y, right - x, bottom - y);
        }

        /// <summary>
        /// Find arrow marker - returns empty, decoder will handle arrow detection with templates
        /// This avoids duplicate/conflicting detection logic
        /// </summary>
        private List<PointF> FindTriangleMarkers(Image<Gray, byte> source, RingRegion region)
        {
            // Arrow detection is handled by RingCodeDecoder with templates
            // Return empty list here to avoid conflicting results
            return new List<PointF>();
        }

        /// <summary>
        /// Calculate the angle at a vertex (in degrees)
        /// </summary>
        private double CalculateVertexAngle(Point p1, Point vertex, Point p2)
        {
            double v1x = p1.X - vertex.X;
            double v1y = p1.Y - vertex.Y;
            double v2x = p2.X - vertex.X;
            double v2y = p2.Y - vertex.Y;

            double dot = v1x * v2x + v1y * v2y;
            double mag1 = Math.Sqrt(v1x * v1x + v1y * v1y);
            double mag2 = Math.Sqrt(v2x * v2x + v2y * v2y);

            if (mag1 < 0.001 || mag2 < 0.001) return 180;

            double cosAngle = dot / (mag1 * mag2);
            cosAngle = Math.Max(-1, Math.Min(1, cosAngle));

            return Math.Acos(cosAngle) * 180 / Math.PI;
        }

        /// <summary>
        /// Calculate rotation angle
        /// </summary>
        private double CalculateRotationAngle(RingRegion region)
        {
            if (region.TrianglePoints.Count == 0)
                return 0;

            var mainTriangle = region.TrianglePoints.First();
            double angle = Math.Atan2(
                mainTriangle.Y - region.Center.Y,
                mainTriangle.X - region.Center.X
            ) * 180 / Math.PI;

            return angle;
        }

        /// <summary>
        /// Sort rings by position
        /// </summary>
        private List<RingRegion> SortRingsByPosition(List<RingRegion> regions)
        {
            if (regions.Count == 0) return regions;

            double rowTolerance = regions.Average(r => r.OuterRadius);

            return regions
                .OrderBy(r => Math.Floor(r.Center.Y / rowTolerance))
                .ThenBy(r => r.Center.X)
                .ToList();
        }
    }
}
