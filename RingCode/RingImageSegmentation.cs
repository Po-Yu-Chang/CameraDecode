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
        private readonly int _maxRadius = 150;
        private readonly double _circularityThreshold = 0.55; // More lenient

        /// <summary>
        /// Main segmentation method
        /// </summary>
        public SegmentationResult SegmentImage(Image<Bgr, byte> sourceImage)
        {
            var result = new SegmentationResult();

            try
            {
                var gray = sourceImage.Convert<Gray, byte>();

                // Step 1: CLAHE for contrast normalization
                var normalized = ApplyCLAHE(gray);

                // Step 2: Enhance edges with morphology
                var enhanced = EnhanceCircleEdges(normalized);

                // Step 3: Pyramid denoise
                var denoised = ApplyPyramidDenoise(enhanced);
                result.ProcessedImage = denoised;

                // Step 4: Multi-level detection
                result.DetectedRings = FindRingCodesMultiLevel(denoised, normalized, gray);

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

            // Method 2: Single adaptive threshold (backup)
            if (allRegions.Count < 10)
            {
                allRegions.AddRange(FindWithAdaptiveThreshold(preprocessed, original,
                    AdaptiveThresholdType.GaussianC, 41, 4));
            }

            // Merge with NMS
            var merged = ApplyNMS(allRegions, 0.5);

            return merged;
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
        /// Refine outer radius - fast method
        /// If detected radius is too small, expand to find outer boundary
        /// </summary>
        private float RefineOuterRadius(Image<Gray, byte> source, PointF center, float initialRadius)
        {
            // Quick check: if radius seems reasonable (> 45px), keep it
            if (initialRadius > 45) return initialRadius;

            // For small detected radius, check if there's pattern at 2x radius
            int cx = (int)center.X;
            int cy = (int)center.Y;
            float testRadius = initialRadius * 2;

            // Sample 12 points at 2x radius
            int patternCount = 0;
            for (int i = 0; i < 12; i++)
            {
                double angle = i * Math.PI / 6;
                int x = (int)(cx + testRadius * Math.Cos(angle));
                int y = (int)(cy + testRadius * Math.Sin(angle));

                if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                {
                    // Check if pixel has significant intensity variation from neighbors
                    int val = source.Data[y, x, 0];
                    if (val > 80 && val < 200) patternCount++;
                }
            }

            // If pattern found at 2x radius, use it
            return patternCount >= 6 ? testRadius : initialRadius;
        }

        /// <summary>
        /// Apply Non-Maximum Suppression
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
        /// Find arrow marker by analyzing blob shape
        /// Arrow (Y-shape) has LOW solidity, regular segments have HIGH solidity
        /// </summary>
        private List<PointF> FindTriangleMarkers(Image<Gray, byte> source, RingRegion region)
        {
            var trianglePoints = new List<PointF>();

            try
            {
                int cx = (int)region.Center.X;
                int cy = (int)region.Center.Y;
                float outerR = region.OuterRadius;

                // Step 1: Apply CLAHE + Otsu to get binary image
                var normalized = new Image<Gray, byte>(source.Size);
                CvInvoke.CLAHE(source, 2.0, new System.Drawing.Size(8, 8), normalized);

                var binary = new Image<Gray, byte>(source.Size);
                CvInvoke.Threshold(normalized, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

                // Step 2: Mask to outer ring area (0.5R to 1.1R)
                var ringMask = new Image<Gray, byte>(source.Size);
                CvInvoke.Circle(ringMask, new Point(cx, cy), (int)(outerR * 1.1), new MCvScalar(255), -1);
                CvInvoke.Circle(ringMask, new Point(cx, cy), (int)(outerR * 0.5), new MCvScalar(0), -1);
                var masked = binary.Copy(ringMask);

                // Step 3: Find all separate blobs
                using var contours = new VectorOfVectorOfPoint();
                using var hierarchy = new Mat();
                CvInvoke.FindContours(masked, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                // Step 4: Find blob with LOWEST solidity (Y-shape has gaps between prongs)
                // Solidity = Area / ConvexHullArea
                double bestScore = 0;
                int bestIdx = -1;
                double minBlobArea = outerR * outerR * 0.02;

                for (int i = 0; i < contours.Size; i++)
                {
                    var contour = contours[i];
                    double area = CvInvoke.ContourArea(contour);

                    if (area < minBlobArea) continue;

                    // Calculate convex hull area
                    using var hull = new VectorOfPoint();
                    CvInvoke.ConvexHull(contour, hull);
                    double hullArea = CvInvoke.ContourArea(hull);

                    if (hullArea < 1) continue;

                    // Solidity: low for Y-shape (0.4-0.7), high for rectangle (0.8-1.0)
                    double solidity = area / hullArea;

                    // Arrow should have solidity between 0.3 and 0.75
                    // Score: prefer lower solidity + larger area
                    if (solidity > 0.25 && solidity < 0.8)
                    {
                        double score = area * (1.0 - solidity);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestIdx = i;
                        }
                    }
                }

                if (bestIdx < 0) return trianglePoints;

                // Step 5: Find the tip (furthest point from ring center)
                var arrowContour = contours[bestIdx];
                var points = arrowContour.ToArray();

                double maxDist = 0;
                Point tipPoint = new Point(cx, cy);

                foreach (var pt in points)
                {
                    double dist = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                    if (dist > maxDist)
                    {
                        maxDist = dist;
                        tipPoint = pt;
                    }
                }

                trianglePoints.Add(new PointF(tipPoint.X, tipPoint.Y));
            }
            catch { }

            return trianglePoints;
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
