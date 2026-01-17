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

        // Store last validation metrics for quality scoring
        private (double centerMean, double ringMean, double outerMean, double ringVsCenter, double ringVsOuter) _lastValidationMetrics;

        /// <summary>
        /// Calculate validation quality score (0-1) based on contrast ratios
        /// Higher score = better ring structure
        /// </summary>
        private float GetValidationQualityScore()
        {
            var m = _lastValidationMetrics;

            // Score based on ring vs center contrast (max contribution: 0.5)
            // Good ring should have ringVsCenter > 50, excellent > 100
            float centerScore = Math.Min(1.0f, (float)m.ringVsCenter / 100f) * 0.5f;

            // Score based on ring vs outer contrast (max contribution: 0.3)
            // Good ring should have ringVsOuter > 30, excellent > 80
            float outerScore = Math.Min(1.0f, (float)m.ringVsOuter / 80f) * 0.3f;

            // Score based on center darkness (max contribution: 0.2)
            // Center should be dark (< 80), excellent if < 40
            float darkScore = m.centerMean < 40 ? 0.2f :
                              m.centerMean < 80 ? 0.15f :
                              m.centerMean < 120 ? 0.1f : 0.05f;

            return centerScore + outerScore + darkScore;
        }

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
            int minDim = Math.Min(original.Width, original.Height);

            // Minimum ring size = 15% of image dimension
            float minRingRadius = minDim * 0.15f;
            Log($"=== FindRingCodesMultiLevel: image={original.Width}x{original.Height}, minRingRadius={minRingRadius:F0} ===");

            // Method 1 (FIRST): Try single large ring detection using HoughCircles
            // This is most reliable for single ring codes that fill most of the image
            Log("Method 1: Single large ring detection (HoughCircles)...");
            var singleRing = FindSingleLargeRing(preprocessed, original);
            if (singleRing != null)
            {
                // Validate: ring should be at least 30% of image size
                float ringDiameter = singleRing.OuterRadius * 2;
                Log($"  Single ring found: radius={singleRing.OuterRadius:F0}, diameter={ringDiameter:F0}, ratio={ringDiameter/minDim:P0}");
                if (ringDiameter > minDim * 0.3f)
                {
                    Log($"  ACCEPTED: Single large ring (>{minDim * 0.3f:F0})");
                    allRegions.Add(singleRing);
                    return allRegions;  // Return immediately - no need for other methods
                }
                else
                {
                    Log($"  REJECTED: Ring too small (<{minDim * 0.3f:F0})");
                }
            }
            else
            {
                Log("  No single ring found by HoughCircles");
            }

            // Method 2: Otsu threshold (for multiple smaller rings)
            Log("Trying Otsu threshold for multiple rings...");
            var otsuRegions = FindWithOtsuThreshold(preprocessed, original);
            // Filter: only keep rings that are reasonably sized (> 10% of image)
            foreach (var region in otsuRegions)
            {
                float ringDiameter = region.OuterRadius * 2;
                if (ringDiameter > minDim * 0.10f)
                {
                    allRegions.Add(region);
                }
            }

            // Method 3: Inverted Otsu for white rings with gray marks
            if (allRegions.Count == 0)
            {
                Log("Trying inverted threshold for white ring...");
                var invertedRegions = FindWithInvertedThreshold(preprocessed, original);
                foreach (var region in invertedRegions)
                {
                    float ringDiameter = region.OuterRadius * 2;
                    if (ringDiameter > minDim * 0.10f)
                    {
                        allRegions.Add(region);
                    }
                }
            }

            // Method 4: Single adaptive threshold (backup)
            if (allRegions.Count < 10 && allRegions.Count == 0)
            {
                var adaptiveRegions = FindWithAdaptiveThreshold(preprocessed, original,
                    AdaptiveThresholdType.GaussianC, 41, 4);
                foreach (var region in adaptiveRegions)
                {
                    float ringDiameter = region.OuterRadius * 2;
                    if (ringDiameter > minDim * 0.10f)
                    {
                        allRegions.Add(region);
                    }
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

        // Debug output path
        private static readonly string DebugOutputPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "RingDebug");

        /// <summary>
        /// Detect single large ring using contour hierarchy to find ring-shaped regions
        /// A ring has: outer boundary (parent) + inner hole (child)
        /// Uses the OUTER boundary's circularity and the presence of a hole
        /// Preprocessing uses HALCON-style local_threshold with adapted_std_deviation
        /// </summary>
        private RingRegion FindSingleLargeRing(Image<Gray, byte> preprocessed, Image<Gray, byte> original)
        {
            try
            {
                // Create debug output directory
                if (!Directory.Exists(DebugOutputPath))
                    Directory.CreateDirectory(DebugOutputPath);

                int imgWidth = original.Width;
                int imgHeight = original.Height;
                int minDim = Math.Min(imgWidth, imgHeight);

                var debugLog = new System.Text.StringBuilder();
                debugLog.AppendLine($"=== Ring Region Detection (HALCON-style local_threshold) ===");
                debugLog.AppendLine($"Image size: {imgWidth}x{imgHeight}, minDim: {minDim}");

                // Save original for debug
                original.Save(Path.Combine(DebugOutputPath, "01_original.png"));

                // ============ Simple Otsu + Blob Analysis ============
                // Step 1: Gaussian blur to reduce noise
                var blurred = new Image<Gray, byte>(original.Size);
                CvInvoke.GaussianBlur(original, blurred, new System.Drawing.Size(5, 5), 1);

                // Step 2: Otsu threshold - simple and robust
                var binary = new Image<Gray, byte>(original.Size);
                double otsuThresh = CvInvoke.Threshold(blurred, binary, 0, 255, ThresholdType.Otsu | ThresholdType.Binary);
                binary.Save(Path.Combine(DebugOutputPath, "02_otsu.png"));

                MCvScalar mean = CvInvoke.Mean(original);
                double meanGray = mean.V0;
                debugLog.AppendLine($"Mean brightness: {meanGray:F1}");
                debugLog.AppendLine($"Otsu threshold: {otsuThresh:F1}");

                // Step 3: Morphological cleaning
                var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new System.Drawing.Size(5, 5), new Point(-1, -1));
                CvInvoke.MorphologyEx(binary, binary, MorphOp.Close, kernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));
                CvInvoke.MorphologyEx(binary, binary, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

                // Step 4: Blob analysis - find connected components
                var labels = new Mat();
                var stats = new Mat();
                var centroids = new Mat();
                int numLabels = CvInvoke.ConnectedComponentsWithStats(binary, labels, stats, centroids);

                debugLog.AppendLine($"Found {numLabels - 1} blobs (excluding background)");

                // Create debug image for blob visualization
                var blobDebug = original.Convert<Bgr, byte>();

                // Step 5: Filter blobs by area and select the best ring candidate
                var statsData = new int[numLabels * 5];
                stats.CopyTo(statsData);

                var centroidsData = new double[numLabels * 2];
                centroids.CopyTo(centroidsData);

                // Area range for ring: 5% to 60% of image
                double minBlobArea = minDim * minDim * 0.05;
                double maxBlobArea = minDim * minDim * 0.60;

                debugLog.AppendLine($"Blob area filter: {minBlobArea:F0} - {maxBlobArea:F0}");
                debugLog.AppendLine();

                float bestScore = 0;
                System.Drawing.PointF bestCenter = new System.Drawing.PointF(imgWidth / 2f, imgHeight / 2f);
                float bestRadius = 0;
                int bestLabel = -1;

                for (int label = 1; label < numLabels; label++) // Skip background (label 0)
                {
                    int x = statsData[label * 5 + 0]; // Left
                    int y = statsData[label * 5 + 1]; // Top
                    int w = statsData[label * 5 + 2]; // Width
                    int h = statsData[label * 5 + 3]; // Height
                    int area = statsData[label * 5 + 4]; // Area

                    double cx = centroidsData[label * 2 + 0];
                    double cy = centroidsData[label * 2 + 1];

                    // Skip if area out of range
                    if (area < minBlobArea || area > maxBlobArea)
                    {
                        debugLog.AppendLine($"Blob #{label}: area={area} -> REJECTED (area out of range)");
                        continue;
                    }

                    // Calculate equivalent radius and aspect ratio
                    float equivRadius = (float)Math.Sqrt(area / Math.PI);
                    float aspectRatio = (float)w / h;
                    float boundingCircularity = (float)(4 * Math.PI * area) / (float)(Math.PI * Math.Max(w, h) * Math.Max(w, h));

                    // Score based on: circularity (aspect ratio close to 1) and size
                    float aspectScore = 1.0f - Math.Abs(aspectRatio - 1.0f); // 1 = perfect square
                    float sizeScore = (float)(area / maxBlobArea);

                    // Bonus for being near image center
                    float distFromCenter = (float)Math.Sqrt(Math.Pow(cx - imgWidth / 2, 2) + Math.Pow(cy - imgHeight / 2, 2));
                    float centerScore = 1.0f - (distFromCenter / (minDim * 0.5f));
                    centerScore = Math.Max(0, centerScore);

                    float score = aspectScore * 0.4f + sizeScore * 0.3f + centerScore * 0.3f;

                    debugLog.AppendLine($"Blob #{label}: center=({cx:F0},{cy:F0}), r={equivRadius:F0}, " +
                        $"area={area}, aspect={aspectRatio:F2}, score={score:F2}");

                    // Draw blob bounding box
                    var color = score > 0.3f ? new MCvScalar(0, 255, 0) : new MCvScalar(128, 128, 128);
                    CvInvoke.Rectangle(blobDebug, new Rectangle(x, y, w, h), color, 2);

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestCenter = new System.Drawing.PointF((float)cx, (float)cy);
                        bestRadius = equivRadius;
                        bestLabel = label;
                    }
                }

                // Draw best blob
                if (bestLabel > 0)
                {
                    CvInvoke.Circle(blobDebug, new Point((int)bestCenter.X, (int)bestCenter.Y),
                        (int)bestRadius, new MCvScalar(255, 0, 255), 3);
                    CvInvoke.Circle(blobDebug, new Point((int)bestCenter.X, (int)bestCenter.Y),
                        5, new MCvScalar(0, 255, 255), -1);
                }

                blobDebug.Save(Path.Combine(DebugOutputPath, "03_blob_analysis.png"));
                binary.Save(Path.Combine(DebugOutputPath, "04_binary_cleaned.png"));

                // Check if blob analysis found a valid ring
                if (bestLabel < 0 || bestRadius < minDim * 0.05f)
                {
                    debugLog.AppendLine("\nNo valid ring blob found!");
                    File.WriteAllText(Path.Combine(DebugOutputPath, "debug_log.txt"), debugLog.ToString());
                    Log("No ring blob found");
                    return null;
                }

                debugLog.AppendLine();
                debugLog.AppendLine($"SELECTED BLOB: center=({bestCenter.X:F0},{bestCenter.Y:F0}), " +
                    $"r={bestRadius:F0}, score={bestScore:F2}");

                Log($"Found ring blob: center=({bestCenter.X:F0},{bestCenter.Y:F0}), r={bestRadius:F0}");

                // Step 6: HALCON Metrology style circle fitting - refine center and radius using edge detection
                debugLog.AppendLine();
                debugLog.AppendLine("=== Metrology Circle Fitting ===");

                var (refinedCenter, refinedOuterR, fitSuccess) = FitCircleMetrology(original, bestCenter, bestRadius, true);
                if (fitSuccess)
                {
                    debugLog.AppendLine($"Outer fit: SUCCESS center=({refinedCenter.X:F1},{refinedCenter.Y:F1}), r={refinedOuterR:F1}");
                    bestCenter = refinedCenter;
                    bestRadius = refinedOuterR;
                    Log($"Metrology refined: center=({bestCenter.X:F1},{bestCenter.Y:F1}), r={bestRadius:F1}");
                }
                else
                {
                    debugLog.AppendLine($"Outer fit: FAILED, using blob r={bestRadius:F0}");
                }

                // Estimate inner radius and fit it too
                float estimatedInnerR = bestRadius * 0.4f;
                var (innerCenter, refinedInnerR, innerFitSuccess) = FitCircleMetrology(original, bestCenter, estimatedInnerR, false);
                if (innerFitSuccess)
                {
                    debugLog.AppendLine($"Inner fit: SUCCESS r={refinedInnerR:F1}");
                    estimatedInnerR = refinedInnerR;
                }
                else
                {
                    debugLog.AppendLine($"Inner fit: FAILED, using estimated r={estimatedInnerR:F0}");
                }

                float bestInnerRadius = estimatedInnerR;
                float bestOuterRadius = bestRadius;
                debugLog.AppendLine($"After Metrology: outerR={bestOuterRadius:F1}, innerR={bestInnerRadius:F1}");

                // Step 7: Refine using radial scanning for data boundaries
                var (innerR, outerR, dataOuterR) = FindWhiteRingBoundaries(original, bestCenter, bestOuterRadius);
                debugLog.AppendLine($"Radial scan: innerR={innerR:F1}, outerR={outerR:F1}, dataOuterR={dataOuterR:F1}");

                // Save debug log now
                File.WriteAllText(Path.Combine(DebugOutputPath, "debug_log.txt"), debugLog.ToString());

                // Validate and use refined values if reasonable
                if (innerR > 10 && outerR > innerR && outerR < bestOuterRadius * 1.3f)
                {
                    Log($"Refined ring: innerR={innerR:F0}, outerR={outerR:F0}, dataOuterR={dataOuterR:F0}");
                }
                else
                {
                    // Use contour-based values
                    innerR = bestInnerRadius;
                    outerR = bestOuterRadius;
                    dataOuterR = bestOuterRadius * 0.97f;
                    Log($"Using contour values: innerR={innerR:F0}, outerR={outerR:F0}");
                }

                // Create region
                var region = new RingRegion
                {
                    Center = bestCenter,
                    OuterRadius = dataOuterR,
                    InnerRadius = innerR,
                    BoundingBox = GetBoundingBox(bestCenter, outerR * 1.1f, original.Size),
                    MatchScore = bestScore * 100
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

                // Use radial scanning to find actual white ring boundaries
                var (innerR, outerR, dataOuterR) = FindWhiteRingBoundaries(original, center, refinedRadius);

                // Calculate pattern score
                double patternScore = CalculatePatternScore(original, center, refinedRadius);

                var region = new RingRegion
                {
                    Center = center,
                    OuterRadius = dataOuterR,  // Use data ring outer, not physical
                    InnerRadius = innerR,      // Use actual inner radius from scan
                    BoundingBox = GetBoundingBox(center, outerR * 1.1f, original.Size),
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
        /// Validate that the detected center has a dark hole (ring structure)
        /// Ring code should have: DARK center hole -> BRIGHT ring -> DARK background
        /// Uses ADAPTIVE thresholds and robust sampling (multiple radii, upper percentile)
        /// </summary>
        private bool ValidateCenterHasDarkHole(Image<Gray, byte> source, float cx, float cy, float radius)
        {
            int icx = (int)cx;
            int icy = (int)cy;

            // Check bounds
            if (icx < 10 || icx >= source.Width - 10 || icy < 10 || icy >= source.Height - 10)
                return false;

            // Step 1: Sample center region (should be DARK - the hole)
            int centerSamples = 0;
            double centerSum = 0;
            int sampleRadius = Math.Max(5, (int)(radius * 0.08));  // Small area at center

            for (int dy = -sampleRadius; dy <= sampleRadius; dy++)
            {
                for (int dx = -sampleRadius; dx <= sampleRadius; dx++)
                {
                    int px = icx + dx;
                    int py = icy + dy;
                    if (px >= 0 && px < source.Width && py >= 0 && py < source.Height)
                    {
                        centerSum += source.Data[py, px, 0];
                        centerSamples++;
                    }
                }
            }

            double centerMean = centerSamples > 0 ? centerSum / centerSamples : 255;

            // Step 2: Sample ring area at MULTIPLE radii to avoid hitting only data marks
            // Use 75th percentile (upper quartile) to get the WHITE ring brightness, not dark marks
            var ringPixels = new List<byte>();
            float[] ringRadii = { 0.45f, 0.55f, 0.65f, 0.75f };  // Multiple sampling radii
            int numAngles = 24;  // More angles for better coverage

            foreach (float rMult in ringRadii)
            {
                float ringR = radius * rMult;
                for (int a = 0; a < numAngles; a++)
                {
                    double angle = a * 2 * Math.PI / numAngles;
                    int px = (int)(cx + ringR * Math.Cos(angle));
                    int py = (int)(cy + ringR * Math.Sin(angle));
                    if (px >= 0 && px < source.Width && py >= 0 && py < source.Height)
                    {
                        ringPixels.Add(source.Data[py, px, 0]);
                    }
                }
            }

            // Use 75th percentile to represent "white ring" (ignores dark data marks)
            double ringMean = 0;
            if (ringPixels.Count > 0)
            {
                ringPixels.Sort();
                int idx75 = (int)(ringPixels.Count * 0.75);
                ringMean = ringPixels[Math.Min(idx75, ringPixels.Count - 1)];
            }

            // Step 3: Sample OUTER background (beyond the ring)
            var outerPixels = new List<byte>();
            float outerRadius = radius * 1.15f;

            for (int a = 0; a < numAngles; a++)
            {
                double angle = a * 2 * Math.PI / numAngles;
                int px = (int)(cx + outerRadius * Math.Cos(angle));
                int py = (int)(cy + outerRadius * Math.Sin(angle));
                if (px >= 0 && px < source.Width && py >= 0 && py < source.Height)
                {
                    outerPixels.Add(source.Data[py, px, 0]);
                }
            }

            // Use median for outer (more stable)
            double outerMean = centerMean;
            if (outerPixels.Count > 0)
            {
                outerPixels.Sort();
                outerMean = outerPixels[outerPixels.Count / 2];
            }

            // ADAPTIVE VALIDATION: Ring structure means ring is BRIGHTER than both center AND outer
            double ringVsCenter = ringMean - centerMean;
            double ringVsOuter = ringMean - outerMean;

            // Key insight: For a valid ring code, the ring area should be significantly brighter
            // than both the center hole AND the outer background
            bool hasValidStructure = ringVsCenter > 25 && ringVsOuter > 15;

            // Additional check: center should not be as bright as ring (< 85% of ring brightness)
            bool centerIsDarker = centerMean < ringMean * 0.85;

            bool isValid = hasValidStructure && centerIsDarker;

            Log($"    Circle ({cx:F0},{cy:F0}): center={centerMean:F0}, ring75%={ringMean:F0}, outer={outerMean:F0}, " +
                $"ringVsCenter={ringVsCenter:F0}, ringVsOuter={ringVsOuter:F0}, valid={isValid}");

            // Store validation metrics for quality scoring
            _lastValidationMetrics = (centerMean, ringMean, outerMean, ringVsCenter, ringVsOuter);

            return isValid;
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

        /// <summary>
        /// Find actual white ring boundaries using radial scanning with adaptive thresholds
        /// Returns (innerRadius, outerRadius, dataRingOuterRadius)
        /// </summary>
        public (float innerR, float outerR, float dataOuterR) FindWhiteRingBoundaries(
            Image<Gray, byte> source, PointF center, float physicalRadius)
        {
            int cx = (int)center.X;
            int cy = (int)center.Y;
            int numAngles = 36;  // Every 10 degrees

            // Step 1: Sample pixels to determine adaptive thresholds
            var ringPixels = new List<byte>();
            var centerPixels = new List<byte>();

            for (int a = 0; a < numAngles; a++)
            {
                double angle = a * 2 * Math.PI / numAngles;
                float cosA = (float)Math.Cos(angle);
                float sinA = (float)Math.Sin(angle);

                // Sample center area (should be dark)
                for (float r = 10; r < physicalRadius * 0.3f; r += 5)
                {
                    int x = (int)(cx + r * cosA);
                    int y = (int)(cy + r * sinA);
                    if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                        centerPixels.Add(source.Data[y, x, 0]);
                }

                // Sample ring area (should be mostly white with some dark marks)
                for (float r = physicalRadius * 0.4f; r < physicalRadius * 0.95f; r += 5)
                {
                    int x = (int)(cx + r * cosA);
                    int y = (int)(cy + r * sinA);
                    if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                        ringPixels.Add(source.Data[y, x, 0]);
                }
            }

            // Calculate adaptive thresholds
            centerPixels.Sort();
            ringPixels.Sort();

            byte centerMedian = centerPixels.Count > 0 ? centerPixels[centerPixels.Count / 2] : (byte)50;
            byte ringBright = ringPixels.Count > 0 ? ringPixels[(int)(ringPixels.Count * 0.75)] : (byte)200;

            // White threshold: 70% of the way from center to ring bright
            int whiteThreshold = centerMedian + (int)((ringBright - centerMedian) * 0.6);
            // Dark threshold: 30% of the way from center to ring bright
            int darkThreshold = centerMedian + (int)((ringBright - centerMedian) * 0.3);

            Log($"  Adaptive thresholds: centerMedian={centerMedian}, ringBright={ringBright}, white>{whiteThreshold}, dark<{darkThreshold}");

            var whiteRingStartRadii = new List<float>();
            var whiteRingEndRadii = new List<float>();

            for (int a = 0; a < numAngles; a++)
            {
                double angle = a * 2 * Math.PI / numAngles;
                float cosA = (float)Math.Cos(angle);
                float sinA = (float)Math.Sin(angle);

                // Phase 1: Find where white ring starts (skip black center)
                float whiteStart = 0;
                bool inWhiteRing = false;
                int consecutiveWhite = 0;

                for (float r = 20; r < physicalRadius * 1.2f; r += 2)
                {
                    int x = (int)(cx + r * cosA);
                    int y = (int)(cy + r * sinA);
                    if (x < 0 || x >= source.Width || y < 0 || y >= source.Height) break;

                    byte pixel = source.Data[y, x, 0];

                    if (pixel > whiteThreshold)  // Bright white pixel (adaptive)
                    {
                        consecutiveWhite++;
                        if (!inWhiteRing && consecutiveWhite > 5)
                        {
                            whiteStart = r - 10;
                            inWhiteRing = true;
                            break;
                        }
                    }
                    else
                    {
                        consecutiveWhite = 0;
                    }
                }

                if (!inWhiteRing) continue;

                // Phase 2: Find where white ring ends (transition to dark background)
                float whiteEnd = 0;
                int consecutiveDark = 0;

                for (float r = whiteStart; r < physicalRadius * 1.3f; r += 2)
                {
                    int x = (int)(cx + r * cosA);
                    int y = (int)(cy + r * sinA);
                    if (x < 0 || x >= source.Width || y < 0 || y >= source.Height) break;

                    byte pixel = source.Data[y, x, 0];

                    if (pixel < darkThreshold)  // Dark pixel (adaptive)
                    {
                        consecutiveDark++;
                        if (consecutiveDark > 15)  // Reduced from 20 for better detection
                        {
                            // This is the outer dark background
                            whiteEnd = r - 30;  // Adjusted from -40
                            break;
                        }
                    }
                    else
                    {
                        consecutiveDark = 0;
                    }
                }

                if (whiteEnd > whiteStart)
                {
                    whiteRingStartRadii.Add(whiteStart);
                    whiteRingEndRadii.Add(whiteEnd);
                }
            }

            Log($"  Radial scan found {whiteRingStartRadii.Count} valid samples");

            // Calculate ring boundaries using median for robustness
            float innerRadius, outerRadius, dataOuterRadius;
            if (whiteRingStartRadii.Count >= 8)  // Reduced from 10 for more lenient detection
            {
                whiteRingStartRadii.Sort();
                whiteRingEndRadii.Sort();

                innerRadius = whiteRingStartRadii[whiteRingStartRadii.Count / 2];
                outerRadius = whiteRingEndRadii[whiteRingEndRadii.Count / 2];

                // Data ring outer = 97% of white ring (arrow tip is at outer edge of data)
                dataOuterRadius = innerRadius + (outerRadius - innerRadius) * 0.97f;

                Log($"  White ring scan: inner={innerRadius:F0}, outer={outerRadius:F0}, dataOuter={dataOuterRadius:F0}");
            }
            else
            {
                // Better fallback: estimate based on typical ring code proportions
                // Inner radius is typically 40-42% of physical radius
                // Data outer is typically 97% of physical radius
                innerRadius = physicalRadius * 0.41f;
                outerRadius = physicalRadius;
                dataOuterRadius = physicalRadius * 0.97f;
                Log($"  White ring scan FALLBACK ({whiteRingStartRadii.Count} samples): inner={innerRadius:F0}, outer={outerRadius:F0}, dataOuter={dataOuterRadius:F0}");
            }

            return (innerRadius, outerRadius, dataOuterRadius);
        }

        /// <summary>
        /// Fit circle using outer boundary contour (like HALCON Metrology approach)
        /// Uses edge detection and finds circular contours near expected radius
        /// </summary>
        public (PointF center, float radius, bool success) FitCircleFromContourBoundary(
            Image<Gray, byte> source, PointF roughCenter, float roughRadius)
        {
            try
            {
                int cx = (int)roughCenter.X;
                int cy = (int)roughCenter.Y;
                int r = (int)(roughRadius * 1.3f);

                // Create ROI around the ring
                int roiX = Math.Max(0, cx - r);
                int roiY = Math.Max(0, cy - r);
                int roiW = Math.Min(r * 2, source.Width - roiX);
                int roiH = Math.Min(r * 2, source.Height - roiY);

                if (roiW < 50 || roiH < 50)
                    return (roughCenter, roughRadius, false);

                // Extract ROI
                source.ROI = new Rectangle(roiX, roiY, roiW, roiH);
                var roi = source.Clone();
                source.ROI = Rectangle.Empty;

                // Local center in ROI coordinates
                float localCx = roughCenter.X - roiX;
                float localCy = roughCenter.Y - roiY;

                // Step 1: Use Canny edge detection to find ring edges
                var blurred = new Image<Gray, byte>(roi.Size);
                CvInvoke.GaussianBlur(roi, blurred, new System.Drawing.Size(5, 5), 1.5);

                var edges = new Image<Gray, byte>(roi.Size);
                CvInvoke.Canny(blurred, edges, 50, 150);

                // Step 2: Dilate edges slightly to connect broken edges
                var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                    new System.Drawing.Size(3, 3), new Point(-1, -1));
                CvInvoke.Dilate(edges, edges, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

                // Step 3: Find contours from edges
                using var contours = new VectorOfVectorOfPoint();
                using var hierarchy = new Mat();
                CvInvoke.FindContours(edges.Clone(), contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxNone);

                if (contours.Size == 0)
                {
                    Log("  ContourFit: No edge contours found");
                    return (roughCenter, roughRadius, false);
                }

                // Step 4: Find contours that are circular and close to expected radius
                // Sample points from each contour and check if they form a circle around the center
                double expectedRadius = roughRadius;
                double radiusTolerance = roughRadius * 0.25;  // Allow 25% deviation

                int bestIdx = -1;
                double bestScore = 0;
                double bestRadius = 0;

                for (int i = 0; i < contours.Size; i++)
                {
                    var contour = contours[i];
                    if (contour.Size < 20) continue;  // Need enough points

                    // Sample points and calculate distance from center
                    var points = contour.ToArray();
                    var distances = new List<double>();

                    foreach (var pt in points)
                    {
                        double dist = Math.Sqrt(Math.Pow(pt.X - localCx, 2) + Math.Pow(pt.Y - localCy, 2));
                        distances.Add(dist);
                    }

                    double meanDist = distances.Average();
                    double stdDist = Math.Sqrt(distances.Average(d => Math.Pow(d - meanDist, 2)));

                    // Check if this contour is circular (low std deviation) and at expected radius
                    double radiusDiff = Math.Abs(meanDist - expectedRadius);
                    double circularity = 1.0 - (stdDist / meanDist);  // Higher = more circular

                    // Score: prefer circular contours at expected radius
                    if (radiusDiff < radiusTolerance && circularity > 0.7)
                    {
                        double score = circularity * (1.0 - radiusDiff / radiusTolerance) * contour.Size;

                        Log($"    Contour {i}: meanR={meanDist:F1}, std={stdDist:F1}, circ={circularity:F2}, score={score:F0}");

                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestIdx = i;
                            bestRadius = meanDist;
                        }
                    }
                }

                if (bestIdx < 0)
                {
                    Log("  ContourFit: No suitable circular contour found");
                    return (roughCenter, roughRadius, false);
                }

                var bestContour = contours[bestIdx];

                // Step 5: Use FitEllipse on the best circular contour
                if (bestContour.Size < 5)
                {
                    return (roughCenter, roughRadius, false);
                }

                var ellipse = CvInvoke.FitEllipse(bestContour);

                // Convert back to original image coordinates
                float fittedCenterX = ellipse.Center.X + roiX;
                float fittedCenterY = ellipse.Center.Y + roiY;
                float fittedRadius = (ellipse.Size.Width + ellipse.Size.Height) / 4f;

                // Validate
                float centerDist = (float)Math.Sqrt(Math.Pow(fittedCenterX - roughCenter.X, 2) +
                                                     Math.Pow(fittedCenterY - roughCenter.Y, 2));
                float radiusDiffPct = Math.Abs(fittedRadius - roughRadius) / roughRadius;

                Log($"  ContourFit: center=({fittedCenterX:F1},{fittedCenterY:F1}), radius={fittedRadius:F1}, " +
                    $"ellipse=({ellipse.Size.Width:F1}x{ellipse.Size.Height:F1}), centerDist={centerDist:F1}, radiusDiff={radiusDiffPct:P0}");

                // Accept if reasonable
                if (centerDist < roughRadius * 0.15f && radiusDiffPct < 0.25f)
                {
                    return (new PointF(fittedCenterX, fittedCenterY), fittedRadius, true);
                }
                else
                {
                    Log($"  ContourFit: Result rejected");
                    return (roughCenter, roughRadius, false);
                }
            }
            catch (Exception ex)
            {
                Log($"  ContourFit error: {ex.Message}");
                return (roughCenter, roughRadius, false);
            }
        }

        /// <summary>
        /// HALCON Metrology style circle fitting
        /// 1. Sample points around expected circle
        /// 2. Radial search for edge (max gradient)
        /// 3. Least squares circle fit on edge points
        /// </summary>
        public (PointF center, float radius, bool success) FitCircleMetrology(
            Image<Gray, byte> source, PointF roughCenter, float roughRadius, bool isOuterEdge)
        {
            try
            {
                int numSamples = 72; // Sample every 5 degrees
                float searchRange = roughRadius * 0.2f; // Search ±20% of radius
                float minSearchR = roughRadius - searchRange;
                float maxSearchR = roughRadius + searchRange;

                // Compute gradient magnitude using Sobel
                var gradX = new Image<Gray, float>(source.Size);
                var gradY = new Image<Gray, float>(source.Size);
                var gradMag = new Image<Gray, float>(source.Size);

                CvInvoke.Sobel(source, gradX, Emgu.CV.CvEnum.DepthType.Cv32F, 1, 0, 3);
                CvInvoke.Sobel(source, gradY, Emgu.CV.CvEnum.DepthType.Cv32F, 0, 1, 3);

                // Magnitude = sqrt(gx² + gy²)
                var gradX2 = new Image<Gray, float>(source.Size);
                var gradY2 = new Image<Gray, float>(source.Size);
                CvInvoke.Multiply(gradX, gradX, gradX2);
                CvInvoke.Multiply(gradY, gradY, gradY2);
                CvInvoke.Add(gradX2, gradY2, gradMag);
                CvInvoke.Sqrt(gradMag, gradMag);

                // Collect edge points
                var edgePoints = new List<PointF>();
                float gradThreshold = 30; // Minimum gradient to be considered an edge

                for (int i = 0; i < numSamples; i++)
                {
                    double angle = 2 * Math.PI * i / numSamples;
                    float cosA = (float)Math.Cos(angle);
                    float sinA = (float)Math.Sin(angle);

                    // For OUTER edge: search from OUTSIDE inward (find first edge)
                    // For INNER edge: search from INSIDE outward (find first edge)
                    float bestR = roughRadius;
                    bool found = false;

                    int numSteps = (int)(searchRange * 2);

                    if (isOuterEdge)
                    {
                        // Search from outside inward
                        for (int step = numSteps - 1; step >= 0; step--)
                        {
                            float r = minSearchR + step;
                            int px = (int)(roughCenter.X + r * cosA);
                            int py = (int)(roughCenter.Y + r * sinA);

                            if (px < 0 || px >= source.Width || py < 0 || py >= source.Height)
                                continue;

                            float grad = gradMag.Data[py, px, 0];
                            if (grad > gradThreshold)
                            {
                                bestR = r;
                                found = true;
                                break; // First significant edge from outside
                            }
                        }
                    }
                    else
                    {
                        // Search from inside outward
                        for (int step = 0; step < numSteps; step++)
                        {
                            float r = minSearchR + step;
                            int px = (int)(roughCenter.X + r * cosA);
                            int py = (int)(roughCenter.Y + r * sinA);

                            if (px < 0 || px >= source.Width || py < 0 || py >= source.Height)
                                continue;

                            float grad = gradMag.Data[py, px, 0];
                            if (grad > gradThreshold)
                            {
                                bestR = r;
                                found = true;
                                break; // First significant edge from inside
                            }
                        }
                    }

                    if (found)
                    {
                        float edgeX = roughCenter.X + bestR * cosA;
                        float edgeY = roughCenter.Y + bestR * sinA;
                        edgePoints.Add(new PointF(edgeX, edgeY));
                    }
                }

                // Need at least 50% of samples for reliable fit
                if (edgePoints.Count < numSamples / 2)
                {
                    Log($"  Metrology: Not enough edge points ({edgePoints.Count}/{numSamples})");
                    return (roughCenter, roughRadius, false);
                }

                // Least squares circle fitting
                // Minimize sum of (x - a)² + (y - b)² - r² = 0
                // Linearize: x² + y² = 2ax + 2by + (r² - a² - b²)
                // Let c = r² - a² - b², solve for [a, b, c]
                double sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0;
                double sumXY = 0, sumX3 = 0, sumY3 = 0, sumX2Y = 0, sumXY2 = 0;
                int n = edgePoints.Count;

                foreach (var pt in edgePoints)
                {
                    double x = pt.X, y = pt.Y;
                    double x2 = x * x, y2 = y * y;
                    sumX += x; sumY += y;
                    sumX2 += x2; sumY2 += y2;
                    sumXY += x * y;
                    sumX3 += x2 * x; sumY3 += y2 * y;
                    sumX2Y += x2 * y; sumXY2 += x * y2;
                }

                // Solve 3x3 linear system: A * [a, b, c]^T = B
                // A = [[sumX2, sumXY, sumX], [sumXY, sumY2, sumY], [sumX, sumY, n]]
                // B = [(sumX3 + sumXY2)/2, (sumX2Y + sumY3)/2, (sumX2 + sumY2)/2]
                double[,] A = {
                    { sumX2, sumXY, sumX },
                    { sumXY, sumY2, sumY },
                    { sumX, sumY, n }
                };
                double[] B = {
                    (sumX3 + sumXY2) / 2,
                    (sumX2Y + sumY3) / 2,
                    (sumX2 + sumY2) / 2
                };

                // Solve using Gaussian elimination (simple 3x3)
                double[] result = SolveLinearSystem3x3(A, B);
                if (result == null)
                {
                    Log("  Metrology: Linear system solve failed");
                    return (roughCenter, roughRadius, false);
                }

                double a = result[0], b = result[1], c = result[2];
                double fittedR = Math.Sqrt(c + a * a + b * b);

                PointF fittedCenter = new PointF((float)a, (float)b);

                // Validate result
                float centerDist = (float)Math.Sqrt(Math.Pow(a - roughCenter.X, 2) + Math.Pow(b - roughCenter.Y, 2));
                float radiusDiff = Math.Abs((float)fittedR - roughRadius);

                Log($"  Metrology: fitted center=({a:F1},{b:F1}), r={fittedR:F1}, " +
                    $"centerDist={centerDist:F1}, radiusDiff={radiusDiff:F1}, points={edgePoints.Count}");

                // Accept if reasonable (center within 10% of radius, radius within 15%)
                if (centerDist < roughRadius * 0.1f && radiusDiff < roughRadius * 0.15f)
                {
                    return (fittedCenter, (float)fittedR, true);
                }
                else
                {
                    Log("  Metrology: Result rejected (too far from initial estimate)");
                    return (roughCenter, roughRadius, false);
                }
            }
            catch (Exception ex)
            {
                Log($"  Metrology error: {ex.Message}");
                return (roughCenter, roughRadius, false);
            }
        }

        /// <summary>
        /// Solve 3x3 linear system using Gaussian elimination
        /// </summary>
        private double[] SolveLinearSystem3x3(double[,] A, double[] B)
        {
            // Create augmented matrix
            double[,] aug = new double[3, 4];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                    aug[i, j] = A[i, j];
                aug[i, 3] = B[i];
            }

            // Forward elimination
            for (int col = 0; col < 3; col++)
            {
                // Find pivot
                int maxRow = col;
                for (int row = col + 1; row < 3; row++)
                {
                    if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                        maxRow = row;
                }

                // Swap rows
                for (int j = 0; j < 4; j++)
                {
                    double temp = aug[col, j];
                    aug[col, j] = aug[maxRow, j];
                    aug[maxRow, j] = temp;
                }

                // Check for zero pivot
                if (Math.Abs(aug[col, col]) < 1e-10)
                    return null;

                // Eliminate below
                for (int row = col + 1; row < 3; row++)
                {
                    double factor = aug[row, col] / aug[col, col];
                    for (int j = col; j < 4; j++)
                        aug[row, j] -= factor * aug[col, j];
                }
            }

            // Back substitution
            double[] x = new double[3];
            for (int i = 2; i >= 0; i--)
            {
                x[i] = aug[i, 3];
                for (int j = i + 1; j < 3; j++)
                    x[i] -= aug[i, j] * x[j];
                x[i] /= aug[i, i];
            }

            return x;
        }
    }
}
