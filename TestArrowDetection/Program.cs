using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using CameraMaui.RingCode;

class Program
{
    // Set to true to test using RingCodeDecoder (same as MAUI app)
    static bool USE_RINGCODE_DECODER = true;

    // Set to true to test the RingImageSegmentation class circle fitting
    static bool USE_SEGMENTATION_CLASS = true;

    static void Main(string[] args)
    {
        string testDir = @"C:\Users\qoose\Desktop\文件資料\客戶分類\R-RCM\01_Software\RCM\03_Document\NG\20260113-002\NG";
        string outputDir = @"C:\Users\qoose\Desktop\ArrowDetectionTest";

        Directory.CreateDirectory(outputDir);

        // Enable debug output for arrow detection
        RingCodeDecoder.DebugOutputDir = outputDir;

        // Test specific image by name (set to null to test all, or filename to test one)
        string testSpecificFile = null;  // Test all images
        int maxImages = 10;

        var testFiles = Directory.GetFiles(testDir, "*.png");
        Console.WriteLine($"Testing {(maxImages > 0 ? maxImages : testFiles.Length)} of {testFiles.Length} images...\n");

        if (USE_RINGCODE_DECODER)
        {
            // Test using RingCodeDecoder (same logic as MAUI app)
            TestWithRingCodeDecoder(testFiles, outputDir, testSpecificFile, maxImages);
        }
        else
        {
            // Original test code
            var arrowTemplate = CreateArrowTemplate();
            Console.WriteLine($"Using RingImageSegmentation: {USE_SEGMENTATION_CLASS}\n");

            int imageCount = 0;
            foreach (var testImagePath in testFiles)
            {
                string fileName = Path.GetFileNameWithoutExtension(testImagePath);
                if (testSpecificFile != null && fileName != testSpecificFile) continue;
                if (maxImages > 0 && imageCount >= maxImages) break;
                imageCount++;

                Console.WriteLine($"\n===== Testing: {fileName} =====");
                var original = new Image<Gray, byte>(testImagePath);
                Console.WriteLine($"Image size: {original.Width}x{original.Height}");
                TestSingleImage(original, Path.Combine(outputDir, fileName), fileName, arrowTemplate);
                original.Dispose();
            }
        }

        Console.WriteLine("\n\n===== All tests completed =====");
    }

    static void TestWithRingCodeDecoder(string[] testFiles, string outputDir, string testSpecificFile, int maxImages)
    {
        // Enable detailed logging
        RingCodeDecoder.EnableDetailedLog = true;
        RingCodeDecoder.Log = (msg) => Console.WriteLine($"[Decoder] {msg}");
        RingImageSegmentation.Log = (msg) => Console.WriteLine($"  [Seg] {msg}");

        var decoder = new RingCodeDecoder();
        var segmentation = new RingImageSegmentation();

        int imageCount = 0;
        foreach (var testImagePath in testFiles)
        {
            string fileName = Path.GetFileNameWithoutExtension(testImagePath);
            if (testSpecificFile != null && fileName != testSpecificFile) continue;
            if (maxImages > 0 && imageCount >= maxImages) break;
            imageCount++;

            Console.WriteLine($"\n===== Testing with RingCodeDecoder: {fileName} =====");

            var colorImg = new Image<Bgr, byte>(testImagePath);
            var grayImg = colorImg.Convert<Gray, byte>();
            Console.WriteLine($"Image size: {grayImg.Width}x{grayImg.Height}");

            // Step 1: Segmentation
            var segResult = segmentation.SegmentImage(colorImg);
            if (!segResult.Success || segResult.DetectedRings.Count == 0)
            {
                Console.WriteLine("ERROR: Segmentation failed!");
                continue;
            }

            var ring = segResult.DetectedRings[0];
            Console.WriteLine($"Ring: center=({ring.Center.X:F0},{ring.Center.Y:F0}), outerR={ring.OuterRadius:F0}, innerR={ring.InnerRadius:F0}");

            // Step 2: Decode (includes arrow detection)
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var decodeResult = decoder.DecodeRing(grayImg, ring);
            sw.Stop();

            Console.WriteLine($"\n=== RESULT ===");
            Console.WriteLine($"  Arrow angle: {decodeResult.RotationAngle:F1}°");
            Console.WriteLine($"  Template match: {(decoder.LastTemplateMatchResult != null ? $"score={decoder.LastTemplateMatchResult.Score:F3}" : "FAILED")}");
            Console.WriteLine($"  Decoded: {decodeResult.DecodedData}");
            Console.WriteLine($"  Valid: {decodeResult.IsValid}");
            Console.WriteLine($"  Time: {sw.ElapsedMilliseconds}ms");

            // Save result image
            var resultImg = colorImg.Clone();
            int cx = (int)ring.Center.X;
            int cy = (int)ring.Center.Y;

            // Draw arrow direction
            double arrowRad = decodeResult.RotationAngle * Math.PI / 180;
            int arrowEndX = (int)(cx + ring.OuterRadius * 1.1 * Math.Cos(arrowRad));
            int arrowEndY = (int)(cy + ring.OuterRadius * 1.1 * Math.Sin(arrowRad));
            CvInvoke.ArrowedLine(resultImg, new Point(cx, cy), new Point(arrowEndX, arrowEndY),
                new MCvScalar(0, 255, 255), 3, LineType.AntiAlias);

            // Draw circles
            CvInvoke.Circle(resultImg, new Point(cx, cy), (int)ring.OuterRadius, new MCvScalar(0, 255, 0), 2);
            CvInvoke.Circle(resultImg, new Point(cx, cy), (int)ring.InnerRadius, new MCvScalar(0, 255, 0), 2);

            resultImg.Save(Path.Combine(outputDir, $"{fileName}_decoder_result.png"));
            Console.WriteLine($"Saved: {fileName}_decoder_result.png");

            colorImg.Dispose();
            grayImg.Dispose();
            resultImg.Dispose();
        }
    }

    static void TestSingleImage(Image<Gray, byte> original, string outputPrefix, string fileName, VectorOfPoint arrowTemplate)
    {
        PointF center;
        float outerRadius, innerRadius;

        if (USE_SEGMENTATION_CLASS)
        {
            // Use RingImageSegmentation class for circle fitting
            var segmentation = new RingImageSegmentation();
            RingImageSegmentation.Log = (msg) => Console.WriteLine($"  [Seg] {msg}");

            var colorImg = original.Convert<Bgr, byte>();
            var result = segmentation.SegmentImage(colorImg);

            if (result.Success && result.DetectedRings.Count > 0)
            {
                var ring = result.DetectedRings[0];
                center = ring.Center;
                outerRadius = ring.OuterRadius;
                innerRadius = ring.InnerRadius;
                Console.WriteLine($"[Segmentation] Ring found: center=({center.X:F0}, {center.Y:F0}), outerR={outerRadius:F0}, innerR={innerRadius:F0}");
            }
            else
            {
                Console.WriteLine($"[Segmentation] FAILED to find ring, using fallback...");
                (center, outerRadius, innerRadius) = FindRing(original);
            }
        }
        else
        {
            // Use simple method
            (center, outerRadius, innerRadius) = FindRing(original);
            Console.WriteLine($"Ring found: center=({center.X:F0}, {center.Y:F0}), outerR={outerRadius:F0}, innerR={innerRadius:F0}");
        }

        // Save preprocessing overlay for visual inspection
        SavePreprocessingOverlay(original, center, outerRadius, innerRadius, outputPrefix);

        // Step 2: Test arrow detection with shape matching
        var arrowResult = TestArrowDetection(original, center, outerRadius, innerRadius, outputPrefix, arrowTemplate);
        Console.WriteLine($"==> Result for {fileName}: arrow at {arrowResult:F1}°");
    }

    static void SavePreprocessingOverlay(Image<Gray, byte> original, PointF center, float outerR, float innerR, string outputPrefix)
    {
        var overlay = original.Convert<Bgr, byte>();
        int cx = (int)center.X;
        int cy = (int)center.Y;

        // Draw detected outer radius (GREEN - thick)
        CvInvoke.Circle(overlay, new Point(cx, cy), (int)outerR, new MCvScalar(0, 255, 0), 3);

        // Draw detected inner radius (GREEN - thick)
        CvInvoke.Circle(overlay, new Point(cx, cy), (int)innerR, new MCvScalar(0, 255, 0), 3);

        // Draw data ring search area (YELLOW) - 0.50R to 0.98R
        CvInvoke.Circle(overlay, new Point(cx, cy), (int)(outerR * 0.50), new MCvScalar(0, 255, 255), 2);
        CvInvoke.Circle(overlay, new Point(cx, cy), (int)(outerR * 0.98), new MCvScalar(0, 255, 255), 2);

        // Draw arrow tip search area (CYAN) - 0.70R
        CvInvoke.Circle(overlay, new Point(cx, cy), (int)(outerR * 0.70), new MCvScalar(255, 255, 0), 2);

        // Draw center point (RED)
        CvInvoke.Circle(overlay, new Point(cx, cy), 8, new MCvScalar(0, 0, 255), -1);

        // Draw cross at center
        CvInvoke.Line(overlay, new Point(cx - 30, cy), new Point(cx + 30, cy), new MCvScalar(0, 0, 255), 2);
        CvInvoke.Line(overlay, new Point(cx, cy - 30), new Point(cx, cy + 30), new MCvScalar(0, 0, 255), 2);

        // Add legend
        CvInvoke.PutText(overlay, $"OuterR={outerR:F0} InnerR={innerR:F0}", new Point(10, 30),
            FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);
        CvInvoke.PutText(overlay, "GREEN=detected ring boundary", new Point(10, 55),
            FontFace.HersheySimplex, 0.6, new MCvScalar(0, 255, 0), 1);
        CvInvoke.PutText(overlay, "YELLOW=data search (0.50-0.98R)", new Point(10, 80),
            FontFace.HersheySimplex, 0.6, new MCvScalar(0, 255, 255), 1);
        CvInvoke.PutText(overlay, "CYAN=arrow tip area (0.70R)", new Point(10, 105),
            FontFace.HersheySimplex, 0.6, new MCvScalar(255, 255, 0), 1);

        overlay.Save(outputPrefix + "_preprocessing.png");
        Console.WriteLine($"Preprocessing saved: {outputPrefix}_preprocessing.png");
    }

    static (PointF center, float outerRadius, float innerRadius) FindRing(Image<Gray, byte> source)
    {
        // Step 1: Use Otsu to find physical ring boundary first
        var binary = new Image<Gray, byte>(source.Size);
        CvInvoke.Threshold(source, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

        // Find contours to get approximate center
        using var contours = new VectorOfVectorOfPoint();
        using var hierarchy = new Mat();
        CvInvoke.FindContours(binary, contours, hierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

        PointF center = new PointF(source.Width / 2f, source.Height / 2f);
        float physicalOuterR = 100;

        // Find largest circular contour for center
        double bestScore = 0;
        for (int i = 0; i < contours.Size; i++)
        {
            var contour = contours[i];
            double area = CvInvoke.ContourArea(contour);
            double perimeter = CvInvoke.ArcLength(contour, true);
            if (perimeter < 100 || area < 1000) continue;

            double circularity = 4 * Math.PI * area / (perimeter * perimeter);
            if (circularity < 0.5) continue;

            var moments = CvInvoke.Moments(contour);
            if (moments.M00 < 1) continue;

            float cx = (float)(moments.M10 / moments.M00);
            float cy = (float)(moments.M01 / moments.M00);
            float radius = (float)Math.Sqrt(area / Math.PI);

            double score = area * circularity;
            if (score > bestScore && radius > 30)
            {
                bestScore = score;
                center = new PointF(cx, cy);
                physicalOuterR = radius;
            }
        }

        Console.WriteLine($"  Physical ring: center=({center.X:F0},{center.Y:F0}), physicalR={physicalOuterR:F0}");

        // Step 2: Radial scan to find DATA RING boundaries
        // Ring structure: BLACK center hole -> WHITE ring with DARK data marks -> dark background
        int cx_i = (int)center.X;
        int cy_i = (int)center.Y;

        // Scan at multiple angles
        int numAngles = 36;  // Every 10 degrees
        var whiteRingStartRadii = new List<float>();  // Where white ring begins (inner edge)
        var whiteRingEndRadii = new List<float>();    // Where white ring ends (outer edge)
        var dataMarkEndRadii = new List<float>();     // Where data marks end

        for (int a = 0; a < numAngles; a++)
        {
            double angle = a * 2 * Math.PI / numAngles;
            float cosA = (float)Math.Cos(angle);
            float sinA = (float)Math.Sin(angle);

            // Phase 1: Find where white ring starts (skip black center)
            float whiteStart = 0;
            bool inWhiteRing = false;
            int consecutiveWhite = 0;

            for (float r = 20; r < physicalOuterR * 1.2f; r += 2)
            {
                int x = (int)(cx_i + r * cosA);
                int y = (int)(cy_i + r * sinA);
                if (x < 0 || x >= source.Width || y < 0 || y >= source.Height) break;

                byte pixel = source.Data[y, x, 0];

                if (pixel > 180)  // Bright white pixel
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
            float lastDataMark = whiteStart;
            int consecutiveDark = 0;

            for (float r = whiteStart; r < physicalOuterR * 1.3f; r += 2)
            {
                int x = (int)(cx_i + r * cosA);
                int y = (int)(cy_i + r * sinA);
                if (x < 0 || x >= source.Width || y < 0 || y >= source.Height) break;

                byte pixel = source.Data[y, x, 0];

                if (pixel < 100)  // Dark pixel
                {
                    consecutiveDark++;
                    // Check if this is a data mark (surrounded by white) or the outer dark background
                    if (consecutiveDark > 20)
                    {
                        // This is the outer dark background
                        whiteEnd = r - 40;
                        break;
                    }
                    else if (consecutiveDark > 3)
                    {
                        // This might be a data mark
                        lastDataMark = r;
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
                dataMarkEndRadii.Add(lastDataMark);
            }
        }

        // Calculate ring boundaries
        float innerRadius, outerRadius;
        if (whiteRingStartRadii.Count > 10)
        {
            // Use median for robustness
            whiteRingStartRadii.Sort();
            whiteRingEndRadii.Sort();

            innerRadius = whiteRingStartRadii[whiteRingStartRadii.Count / 2];
            float whiteOuter = whiteRingEndRadii[whiteRingEndRadii.Count / 2];

            // Data marks (including arrow) extend to ~95% of white ring
            // Arrow is at the outermost part of the data ring
            outerRadius = innerRadius + (whiteOuter - innerRadius) * 0.95f;

            Console.WriteLine($"  White ring: inner={innerRadius:F0}, outer={whiteOuter:F0}");
            Console.WriteLine($"  Estimated data ring outer: {outerRadius:F0}");
        }
        else
        {
            // Fallback based on physical radius
            innerRadius = physicalOuterR * 0.4f;
            outerRadius = physicalOuterR * 0.85f;
            Console.WriteLine($"  Data ring fallback: inner={innerRadius:F0}, outer={outerRadius:F0}");
        }

        return (center, outerRadius, innerRadius);
    }

    static double TestArrowDetection(Image<Gray, byte> original, PointF center, float outerR, float innerR, string outputPrefix, VectorOfPoint arrowTemplate)
    {
        int cx = (int)center.X;
        int cy = (int)center.Y;

        Console.WriteLine("\n=== Arrow Detection Test ===");

        // === USE RING-BASED PREPROCESSING (same as data mark detection) ===
        // Step 1: Create mask for data ring area only
        var dataMask = new Image<Gray, byte>(original.Size);
        CvInvoke.Circle(dataMask, new Point(cx, cy), (int)outerR, new MCvScalar(255), -1);
        CvInvoke.Circle(dataMask, new Point(cx, cy), (int)innerR, new MCvScalar(0), -1);

        // Step 2: Apply CLAHE for uneven lighting correction
        var enhanced = new Mat();
        CvInvoke.CLAHE(original, 2.0, new Size(8, 8), enhanced);
        var enhancedImg = enhanced.ToImage<Gray, byte>();

        // Step 3: Collect pixel values from ring area only
        var ringPixels = new List<byte>();
        for (int y = 0; y < enhancedImg.Height; y++)
        {
            for (int x = 0; x < enhancedImg.Width; x++)
            {
                if (dataMask.Data[y, x, 0] > 0)
                {
                    ringPixels.Add(enhancedImg.Data[y, x, 0]);
                }
            }
        }

        // Calculate statistics for ring area
        double meanValue = ringPixels.Average(p => (double)p);
        double stdDev = Math.Sqrt(ringPixels.Average(p => Math.Pow(p - meanValue, 2)));
        double calculatedThresh = meanValue - 1.5 * stdDev;
        byte optimalThresh = (byte)Math.Max(80, Math.Min(180, calculatedThresh));

        Console.WriteLine($"  Ring-based threshold: {optimalThresh} (mean={meanValue:F0}, std={stdDev:F0})");

        // Step 4: Apply binary threshold to find dark marks
        var binaryResult = new Image<Gray, byte>(enhancedImg.Size);
        CvInvoke.Threshold(enhancedImg, binaryResult, optimalThresh, 255, ThresholdType.BinaryInv);

        // Step 5: Apply mask
        var maskedBinary = new Image<Gray, byte>(original.Size);
        CvInvoke.BitwiseAnd(binaryResult, dataMask, maskedBinary);

        // Step 6: Morphological cleanup (OPEN only - no CLOSE to preserve Y-shape)
        var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
        CvInvoke.MorphologyEx(maskedBinary, maskedBinary, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        // No CLOSE operation - preserve arrow Y-shape low solidity

        // Save preprocessed binary for arrow detection
        maskedBinary.Save(outputPrefix + "_arrow_binary.png");

        // ========================================================================
        // TEMPLATE MATCHING (PRIMARY METHOD - similar to HALCON find_scaled_shape_model)
        // ========================================================================
        Console.WriteLine("\n=== Template Matching (HALCON-style) ===");

        // Determine if ring is light or dark based on mean brightness
        bool isLightRing = meanValue > 127;
        Console.WriteLine($"  Ring type: {(isLightRing ? "LIGHT" : "DARK")} (mean={meanValue:F0})");

        var (templateScore, templateAngle, templateCenter) = FindArrowByTemplateMatching(maskedBinary, center, innerR, outerR, isLightRing);

        // If template match is confident, use it directly
        bool useTemplateResult = templateScore >= 0.45;

        // Find contours and analyze
        var resultImage = original.Convert<Bgr, byte>();

        // Draw search zone
        CvInvoke.Circle(resultImage, new Point(cx, cy), (int)outerR, new MCvScalar(0, 255, 255), 1);
        CvInvoke.Circle(resultImage, new Point(cx, cy), (int)innerR, new MCvScalar(0, 255, 255), 1);

        // Find contours from preprocessed binary
        var allCandidates = new List<(double score, double solidity, double angle, double area,
            double centroidDist, double tipDist, VectorOfPoint contour, PointF basePoint, PointF tipPoint, string method)>();

        using var contours = new VectorOfVectorOfPoint();
        using var hierarchy = new Mat();
        CvInvoke.FindContours(maskedBinary.Clone(), contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

        Console.WriteLine($"\nPreprocessed binary: Found {contours.Size} contours");

        for (int i = 0; i < contours.Size; i++)
        {
            var contour = contours[i];
            double area = CvInvoke.ContourArea(contour);

            // Arrow should have reasonable area - NOT tiny noise!
            double minArea = outerR * outerR * 0.005;  // ~550 for outerR=332
            double maxArea = outerR * outerR * 0.10;   // ~11000 for outerR=332
            if (area < minArea || area > maxArea) continue;

            var moments = CvInvoke.Moments(contour);
            if (moments.M00 < 1) continue;

            float ctrX = (float)(moments.M10 / moments.M00);
            float ctrY = (float)(moments.M01 / moments.M00);

            double centroidDist = Math.Sqrt(Math.Pow(ctrX - cx, 2) + Math.Pow(ctrY - cy, 2));
            double centroidDistRatio = centroidDist / outerR;

            // Find tip (furthest from center) and base (closest to center)
            var points = contour.ToArray();
            PointF tip = new PointF(ctrX, ctrY);
            PointF basePoint = new PointF(ctrX, ctrY);
            double maxTipDist = 0;
            double minBaseDist = double.MaxValue;

            foreach (var pt in points)
            {
                double d = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                if (d > maxTipDist)
                {
                    maxTipDist = d;
                    tip = new PointF(pt.X, pt.Y);
                }
                if (d < minBaseDist)
                {
                    minBaseDist = d;
                    basePoint = new PointF(pt.X, pt.Y);
                }
            }

            double tipDistRatio = maxTipDist / outerR;

            // Calculate solidity - arrow (Y/triangle) has LOW solidity
            using var hull = new VectorOfPoint();
            CvInvoke.ConvexHull(contour, hull);
            double hullArea = CvInvoke.ContourArea(hull);
            double solidity = hullArea > 0 ? area / hullArea : 1.0;

            // Calculate angle from center to BASE (closest point) - this is the arrow direction
            // Use image coordinate convention: Y increases downward, so 0°=right, 90°=down, 180°=left, 270°=up
            double angle = Math.Atan2(basePoint.Y - cy, basePoint.X - cx) * 180 / Math.PI;
            if (angle < 0) angle += 360;

            // Check if tip points outward (arrow characteristic)
            bool tipPointsOutward = maxTipDist > minBaseDist && (maxTipDist - minBaseDist) > 3;

            // Calculate elongation ratio (tip-base distance vs area)
            double elongation = (maxTipDist - minBaseDist) / Math.Sqrt(area);

            // === CONVEXITY DEFECTS ANALYSIS for Y-shape detection ===
            // Y-shaped arrow has significant concavity (defect) where the two branches meet
            int significantDefects = 0;
            double maxDefectDepth = 0;
            try
            {
                // Get convex hull indices
                using var hullIndices = new VectorOfInt();
                CvInvoke.ConvexHull(contour, hullIndices, false, false);

                if (hullIndices.Size >= 3)
                {
                    using var defects = new Mat();
                    CvInvoke.ConvexityDefects(contour, hullIndices, defects);

                    if (!defects.IsEmpty && defects.Rows > 0)
                    {
                        var defectData = new int[defects.Rows * 4];
                        defects.CopyTo(defectData);

                        for (int d = 0; d < defects.Rows; d++)
                        {
                            // defect: start_index, end_index, farthest_point_index, distance (fixpoint 8.8)
                            double depth = defectData[d * 4 + 3] / 256.0;  // Convert from fixpoint

                            // Significant defect if depth > 10% of contour size
                            double minDepth = Math.Sqrt(area) * 0.15;
                            if (depth > minDepth)
                            {
                                significantDefects++;
                                if (depth > maxDefectDepth)
                                    maxDefectDepth = depth;
                            }
                        }
                    }
                }
            }
            catch { /* Ignore defect errors */ }

            // Y-shape arrow typically has 1-2 significant defects (the concave part of Y)
            // Rectangular data marks have 0 defects (fully convex)
            // Use defect depth RATIO - deeper defect relative to size = more Y-shaped
            double defectRatio = maxDefectDepth / Math.Sqrt(area);

            // === SHAPE MATCHING with Y-arrow template ===
            double shapeMatchScore = CalculateShapeMatchScore(contour, arrowTemplate);

            // Score calculation - SHAPE MATCHING IS PRIMARY
            // If shape matches well, it's very likely the arrow
            double baseScore = CalculateArrowScore(solidity, centroidDistRatio, tipDistRatio, area, outerR, tipPointsOutward, elongation);

            // Combined score: low solidity is the PRIMARY indicator of Y-arrow
            // Y-arrows have distinct low solidity (0.4-0.65) that data marks don't have
            double score;

            // Primary indicator: VERY LOW solidity (< 0.60) strongly suggests Y-arrow
            bool hasVeryLowSolidity = solidity < 0.60;

            if (shapeMatchScore >= 0.7)
            {
                // Good shape match - this IS the arrow, give guaranteed high score
                score = Math.Max(0.90, baseScore * 0.2 + shapeMatchScore * 0.8 + 0.15);
            }
            else if (hasVeryLowSolidity)
            {
                // Very low solidity is a strong Y-shape indicator even without good shapeMatch
                // This handles cases where template matching fails but solidity is clearly Y-like
                double solidityBonus = (0.60 - solidity) * 0.5;  // up to 0.10 bonus for solidity=0.40
                score = Math.Max(0.75, baseScore * 0.5 + 0.30 + solidityBonus);
            }
            else if (defectRatio >= 0.20 && solidity < 0.90 && solidity >= 0.60)
            {
                // Medium-high defect ratio with medium solidity indicates Y-shape
                // This catches Y-arrows that appear more solid due to thresholding
                // Cap the boost to avoid oversized merged marks from winning
                double defectBoostScore = Math.Min(0.75, 0.55 + defectRatio * 0.6);
                score = Math.Max(defectBoostScore, baseScore * 0.6);
                Console.WriteLine($"    -> Defect boost: defectRatio={defectRatio:F2}, solidity={solidity:F2}");
            }
            else if (shapeMatchScore >= 0.5)
            {
                // Medium shape match - balanced weighting
                score = baseScore * 0.4 + shapeMatchScore * 0.6;
            }
            else
            {
                // Poor shape match AND high solidity - likely a data mark, not arrow
                // Cap the max score
                score = Math.Min(0.65, baseScore * 0.6);
            }

            // Note: defectBonus removed as defect-based scoring is now handled in the main scoring branches above

            // PENALTY for very small contours - likely noise, not real arrow
            // Real Y-arrow has area > 2000 typically (based on typical ring size ~330R)
            double minExpectedArea = outerR * outerR * 0.02;  // ~2% of ring area (~2200 for R=330)
            double maxExpectedArea = outerR * outerR * 0.04;  // ~4% of ring area (~4400 for R=330)

            if (area < minExpectedArea && shapeMatchScore < 0.7)
            {
                double areaRatio = area / minExpectedArea;
                double areaPenalty = 0.25 * (1.0 - areaRatio);
                // Extra penalty for VERY small contours (< 50% of expected)
                if (areaRatio < 0.5)
                    areaPenalty += 0.15;
                score -= areaPenalty;
            }

            // PENALTY for oversized contours - likely merged data marks, not Y-arrow
            // Merged marks often have low solidity (concave regions) but are too large
            if (area > maxExpectedArea)
            {
                double oversizeRatio = area / maxExpectedArea;
                // Stronger penalty: merged marks with low solidity are deceptive
                double oversizePenalty = 0.30 * (oversizeRatio - 1.0);  // 0.30 per 100% oversize
                // Extra penalty for VERY oversized (> 1.5x expected max)
                if (oversizeRatio > 1.5)
                    oversizePenalty += 0.20;
                // Cap the boost from low solidity for oversized contours
                if (hasVeryLowSolidity && oversizeRatio > 1.2)
                {
                    // Oversized + low solidity = likely merged marks, not Y-arrow
                    oversizePenalty += 0.15;
                }
                score -= oversizePenalty;
                Console.WriteLine($"    -> Oversize penalty: -{oversizePenalty:F2} (area {area:F0} > max {maxExpectedArea:F0})");
            }

            Console.WriteLine($"  Contour {i}: area={area:F0}, solidity={solidity:F3}, defect={defectRatio:F2}, " +
                $"centroid={centroidDistRatio:F2}R, tip={tipDistRatio:F2}R, angle={angle:F0}°, " +
                $"shapeMatch={shapeMatchScore:F2}, score={score:F2}");

            if (score > 0.1)
            {
                var contourCopy = new VectorOfPoint(contour.ToArray());
                allCandidates.Add((score, solidity, angle, area, centroidDistRatio, tipDistRatio,
                    contourCopy, basePoint, tip, "Preprocessed"));
            }
        }

        // === Y-SHAPE DETECTION: DISABLED due to high false positive rate ===
        // Adjacent data marks (15° apart) were incorrectly being detected as Y-pairs
        // Rely on single-contour detection with low solidity instead
        Console.WriteLine($"\n=== Y-SHAPE PAIR DETECTION (disabled) ===");

        // Sort and display results
        Console.WriteLine($"\n=== Total candidates: {allCandidates.Count} ===");

        var sorted = allCandidates.OrderByDescending(c => c.score).ToList();

        // === OVERLAY PREPROCESSED BINARY ON RESULT IMAGE ===
        // Convert binary mask to color and overlay with transparency
        for (int y = 0; y < maskedBinary.Height; y++)
        {
            for (int x = 0; x < maskedBinary.Width; x++)
            {
                if (maskedBinary.Data[y, x, 0] > 128)
                {
                    // Overlay cyan color for detected marks
                    resultImage.Data[y, x, 0] = (byte)Math.Min(255, resultImage.Data[y, x, 0] + 100);  // B
                    resultImage.Data[y, x, 1] = (byte)Math.Min(255, resultImage.Data[y, x, 1] + 100);  // G
                }
            }
        }

        // Draw all candidates with different colors
        for (int i = 0; i < sorted.Count; i++)
        {
            var c = sorted[i];
            var color = i == 0 ? new MCvScalar(0, 165, 255) :  // Orange for best (detected arrow)
                       i == 1 ? new MCvScalar(255, 0, 0) :     // Blue for 2nd
                       new MCvScalar(128, 128, 128);           // Gray for others

            CvInvoke.DrawContours(resultImage, new VectorOfVectorOfPoint(c.contour), -1, color, 2);
            CvInvoke.Circle(resultImage, new Point((int)c.basePoint.X, (int)c.basePoint.Y), 5, color, -1);

            // Draw line from center through base point (arrow direction)
            CvInvoke.Line(resultImage, new Point(cx, cy), new Point((int)c.basePoint.X, (int)c.basePoint.Y), color, 1);

            Console.WriteLine($"  #{i+1}: score={c.score:F2}, angle={c.angle:F0}°, " +
                $"solidity={c.solidity:F3}, elongation, method={c.method}");
        }

        // ========================================================================
        // DETERMINE FINAL ARROW ANGLE
        // ========================================================================
        double finalArrowAngle = -1;
        string detectionMethod = "NONE";

        if (useTemplateResult)
        {
            // Use template matching result (HALCON-style)
            finalArrowAngle = templateAngle;
            detectionMethod = "TEMPLATE";
            Console.WriteLine($"\n=== USING TEMPLATE MATCH: score={templateScore:F3}, angle={templateAngle:F1}° ===");
        }
        else if (sorted.Count > 0 && sorted[0].score >= 0.5)
        {
            // Fallback to contour-based detection
            finalArrowAngle = sorted[0].angle;
            detectionMethod = sorted[0].method;
            Console.WriteLine($"\n=== USING CONTOUR DETECTION: score={sorted[0].score:F2}, angle={sorted[0].angle:F1}° ===");
        }
        else if (sorted.Count > 0)
        {
            // Low confidence - use template if any score, else contour
            if (templateScore > 0.25)
            {
                finalArrowAngle = templateAngle;
                detectionMethod = "TEMPLATE(low)";
                Console.WriteLine($"\n=== USING TEMPLATE (low confidence): score={templateScore:F3}, angle={templateAngle:F1}° ===");
            }
            else
            {
                finalArrowAngle = sorted[0].angle;
                detectionMethod = sorted[0].method + "(low)";
                Console.WriteLine($"\n=== USING CONTOUR (low confidence): score={sorted[0].score:F2}, angle={sorted[0].angle:F1}° ===");
            }
        }

        if (finalArrowAngle >= 0)
        {
            int sector = (int)Math.Round(finalArrowAngle / 15.0) % 24;
            double snappedAngle = sector * 15.0;
            var best = sorted.Count > 0 ? sorted[0] :
                (score: templateScore, solidity: 0.0, angle: templateAngle, area: 0.0, centroidDist: 0.0, tipDist: 0.0,
                 contour: (VectorOfPoint)null, basePoint: templateCenter, tipPoint: templateCenter, method: "Template");

            Console.WriteLine($"\n=== DETECTED ARROW ({detectionMethod}): {finalArrowAngle:F1}° -> sector {sector} ({snappedAngle}°) ===");

            // Draw line from CENTER in direction of detected arrow angle
            // Use finalArrowAngle directly (more reliable than basePoint for template matching)
            double dirAngleRad = finalArrowAngle * Math.PI / 180;
            int endX = cx + (int)(outerR * 1.1 * Math.Cos(dirAngleRad));
            int endY = cy + (int)(outerR * 1.1 * Math.Sin(dirAngleRad));
            Console.WriteLine($"  Drawing arrow direction: angle={finalArrowAngle:F1}°, endpoint=({endX},{endY})");
            CvInvoke.ArrowedLine(resultImage, new Point(cx, cy), new Point(endX, endY),
                new MCvScalar(0, 255, 0), 3);

            // Mark the detection center with a circle (orange for template, yellow for contour)
            var markerColor = useTemplateResult ? new MCvScalar(0, 165, 255) : new MCvScalar(0, 255, 255);
            Point detectPt = useTemplateResult ? Point.Round(templateCenter) : Point.Round(best.basePoint);
            CvInvoke.Circle(resultImage, detectPt, 8, markerColor, -1);  // Filled
            CvInvoke.Circle(resultImage, detectPt, 8, new MCvScalar(0, 0, 0), 2);  // Black outline

            // Add text showing detection method and angle
            CvInvoke.PutText(resultImage, $"Arrow: {snappedAngle}deg ({detectionMethod})", new Point(10, 30),
                FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);

            // === ROTATE IMAGE TO ALIGN ARROW ===
            // Target: rotate so arrow points to 0° (right direction)
            double targetAngle = 0.0;
            double rotationAngle = targetAngle - finalArrowAngle;

            Console.WriteLine($"\n=== ROTATION ===");
            Console.WriteLine($"  Detected angle: {finalArrowAngle:F1}°");
            Console.WriteLine($"  Target angle: {targetAngle}°");
            Console.WriteLine($"  Rotation needed: {rotationAngle:F1}°");

            // Create rotation matrix around the ring center
            var rotationMatrix = new Mat();
            CvInvoke.GetRotationMatrix2D(new PointF(cx, cy), -rotationAngle, 1.0, rotationMatrix);

            // Rotate the original image
            var rotatedImage = new Image<Gray, byte>(original.Size);
            CvInvoke.WarpAffine(original, rotatedImage, rotationMatrix, original.Size,
                Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

            // Draw reference lines on rotated image
            var rotatedResult = rotatedImage.Convert<Bgr, byte>();

            // Draw ring boundary
            CvInvoke.Circle(rotatedResult, new Point(cx, cy), (int)outerR, new MCvScalar(0, 255, 0), 2);
            CvInvoke.Circle(rotatedResult, new Point(cx, cy), (int)innerR, new MCvScalar(0, 255, 0), 2);

            // Draw arrow at target position (180° = left)
            double targetRad = targetAngle * Math.PI / 180;
            int arrowEndX = (int)(cx + outerR * 1.2 * Math.Cos(targetRad));
            int arrowEndY = (int)(cy + outerR * 1.2 * Math.Sin(targetRad));
            CvInvoke.ArrowedLine(rotatedResult, new Point(cx, cy), new Point(arrowEndX, arrowEndY),
                new MCvScalar(0, 255, 0), 3);

            // Draw 24 sector lines for reference
            for (int s = 0; s < 24; s++)
            {
                double sectorRad = s * 15.0 * Math.PI / 180;
                int sectorEndX = (int)(cx + outerR * Math.Cos(sectorRad));
                int sectorEndY = (int)(cy + outerR * Math.Sin(sectorRad));
                var sectorColor = (s == 0) ? new MCvScalar(0, 255, 255) : new MCvScalar(128, 128, 128);  // Highlight sector 0 (right)
                int thickness = (s == 0) ? 2 : 1;
                CvInvoke.Line(rotatedResult, new Point(cx, cy), new Point(sectorEndX, sectorEndY), sectorColor, thickness);
            }

            CvInvoke.PutText(rotatedResult, $"Rotated: arrow at 0deg (right)", new Point(10, 30),
                FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);
            CvInvoke.PutText(rotatedResult, $"(rotated {rotationAngle:F1}deg)", new Point(10, 60),
                FontFace.HersheySimplex, 0.6, new MCvScalar(0, 255, 255), 1);

            rotatedResult.Save(outputPrefix + "_rotated.png");
            Console.WriteLine($"Rotated image saved to: {outputPrefix}_rotated.png");

            // === PREPROCESSING FOR ROTATED IMAGE DATA MARKS ===
            // Strategy: Calculate threshold only for the data ring area (white ring with black marks)

            // Step 1: Create mask for data ring area only (rotated)
            var rotDataMask = new Image<Gray, byte>(rotatedImage.Size);
            CvInvoke.Circle(rotDataMask, new Point(cx, cy), (int)outerR, new MCvScalar(255), -1);
            CvInvoke.Circle(rotDataMask, new Point(cx, cy), (int)innerR, new MCvScalar(0), -1);

            // Step 2: Apply CLAHE for uneven lighting correction
            var rotEnhanced = new Mat();
            CvInvoke.CLAHE(rotatedImage, 2.0, new Size(8, 8), rotEnhanced);
            var rotEnhancedImg = rotEnhanced.ToImage<Gray, byte>();

            // Step 3: Collect pixel values from ring area only
            var rotRingPixels = new List<byte>();
            for (int ry = 0; ry < rotEnhancedImg.Height; ry++)
            {
                for (int rx = 0; rx < rotEnhancedImg.Width; rx++)
                {
                    if (rotDataMask.Data[ry, rx, 0] > 0)
                    {
                        rotRingPixels.Add(rotEnhancedImg.Data[ry, rx, 0]);
                    }
                }
            }

            // Calculate statistics for ring area
            double rotMeanValue = rotRingPixels.Average(p => (double)p);
            double rotStdDev = Math.Sqrt(rotRingPixels.Average(p => Math.Pow(p - rotMeanValue, 2)));
            double rotCalculatedThresh = rotMeanValue - 1.5 * rotStdDev;
            byte rotOptimalThresh = (byte)Math.Max(80, Math.Min(180, rotCalculatedThresh));

            Console.WriteLine($"\n=== ROTATED IMAGE THRESHOLDING ===");
            Console.WriteLine($"  Mean brightness: {rotMeanValue:F1}");
            Console.WriteLine($"  Std dev: {rotStdDev:F1}");
            Console.WriteLine($"  Applied threshold: {rotOptimalThresh}");

            // Step 4: Apply binary threshold to find dark marks
            var rotBinaryResult = new Image<Gray, byte>(rotEnhancedImg.Size);
            CvInvoke.Threshold(rotEnhancedImg, rotBinaryResult, rotOptimalThresh, 255, ThresholdType.BinaryInv);

            // Step 5: Apply mask to get only marks in ring area
            var maskedRotatedBinary = new Image<Gray, byte>(rotatedImage.Size);
            CvInvoke.BitwiseAnd(rotBinaryResult, rotDataMask, maskedRotatedBinary);

            // Step 6: Morphological cleanup (reduced CLOSE to preserve Y-shape)
            var rotKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(maskedRotatedBinary, maskedRotatedBinary, MorphOp.Open, rotKernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
            CvInvoke.MorphologyEx(maskedRotatedBinary, maskedRotatedBinary, MorphOp.Close, rotKernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            // Save debug images
            rotEnhancedImg.Save(outputPrefix + "_clahe.png");
            rotBinaryResult.Save(outputPrefix + "_binary.png");
            maskedRotatedBinary.Save(outputPrefix + "_masked.png");
            Console.WriteLine($"Debug images saved: _clahe.png, _binary.png, _masked.png");

            // Find contours of data marks
            using var dataContours = new VectorOfVectorOfPoint();
            using var dataHierarchy = new Mat();
            CvInvoke.FindContours(maskedRotatedBinary.Clone(), dataContours, dataHierarchy,
                RetrType.External, ChainApproxMethod.ChainApproxSimple);

            // Draw contours on result
            var finalResult = rotatedImage.Convert<Bgr, byte>();

            // Draw ring boundary
            CvInvoke.Circle(finalResult, new Point(cx, cy), (int)outerR, new MCvScalar(0, 255, 0), 2);
            CvInvoke.Circle(finalResult, new Point(cx, cy), (int)innerR, new MCvScalar(0, 255, 0), 2);

            // Draw all data mark contours in CYAN
            CvInvoke.DrawContours(finalResult, dataContours, -1, new MCvScalar(255, 255, 0), 2);

            // === HALCON SAMPLING METHOD ===
            // Ring structure (based on HALCON):
            // - Inner ring: from innerR to centerR
            // - Outer ring: from centerR to outerR
            float centerR = (innerR + outerR) / 2;  // Divides inner/outer layers
            float innerSampleR = (innerR + centerR) / 2;  // Midpoint of inner layer
            float outerSampleR = (centerR + outerR) / 2;  // Midpoint of outer layer

            Console.WriteLine($"\n=== HALCON SAMPLING ===");
            Console.WriteLine($"  innerR={innerR:F0}, centerR={centerR:F0}, outerR={outerR:F0}");
            Console.WriteLine($"  Inner sample at: {innerSampleR:F0}");
            Console.WriteLine($"  Outer sample at: {outerSampleR:F0}");

            // Draw center ring boundary (magenta)
            CvInvoke.Circle(finalResult, new Point(cx, cy), (int)centerR, new MCvScalar(255, 0, 255), 1);

            // Use dense sampling like HALCON (multiple samples per sector)
            const int ANGULAR_SAMPLES = 5;  // Samples across sector width
            const int RADIAL_SAMPLES = 3;   // Samples across layer depth
            const double FILL_THRESHOLD = 0.40;  // 40% threshold

            var binaryString = new System.Text.StringBuilder();

            Console.WriteLine($"\n  Sect | Inner | Outer | Bits");
            Console.WriteLine($"  -----|-------|-------|------");

            for (int s = 0; s < 24; s++)
            {
                double sectorCenterRad = s * 15.0 * Math.PI / 180;
                double sectorHalfWidth = 7.5 * Math.PI / 180;  // Half of 15 degrees
                double angleMargin = sectorHalfWidth * 0.2;  // 20% margin from edges

                // Sample inner layer
                int innerDarkCount = 0, innerTotalCount = 0;
                for (int ai = 0; ai < ANGULAR_SAMPLES; ai++)
                {
                    double angleOffset = -sectorHalfWidth + angleMargin +
                        (2 * (sectorHalfWidth - angleMargin)) * ai / (ANGULAR_SAMPLES - 1);
                    double angle = sectorCenterRad + angleOffset;

                    for (int ri = 0; ri < RADIAL_SAMPLES; ri++)
                    {
                        float r = innerR + (centerR - innerR) * (0.2f + 0.6f * ri / (RADIAL_SAMPLES - 1));
                        int px = (int)(cx + r * Math.Cos(angle));
                        int py = (int)(cy + r * Math.Sin(angle));

                        if (px >= 0 && px < maskedRotatedBinary.Width && py >= 0 && py < maskedRotatedBinary.Height)
                        {
                            innerTotalCount++;
                            // Use binary mask (white=255 means mark detected)
                            if (maskedRotatedBinary.Data[py, px, 0] > 128)
                                innerDarkCount++;
                        }
                    }
                }

                // Sample outer layer
                int outerDarkCount = 0, outerTotalCount = 0;
                for (int ai = 0; ai < ANGULAR_SAMPLES; ai++)
                {
                    double angleOffset = -sectorHalfWidth + angleMargin +
                        (2 * (sectorHalfWidth - angleMargin)) * ai / (ANGULAR_SAMPLES - 1);
                    double angle = sectorCenterRad + angleOffset;

                    for (int ri = 0; ri < RADIAL_SAMPLES; ri++)
                    {
                        float r = centerR + (outerR - centerR) * (0.2f + 0.6f * ri / (RADIAL_SAMPLES - 1));
                        int px = (int)(cx + r * Math.Cos(angle));
                        int py = (int)(cy + r * Math.Sin(angle));

                        if (px >= 0 && px < maskedRotatedBinary.Width && py >= 0 && py < maskedRotatedBinary.Height)
                        {
                            outerTotalCount++;
                            // Use binary mask (white=255 means mark detected)
                            if (maskedRotatedBinary.Data[py, px, 0] > 128)
                                outerDarkCount++;
                        }
                    }
                }

                double innerRatio = innerTotalCount > 0 ? (double)innerDarkCount / innerTotalCount : 0;
                double outerRatio = outerTotalCount > 0 ? (double)outerDarkCount / outerTotalCount : 0;

                int innerBit = innerRatio >= FILL_THRESHOLD ? 1 : 0;
                int outerBit = outerRatio >= FILL_THRESHOLD ? 1 : 0;

                binaryString.Append(innerBit);
                binaryString.Append(outerBit);

                Console.WriteLine($"  {s,4} | {innerRatio,5:P0} | {outerRatio,5:P0} | {innerBit}{outerBit}");

                // Draw sample points at sector center
                int innerX = (int)(cx + innerSampleR * Math.Cos(sectorCenterRad));
                int innerY = (int)(cy + innerSampleR * Math.Sin(sectorCenterRad));
                int outerX = (int)(cx + outerSampleR * Math.Cos(sectorCenterRad));
                int outerY = (int)(cy + outerSampleR * Math.Sin(sectorCenterRad));

                var innerColor = innerBit == 1 ? new MCvScalar(0, 0, 255) : new MCvScalar(0, 255, 0);
                var outerColor = outerBit == 1 ? new MCvScalar(0, 0, 255) : new MCvScalar(0, 255, 0);

                CvInvoke.Circle(finalResult, new Point(innerX, innerY), 5, innerColor, -1);
                CvInvoke.Circle(finalResult, new Point(outerX, outerY), 5, outerColor, -1);

                // Add sector number
                int labelX = (int)(cx + (outerR + 15) * Math.Cos(sectorCenterRad));
                int labelY = (int)(cy + (outerR + 15) * Math.Sin(sectorCenterRad));
                CvInvoke.PutText(finalResult, $"{s}", new Point(labelX - 5, labelY + 5),
                    FontFace.HersheySimplex, 0.35, new MCvScalar(255, 255, 255), 1);
            }

            Console.WriteLine($"\n  Binary: {binaryString}");

            // Draw arrow direction
            CvInvoke.ArrowedLine(finalResult, new Point(cx, cy),
                new Point((int)(cx + outerR * 1.2), cy), new MCvScalar(0, 255, 0), 3);

            CvInvoke.PutText(finalResult, $"Data marks + sample points", new Point(10, 30),
                FontFace.HersheySimplex, 0.7, new MCvScalar(0, 255, 0), 2);
            CvInvoke.PutText(finalResult, $"RED=mark(1), GREEN=empty(0)", new Point(10, 55),
                FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 1);

            finalResult.Save(outputPrefix + "_final.png");
            Console.WriteLine($"Final result saved to: {outputPrefix}_final.png");
        }

        resultImage.Save(outputPrefix + "_result.png");
        Console.WriteLine($"Result saved to: {outputPrefix}_result.png");

        return finalArrowAngle;
    }

    static double CalculateArrowScore(double solidity, double centroidDistRatio, double tipDistRatio,
        double area, float outerR, bool tipPointsOutward, double elongation)
    {
        // Direction score - tip should point outward
        double directionScore = tipPointsOutward ? 1.0 : 0.3;

        // Tip position score - arrow tip should be at OUTER part of data ring (0.80-0.98R)
        double tipPositionScore;
        if (tipDistRatio >= 0.80 && tipDistRatio <= 0.98)
            tipPositionScore = 1.0;
        else if (tipDistRatio >= 0.70 && tipDistRatio <= 1.02)
            tipPositionScore = 0.6;
        else
            tipPositionScore = 0.2;

        // Centroid position score - arrow body should be in data ring (0.65-0.95R)
        double centroidScore;
        if (centroidDistRatio >= 0.70 && centroidDistRatio <= 0.92)
            centroidScore = 1.0;
        else if (centroidDistRatio >= 0.60 && centroidDistRatio <= 0.98)
            centroidScore = 0.6;
        else
            centroidScore = 0.3;

        // Solidity score - arrow (Y/triangle) has LOW solidity (0.4-0.78)
        // Data marks are more rectangular and have HIGH solidity (0.90-1.0)
        // This is the MOST IMPORTANT distinguishing feature
        double solidityScore;
        if (solidity < 0.55)
            solidityScore = 1.0;  // Very likely arrow
        else if (solidity < 0.65)
            solidityScore = 0.9;
        else if (solidity < 0.75)
            solidityScore = 0.7;
        else if (solidity < 0.82)
            solidityScore = 0.4;
        else if (solidity < 0.90)
            solidityScore = 0.15;  // Marginal - might be arrow with filled gaps
        else
            solidityScore = 0.0;  // Very HIGH solidity = definitely NOT arrow

        // Elongation score - arrow is elongated (radially stretched)
        double elongationScore;
        if (elongation >= 1.5 && elongation <= 4.0)
            elongationScore = 1.0;
        else if (elongation >= 1.0 && elongation <= 5.0)
            elongationScore = 0.6;
        else
            elongationScore = 0.2;

        // Area score
        double expectedArea = outerR * outerR * 0.015;
        double areaRatio = area / expectedArea;
        double areaScore = (areaRatio >= 0.2 && areaRatio <= 5.0) ? 0.8 : 0.3;

        // Combined score - balance solidity AND elongation
        // Both low solidity AND high elongation are needed for arrow
        double combinedSolidityElongation = (solidityScore + elongationScore) / 2;
        if (solidity < 0.80 && elongation >= 1.4)
            combinedSolidityElongation += 0.2;  // Bonus for having both good traits

        return directionScore * 0.10 + tipPositionScore * 0.15 +
               centroidScore * 0.10 + combinedSolidityElongation * 0.55 +
               areaScore * 0.10;
    }

    // Template images for multi-angle template matching (HALCON-style find_scaled_shape_model)
    static Image<Gray, byte> _arrowTemplateDark = null;
    static Image<Gray, byte> _arrowTemplateLight = null;
    static Image<Gray, byte> _arrowTemplate = null;  // Currently active template
    static int _templateSize = 100;  // Standard template size (from TemplateCreatorPage)
    static double _templateBaseAngle = 270.0;  // Template arrow points UP (270° in image coordinates)

    /// <summary>
    /// Load Y-shaped arrow templates from MAUI app's LocalCache
    /// Templates were created by user via TemplateCreatorPage
    /// </summary>
    static VectorOfPoint CreateArrowTemplate()
    {
        // MAUI app stores templates in LocalCache
        string basePath = @"C:\Users\qoose\AppData\Local\Packages\41d95e81-9b20-46b4-9997-73aed51e4d49_9zz4h110yvjzm\LocalCache\Local";
        string darkPath = Path.Combine(basePath, "arrow_template_dark.png");
        string lightPath = Path.Combine(basePath, "arrow_template_light.png");

        // Load dark template (white arrow on dark background)
        if (File.Exists(darkPath))
        {
            _arrowTemplateDark = new Image<Gray, byte>(darkPath);
            _templateSize = Math.Max(_arrowTemplateDark.Width, _arrowTemplateDark.Height);
            Console.WriteLine($"Loaded DARK arrow template: {_arrowTemplateDark.Width}x{_arrowTemplateDark.Height}");
        }
        else
        {
            Console.WriteLine($"WARNING: Dark template not found at {darkPath}");
        }

        // Load light template (dark arrow on light background)
        if (File.Exists(lightPath))
        {
            _arrowTemplateLight = new Image<Gray, byte>(lightPath);
            Console.WriteLine($"Loaded LIGHT arrow template: {_arrowTemplateLight.Width}x{_arrowTemplateLight.Height}");
        }
        else
        {
            Console.WriteLine($"WARNING: Light template not found at {lightPath}");
        }

        // Use dark template as default (will switch based on ring type)
        _arrowTemplate = _arrowTemplateDark ?? _arrowTemplateLight;

        if (_arrowTemplate == null)
        {
            Console.WriteLine("ERROR: No arrow templates found! Creating fallback...");
            // Fallback: create programmatic template
            _templateSize = 60;
            _arrowTemplate = new Image<Gray, byte>(_templateSize, _templateSize);
            _arrowTemplate.SetZero();

            int cx = _templateSize / 2;
            int cy = _templateSize / 2;
            var points = new Point[]
            {
                new Point(cx, cy - 25),       // Tip (top - points UP = 270°)
                new Point(cx - 20, cy + 5),   // Left branch end
                new Point(cx - 8, cy - 5),    // Left branch inner
                new Point(cx + 8, cy - 5),    // Right branch inner
                new Point(cx + 20, cy + 5),   // Right branch end
            };

            using (var vp = new VectorOfPoint(points))
            using (var vvp = new VectorOfVectorOfPoint())
            {
                vvp.Push(vp);
                CvInvoke.FillPoly(_arrowTemplate, vvp, new MCvScalar(255));
            }
        }

        // Save active template for debugging
        _arrowTemplate.Save(@"C:\Users\qoose\Desktop\ArrowDetectionTest\arrow_template_active.png");
        Console.WriteLine($"Active template size: {_templateSize}x{_templateSize}, base angle: {_templateBaseAngle}°");

        // Return dummy contour (not used for actual template matching)
        return new VectorOfPoint(new Point[] { new Point(0, 0) });
    }

    /// <summary>
    /// Find arrow using multi-angle template matching (HALCON find_scaled_shape_model equivalent)
    /// Searches the ring region at multiple angles and scales
    /// </summary>
    static (double bestScore, double bestAngle, PointF bestCenter) FindArrowByTemplateMatching(
        Image<Gray, byte> binaryImage, PointF ringCenter, float innerR, float outerR, bool isLightRing = false)
    {
        // The binary image has WHITE marks on BLACK background (due to BinaryInv threshold)
        // For LIGHT ring: arrow is DARK on original -> WHITE in binary -> use LIGHT template (inverted)
        // For DARK ring: arrow is WHITE on original -> WHITE in binary -> use DARK template
        Image<Gray, byte> activeTemplate;
        string templateName;

        if (isLightRing && _arrowTemplateLight != null)
        {
            // Light ring: invert the LIGHT template to match white-on-black binary
            activeTemplate = _arrowTemplateLight.Clone();
            CvInvoke.BitwiseNot(activeTemplate, activeTemplate);
            templateName = "LIGHT (inverted)";
        }
        else
        {
            // Dark ring or fallback: use DARK template directly (already white-on-black)
            activeTemplate = _arrowTemplateDark ?? _arrowTemplateLight;
            templateName = "DARK";
        }

        if (activeTemplate == null)
        {
            Console.WriteLine("  [TemplateMatch] ERROR: Template not loaded!");
            return (0, 0, ringCenter);
        }

        int cx = (int)ringCenter.X;
        int cy = (int)ringCenter.Y;

        double bestScore = 0;
        double bestRotation = 0;  // Rotation angle applied to template
        PointF bestMatchCenter = ringCenter;

        // Scale factors to try (arrow size varies with ring size)
        // Expected arrow size is approximately (outerR - innerR) * 0.4 to 0.6
        float ringWidth = outerR - innerR;
        float expectedArrowSize = ringWidth * 0.5f;
        float baseScale = expectedArrowSize / (float)activeTemplate.Width;

        // More scale variations for better matching
        double[] scales = { baseScale * 0.6, baseScale * 0.75, baseScale * 0.9, baseScale, baseScale * 1.1, baseScale * 1.25, baseScale * 1.4 };

        Console.WriteLine($"  [TemplateMatch] Using {templateName} template ({activeTemplate.Width}x{activeTemplate.Height})");
        Console.WriteLine($"  [TemplateMatch] Searching angles 0-360° in 5° steps, scales: {baseScale*0.6:F2} to {baseScale*1.4:F2}");

        // Search all 360 degrees in steps
        const double angleStep = 5.0;
        int searchCount = 0;

        foreach (double scale in scales)
        {
            // Skip invalid scales
            if (scale < 0.2 || scale > 4.0) continue;

            int scaledWidth = Math.Max(15, (int)(activeTemplate.Width * scale));
            int scaledHeight = Math.Max(15, (int)(activeTemplate.Height * scale));

            // Skip if template would be too large
            if (scaledWidth > binaryImage.Width / 2 || scaledHeight > binaryImage.Height / 2) continue;

            // Create scaled template
            var scaledTemplate = new Image<Gray, byte>(scaledWidth, scaledHeight);
            CvInvoke.Resize(activeTemplate, scaledTemplate, new Size(scaledWidth, scaledHeight), 0, 0, Inter.Linear);

            for (double rotation = 0; rotation < 360; rotation += angleStep)
            {
                searchCount++;

                // Rotate template around its center
                // Positive rotation = counter-clockwise
                var rotationMat = new Mat();
                CvInvoke.GetRotationMatrix2D(new PointF(scaledWidth / 2f, scaledHeight / 2f), rotation, 1.0, rotationMat);

                var rotatedTemplate = new Image<Gray, byte>(scaledTemplate.Size);
                CvInvoke.WarpAffine(scaledTemplate, rotatedTemplate, rotationMat, scaledTemplate.Size,
                    Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

                // Template matching using TM_CCOEFF_NORMED (best for binary images)
                using var matchResult = new Mat();
                CvInvoke.MatchTemplate(binaryImage, rotatedTemplate, matchResult, TemplateMatchingType.CcoeffNormed);

                // Find best match location
                double minVal = 0, maxVal = 0;
                Point minLoc = new Point(), maxLoc = new Point();
                CvInvoke.MinMaxLoc(matchResult, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                // Calculate match center
                float matchCenterX = maxLoc.X + scaledWidth / 2f;
                float matchCenterY = maxLoc.Y + scaledHeight / 2f;

                // Verify match is in the ring area (between inner and outer radius)
                float distFromRingCenter = (float)Math.Sqrt(
                    Math.Pow(matchCenterX - cx, 2) + Math.Pow(matchCenterY - cy, 2));

                bool inRingArea = distFromRingCenter >= innerR * 0.5f && distFromRingCenter <= outerR * 1.15f;

                if (maxVal > bestScore && inRingArea)
                {
                    bestScore = maxVal;
                    bestRotation = rotation;
                    bestMatchCenter = new PointF(matchCenterX, matchCenterY);
                }

                rotatedTemplate.Dispose();
                rotationMat.Dispose();
            }

            scaledTemplate.Dispose();
        }

        // Calculate final arrow angle from MATCH POSITION relative to ring center
        // The Y-arrow on a ring always points OUTWARD from the ring center
        // So the arrow direction is the angle from ring center to match center
        double dx = bestMatchCenter.X - cx;
        double dy = bestMatchCenter.Y - cy;
        double finalAngle = Math.Atan2(dy, dx) * 180.0 / Math.PI;
        if (finalAngle < 0) finalAngle += 360;

        Console.WriteLine($"  [TemplateMatch] Searched {searchCount} angle/scale combinations");
        Console.WriteLine($"  [TemplateMatch] Best match: score={bestScore:F3}, rotation={bestRotation:F1}°");
        Console.WriteLine($"  [TemplateMatch] Match center: ({bestMatchCenter.X:F0},{bestMatchCenter.Y:F0}) relative to ring ({cx},{cy})");
        Console.WriteLine($"  [TemplateMatch] Arrow direction: atan2({dy:F0},{dx:F0}) = {finalAngle:F1}°");

        return (bestScore, finalAngle, bestMatchCenter);
    }

    /// <summary>
    /// Calculate shape match score using Hu Moments (fast, rotation-invariant)
    /// This is a quick pre-filter before expensive template matching
    /// </summary>
    static double CalculateShapeMatchScore(VectorOfPoint contour, VectorOfPoint template)
    {
        try
        {
            // Use Hu moments for rotation-invariant shape comparison
            double matchScore = CvInvoke.MatchShapes(contour, template, ContoursMatchType.I1, 0);

            // Convert to similarity score (lower matchScore = better match)
            // MatchShapes returns 0 for perfect match, higher for worse
            if (matchScore < 0.1)
                return 1.0;  // Very good match
            else if (matchScore < 0.2)
                return 0.8;
            else if (matchScore < 0.3)
                return 0.6;
            else if (matchScore < 0.5)
                return 0.4;
            else if (matchScore < 1.0)
                return 0.2;
            else
                return 0.1;
        }
        catch
        {
            return 0.1;
        }
    }
}
