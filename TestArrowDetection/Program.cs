using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

class Program
{
    static void Main(string[] args)
    {
        string testDir = @"C:\Users\qoose\Desktop\文件資料\客戶分類\R-RCM\01_Software\RCM\03_Document\NG\20260113-002\NG";
        string outputDir = @"C:\Users\qoose\Desktop\ArrowDetectionTest";

        Directory.CreateDirectory(outputDir);

        // Test specific image
        var testFiles = new[] {
            Path.Combine(testDir, "20260100040000601.png")
        };

        Console.WriteLine($"Testing {testFiles.Length} images...\n");

        foreach (var testImagePath in testFiles)
        {
            string fileName = Path.GetFileNameWithoutExtension(testImagePath);
            Console.WriteLine($"\n===== Testing: {fileName} =====");

            // Load image
            var original = new Image<Gray, byte>(testImagePath);
            Console.WriteLine($"Image size: {original.Width}x{original.Height}");

            TestSingleImage(original, Path.Combine(outputDir, fileName), fileName);

            original.Dispose();
        }

        Console.WriteLine("\n\n===== All tests completed =====");
    }

    static void TestSingleImage(Image<Gray, byte> original, string outputPrefix, string fileName)
    {
        // Step 1: Find ring using simple method
        var (center, outerRadius, innerRadius) = FindRing(original);
        Console.WriteLine($"Ring found: center=({center.X:F0}, {center.Y:F0}), outerR={outerRadius:F0}, innerR={innerRadius:F0}");

        // Save preprocessing overlay for visual inspection
        SavePreprocessingOverlay(original, center, outerRadius, innerRadius, outputPrefix);

        // Step 2: Test arrow detection
        var arrowResult = TestArrowDetection(original, center, outerRadius, innerRadius, outputPrefix);
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

    static double TestArrowDetection(Image<Gray, byte> original, PointF center, float outerR, float innerR, string outputPrefix)
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

        // Step 6: Morphological cleanup
        var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
        CvInvoke.MorphologyEx(maskedBinary, maskedBinary, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        var largerKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(5, 5), new Point(-1, -1));
        CvInvoke.MorphologyEx(maskedBinary, maskedBinary, MorphOp.Close, largerKernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));

        // Save preprocessed binary for arrow detection
        maskedBinary.Save(outputPrefix + "_arrow_binary.png");

        // Find contours and analyze
        var resultImage = original.Convert<Bgr, byte>();

        // Draw search zone
        CvInvoke.Circle(resultImage, new Point(cx, cy), (int)outerR, new MCvScalar(0, 255, 255), 1);
        CvInvoke.Circle(resultImage, new Point(cx, cy), (int)innerR, new MCvScalar(0, 255, 255), 1);

        // Find contours from preprocessed binary
        var allCandidates = new List<(double score, double solidity, double angle, double area,
            double centroidDist, double tipDist, VectorOfPoint contour, PointF basePoint, string method)>();

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
            double angle = Math.Atan2(basePoint.Y - cy, basePoint.X - cx) * 180 / Math.PI;
            if (angle < 0) angle += 360;

            // Check if tip points outward (arrow characteristic)
            bool tipPointsOutward = maxTipDist > minBaseDist && (maxTipDist - minBaseDist) > 3;

            // Calculate elongation ratio (tip-base distance vs area)
            double elongation = (maxTipDist - minBaseDist) / Math.Sqrt(area);

            // Score calculation - prioritize arrow characteristics (low solidity, elongation)
            double score = CalculateArrowScore(solidity, centroidDistRatio, tipDistRatio, area, outerR, tipPointsOutward, elongation);

            Console.WriteLine($"  Contour {i}: area={area:F0}, solidity={solidity:F3}, " +
                $"centroid={centroidDistRatio:F2}R, tip={tipDistRatio:F2}R, angle={angle:F0}°, " +
                $"elongation={elongation:F2}, score={score:F2}");

            if (score > 0.1)
            {
                var contourCopy = new VectorOfPoint(contour.ToArray());
                allCandidates.Add((score, solidity, angle, area, centroidDistRatio, tipDistRatio,
                    contourCopy, basePoint, "Preprocessed"));
            }
        }

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
            var color = i == 0 ? new MCvScalar(0, 0, 255) :  // Red for best
                       i == 1 ? new MCvScalar(255, 0, 0) :   // Blue for 2nd
                       new MCvScalar(128, 128, 128);         // Gray for others

            CvInvoke.DrawContours(resultImage, new VectorOfVectorOfPoint(c.contour), -1, color, 2);
            CvInvoke.Circle(resultImage, new Point((int)c.basePoint.X, (int)c.basePoint.Y), 5, color, -1);

            // Draw line from center through base point (arrow direction)
            CvInvoke.Line(resultImage, new Point(cx, cy), new Point((int)c.basePoint.X, (int)c.basePoint.Y), color, 1);

            Console.WriteLine($"  #{i+1}: score={c.score:F2}, angle={c.angle:F0}°, " +
                $"solidity={c.solidity:F3}, elongation, method={c.method}");
        }

        if (sorted.Count > 0)
        {
            var best = sorted[0];
            int sector = (int)Math.Round(best.angle / 15.0) % 24;
            double snappedAngle = sector * 15.0;

            Console.WriteLine($"\n=== DETECTED ARROW: {best.angle:F1}° -> sector {sector} ({snappedAngle}°) ===");

            // Draw final arrow direction - from center through base point, extending outward
            // Use the actual base point direction (not snapped)
            double rad = best.angle * Math.PI / 180;
            int lineEndX = (int)(cx + outerR * 1.3 * Math.Cos(rad));
            int lineEndY = (int)(cy + outerR * 1.3 * Math.Sin(rad));
            CvInvoke.ArrowedLine(resultImage, new Point(cx, cy), new Point(lineEndX, lineEndY),
                new MCvScalar(0, 255, 0), 3);

            // Add text
            CvInvoke.PutText(resultImage, $"Arrow: {snappedAngle}deg", new Point(10, 30),
                FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);

            // === ROTATE IMAGE TO ALIGN ARROW ===
            // Target: rotate so arrow points to 0° (right direction)
            double targetAngle = 0.0;
            double rotationAngle = targetAngle - best.angle;

            Console.WriteLine($"\n=== ROTATION ===");
            Console.WriteLine($"  Detected angle: {best.angle:F1}°");
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

            // Step 6: Morphological operations to clean up noise and fill small holes
            var rotKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(maskedRotatedBinary, maskedRotatedBinary, MorphOp.Open, rotKernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
            var rotLargerKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(5, 5), new Point(-1, -1));
            CvInvoke.MorphologyEx(maskedRotatedBinary, maskedRotatedBinary, MorphOp.Close, rotLargerKernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));

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

        return sorted.Count > 0 ? sorted[0].angle : -1;
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

        // Solidity score - arrow (Y/triangle) has LOW solidity (0.4-0.7)
        // Data marks are more rectangular and have HIGH solidity (0.8-1.0)
        double solidityScore;
        if (solidity < 0.55)
            solidityScore = 1.0;  // Very likely arrow
        else if (solidity < 0.65)
            solidityScore = 0.8;
        else if (solidity < 0.75)
            solidityScore = 0.5;
        else
            solidityScore = 0.1;  // Likely data mark, not arrow

        // Elongation score - arrow is elongated (radially stretched)
        // Data marks are more compact
        double elongationScore;
        if (elongation >= 1.5 && elongation <= 4.0)
            elongationScore = 1.0;  // Good arrow elongation
        else if (elongation >= 1.0 && elongation <= 5.0)
            elongationScore = 0.6;
        else
            elongationScore = 0.2;

        // Area score - arrow has specific size range
        double expectedArea = outerR * outerR * 0.015;
        double areaRatio = area / expectedArea;
        double areaScore = (areaRatio >= 0.2 && areaRatio <= 5.0) ? 0.8 : 0.3;

        return directionScore * 0.15 + tipPositionScore * 0.20 +
               centroidScore * 0.10 + solidityScore * 0.30 +
               elongationScore * 0.15 + areaScore * 0.10;
    }
}
