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
    /// Ring code decoder using Emgu.CV
    /// Based on HALCON Arrow_Decode algorithm
    /// Ring code structure: 24 segments × 2 layers (inner/outer) = 48 bits
    /// </summary>
    public class RingCodeDecoder
    {
        // Ring code parameters (from HALCON code)
        private const int SEGMENTS = 24;
        private const double SEGMENT_ANGLE = 360.0 / SEGMENTS; // 15 degrees per segment

        // Fill threshold: if intersection >= 40% of reference area, it's a "1"
        private const double FILL_THRESHOLD = 0.40;

        // Logging - set EnableDetailedLog=false for faster processing
        public static Action<string> Log { get; set; } = (msg) => System.Diagnostics.Debug.WriteLine($"[Decoder] {msg}");
        public static bool EnableDetailedLog { get; set; } = false;  // Disabled for performance

        // Circle ratios based on HALCON code analysis
        // BigCircle = outer edge, CenterCircle = middle ring, InnerCircle = inner edge
        private const double OUTER_RATIO = 1.0;      // 100% - outer boundary
        private const double CENTER_RATIO = 0.75;    // 75% - divides inner/outer rings
        private const double INNER_RATIO = 0.50;     // 50% - inner boundary (hole edge)

        // Alternative circle ratio configurations to try
        private static readonly (double inner, double center, double outer)[] CIRCLE_CONFIGS = {
            (0.50, 0.75, 1.00),  // Original HALCON config
            (0.55, 0.775, 1.00), // Slightly larger hole
            (0.45, 0.725, 1.00), // Slightly smaller hole
            (0.50, 0.70, 0.90),  // Narrower ring band
            (0.55, 0.80, 1.00),  // Adjusted proportions
            (0.60, 0.80, 1.00),  // Larger hole, narrower rings
        };

        public class RingCodeResult
        {
            public string BinaryString { get; set; } = "";
            public string DecodedData { get; set; } = "";
            public PointF Center { get; set; }
            public float OuterRadius { get; set; }
            public float InnerRadius { get; set; }
            public float MiddleRadius { get; set; }
            public double RotationAngle { get; set; }
            public bool IsValid { get; set; }
            public string ErrorMessage { get; set; } = "";
            public Image<Bgr, byte> ProcessedImage { get; set; }
            public List<PointF> LocatorPoints { get; set; } = new();
            public int RingIndex { get; set; }
            public Image<Gray, byte> ForegroundMask { get; set; }  // For debugging
        }

        /// <summary>
        /// Decode a single ring region with multi-rotation search
        /// Tries multiple rotation angles and picks the one that passes BCC validation
        /// </summary>
        public RingCodeResult DecodeRing(Image<Gray, byte> grayImage, RingImageSegmentation.RingRegion region)
        {
            var result = new RingCodeResult
            {
                RingIndex = region.Index,
                Center = region.Center,
                OuterRadius = region.OuterRadius,
                InnerRadius = region.InnerRadius,
                LocatorPoints = region.TrianglePoints
            };

            Log($"========== Ring #{region.Index} ==========");
            Log($"Center: ({region.Center.X:F1}, {region.Center.Y:F1}), OuterR: {region.OuterRadius:F1}");
            Log($"Triangle points: {region.TrianglePoints.Count}");

            try
            {
                // Step 1: Extract ring region
                var ringImage = ExtractRingRegion(grayImage, region);
                Log($"Step1: Ring region extracted");

                // Step 2: Extract white foreground region
                var foregroundMask = ExtractForegroundRegion(ringImage, region);
                result.ForegroundMask = foregroundMask;  // Store for visualization

                // Count white pixels for debugging
                int whiteCount = 0;
                for (int y = 0; y < foregroundMask.Height; y++)
                    for (int x = 0; x < foregroundMask.Width; x++)
                        if (foregroundMask.Data[y, x, 0] > 128) whiteCount++;
                Log($"Step2: Foreground extracted, white pixels: {whiteCount}");

                // Step 3: Find arrow/triangle and calculate base rotation angle
                double baseAngle = FindArrowAngle(foregroundMask, ringImage, region);
                Log($"Step3: Base rotation angle: {baseAngle:F1}°");

                // Step 4 & 5: Fast decode with fallback
                // First try default config, only try alternatives if failed
                float bigRadius = region.OuterRadius;
                float innerRadius = bigRadius * (float)INNER_RATIO;
                float centerRadius = bigRadius * (float)CENTER_RATIO;

                string bestBinary = DecodeWithRegionIntersection(
                    foregroundMask, region.Center, innerRadius, centerRadius, bigRadius, baseAngle);
                string bestDecoded = DecryptBinaryToLong(bestBinary);
                double bestAngle = baseAngle;
                bool found = bestDecoded != "-1" && bestDecoded != "0";

                // Only try alternatives if first attempt failed
                if (!found)
                {
                    double[] angleOffsets = { 7.5, -7.5, 15, -15 }; // Skip 0, already tried
                    foreach (double offset in angleOffsets)
                    {
                        double testAngle = baseAngle + offset;
                        string binary = DecodeWithRegionIntersection(
                            foregroundMask, region.Center, innerRadius, centerRadius, bigRadius, testAngle);
                        string decoded = DecryptBinaryToLong(binary);

                        if (decoded != "-1" && decoded != "0")
                        {
                            bestBinary = binary;
                            bestDecoded = decoded;
                            bestAngle = testAngle;
                            found = true;
                            break;
                        }
                    }
                }

                if (found)
                {
                    Log($"  Angle {bestAngle:F1}°: VALID -> {bestDecoded}");
                }

                result.MiddleRadius = centerRadius;
                result.RotationAngle = bestAngle;
                result.BinaryString = bestBinary;
                result.DecodedData = bestDecoded;
                result.IsValid = found;
                Log($"Step4: Circles - Inner: {innerRadius:F1}, Center: {centerRadius:F1}, Outer: {bigRadius:F1}");

                Log($"Step5: Binary: {result.BinaryString}");
                Log($"Step6: Decoded: {result.DecodedData}, Valid: {result.IsValid}");

                return result;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = $"Decode error: {ex.Message}";
                Log($"ERROR: {ex.Message}");
                return result;
            }
        }

        /// <summary>
        /// Decode all rings in an image
        /// </summary>
        public List<RingCodeResult> DecodeAllRings(Image<Bgr, byte> sourceImage)
        {
            var results = new List<RingCodeResult>();

            var segmentation = new RingImageSegmentation();
            var segResult = segmentation.SegmentImage(sourceImage);

            if (!segResult.Success)
            {
                return results;
            }

            var grayImage = sourceImage.Convert<Gray, byte>();

            foreach (var ring in segResult.DetectedRings)
            {
                var decoded = DecodeRing(grayImage, ring);
                decoded.ProcessedImage = CreateVisualization(sourceImage.Clone(), decoded);
                results.Add(decoded);
            }

            return results;
        }

        /// <summary>
        /// Extract ring region from image
        /// </summary>
        private Image<Gray, byte> ExtractRingRegion(Image<Gray, byte> source, RingImageSegmentation.RingRegion region)
        {
            // Create circular mask for the ring area
            var mask = new Image<Gray, byte>(source.Size);
            CvInvoke.Circle(mask, Point.Round(region.Center), (int)(region.OuterRadius * 1.1), new MCvScalar(255), -1);

            var masked = source.Copy(mask);
            return masked;
        }

        /// <summary>
        /// Extract white foreground region with CLAHE for lighting normalization
        /// </summary>
        private Image<Gray, byte> ExtractForegroundRegion(Image<Gray, byte> ringImage, RingImageSegmentation.RingRegion region)
        {
            int innerR = (int)(region.OuterRadius * 0.45);
            int outerR = (int)(region.OuterRadius * 1.02);
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;

            // Create ring mask
            var ringMask = new Image<Gray, byte>(ringImage.Size);
            CvInvoke.Circle(ringMask, new Point(cx, cy), outerR, new MCvScalar(255), -1);
            CvInvoke.Circle(ringMask, new Point(cx, cy), innerR, new MCvScalar(0), -1);

            // Step 1: Apply CLAHE to normalize lighting
            var normalized = new Image<Gray, byte>(ringImage.Size);
            CvInvoke.CLAHE(ringImage, 2.0, new System.Drawing.Size(8, 8), normalized);

            // Step 2: Collect histogram ONLY from ring area pixels (on normalized image)
            int[] histogram = new int[256];
            int pixelCount = 0;

            for (int y = Math.Max(0, cy - outerR); y < Math.Min(normalized.Height, cy + outerR); y++)
            {
                for (int x = Math.Max(0, cx - outerR); x < Math.Min(normalized.Width, cx + outerR); x++)
                {
                    double dist = Math.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    if (dist >= innerR && dist <= outerR)
                    {
                        byte val = normalized.Data[y, x, 0];
                        histogram[val]++;
                        pixelCount++;
                    }
                }
            }

            // Step 3: Calculate Otsu threshold for ring area only
            double threshold = CalculateOtsuThreshold(histogram, pixelCount);

            // Adjust threshold slightly higher to be more selective about "white"
            threshold = Math.Min(255, threshold + 5);

            // Step 4: Apply threshold
            var binary = new Image<Gray, byte>(ringImage.Size);
            for (int y = 0; y < normalized.Height; y++)
            {
                for (int x = 0; x < normalized.Width; x++)
                {
                    binary.Data[y, x, 0] = normalized.Data[y, x, 0] > threshold ? (byte)255 : (byte)0;
                }
            }

            // Step 5: Morphological opening to remove small noise
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Open, kernel,
                new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            // Apply ring mask to result
            var foreground = binary.Copy(ringMask);

            return foreground;
        }

        /// <summary>
        /// Calculate Otsu threshold for given histogram
        /// Finds the threshold that maximizes between-class variance
        /// </summary>
        private double CalculateOtsuThreshold(int[] histogram, int totalPixels)
        {
            // Calculate total sum
            double totalSum = 0;
            for (int i = 0; i < 256; i++)
                totalSum += i * histogram[i];

            double sumB = 0;
            int wB = 0;
            double maxVariance = 0;
            double bestThreshold = 128;

            for (int t = 0; t < 256; t++)
            {
                wB += histogram[t];
                if (wB == 0) continue;

                int wF = totalPixels - wB;
                if (wF == 0) break;

                sumB += t * histogram[t];
                double mB = sumB / wB;
                double mF = (totalSum - sumB) / wF;

                // Between-class variance
                double variance = (double)wB * wF * (mB - mF) * (mB - mF);

                if (variance > maxVariance)
                {
                    maxVariance = variance;
                    bestThreshold = t;
                }
            }

            return bestThreshold;
        }

        /// <summary>
        /// Find arrow/triangle angle using the TIP point (not centroid)
        /// </summary>
        private double FindArrowAngle(Image<Gray, byte> foreground, Image<Gray, byte> original,
            RingImageSegmentation.RingRegion region)
        {
            // Method 1: Use triangle TIP points if available (from segmentation)
            if (region.TrianglePoints.Count >= 1)
            {
                // Triangle points should already be TIP points (furthest from center)
                // Find the one furthest from center (should be the arrow tip)
                var arrowTip = region.TrianglePoints
                    .OrderByDescending(pt =>
                        Math.Sqrt(Math.Pow(pt.X - region.Center.X, 2) + Math.Pow(pt.Y - region.Center.Y, 2)))
                    .First();

                double dist = Math.Sqrt(Math.Pow(arrowTip.X - region.Center.X, 2) + Math.Pow(arrowTip.Y - region.Center.Y, 2));
                double angle = Math.Atan2(arrowTip.Y - region.Center.Y, arrowTip.X - region.Center.X);
                double angleDeg = angle * 180 / Math.PI;

                Log($"  Arrow TIP at ({arrowTip.X:F1}, {arrowTip.Y:F1}), dist={dist:F1}, angle={angleDeg:F1}°");
                return angleDeg;
            }

            // Method 2: Find arrow by analyzing foreground pattern
            Log($"  No triangle points, using contour analysis...");
            return FindArrowByContourAnalysis(foreground, region);
        }

        /// <summary>
        /// Find arrow angle by analyzing triangular shapes in the ring
        /// Uses the TIP of the arrow (furthest vertex from center) for precise angle
        /// </summary>
        private double FindArrowByContourAnalysis(Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            // Find contours in the foreground
            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(foreground.Clone(), contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            var candidates = new List<(PointF tipPoint, double area, double angle, double score, int vertices, double tipSharpness)>();

            // Calculate expected segment area for filtering
            double ringWidth = region.OuterRadius - region.InnerRadius;
            double segmentArea = Math.PI * ringWidth * ringWidth / 48;
            double minArea = segmentArea * 0.3;
            double maxArea = segmentArea * 10;

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);

                if (area < minArea || area > maxArea) continue;

                // Approximate polygon
                using var approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, approx, CvInvoke.ArcLength(contour, true) * 0.02, true);

                int vertices = approx.Size;

                // Triangles/arrows typically have 3-7 vertices
                if (vertices >= 3 && vertices <= 7)
                {
                    var moments = CvInvoke.Moments(contour);
                    if (moments.M00 > 0)
                    {
                        float cx = (float)(moments.M10 / moments.M00);
                        float cy = (float)(moments.M01 / moments.M00);

                        double centroidDist = Math.Sqrt(Math.Pow(cx - region.Center.X, 2) + Math.Pow(cy - region.Center.Y, 2));

                        // Centroid should be in the ring area
                        if (centroidDist > region.InnerRadius * 0.6 && centroidDist < region.OuterRadius * 1.1)
                        {
                            // Find the TIP - vertex furthest from ring center
                            var points = approx.ToArray();
                            PointF tipPoint = new PointF(cx, cy);
                            double maxDist = 0;
                            double tipSharpness = 180;
                            int tipIndex = -1;

                            for (int j = 0; j < points.Length; j++)
                            {
                                double dist = Math.Sqrt(Math.Pow(points[j].X - region.Center.X, 2) +
                                    Math.Pow(points[j].Y - region.Center.Y, 2));

                                if (dist > maxDist)
                                {
                                    maxDist = dist;
                                    tipPoint = new PointF(points[j].X, points[j].Y);
                                    tipIndex = j;
                                }
                            }

                            // Calculate tip sharpness (angle at the tip vertex)
                            if (tipIndex >= 0 && points.Length >= 3)
                            {
                                int prev = (tipIndex - 1 + points.Length) % points.Length;
                                int next = (tipIndex + 1) % points.Length;
                                tipSharpness = CalculateVertexAngle(points[prev], points[tipIndex], points[next]);
                            }

                            // Tip must be near the outer edge
                            if (maxDist > region.OuterRadius * 0.7 && maxDist < region.OuterRadius * 1.2)
                            {
                                // Calculate angle from ring center to tip
                                double angle = Math.Atan2(tipPoint.Y - region.Center.Y, tipPoint.X - region.Center.X);

                                // Score calculation
                                double score = area;

                                // Bonus for true triangles
                                if (vertices == 3) score *= 3.0;
                                else if (vertices == 4) score *= 2.0;
                                else if (vertices == 5) score *= 1.5;

                                // Bonus for sharp tip (small angle = more pointed)
                                if (tipSharpness < 50) score *= 3.0;
                                else if (tipSharpness < 70) score *= 2.0;
                                else if (tipSharpness < 90) score *= 1.5;

                                // Bonus for tip being far from center
                                double tipDistRatio = maxDist / region.OuterRadius;
                                score *= (tipDistRatio * tipDistRatio);

                                candidates.Add((tipPoint, area, angle * 180 / Math.PI, score, vertices, tipSharpness));
                            }
                        }
                    }
                }
            }

            // Return angle to the TIP of the best scoring shape
            if (candidates.Count > 0)
            {
                var best = candidates.OrderByDescending(c => c.score).First();
                Log($"  Arrow TIP found: vertices={best.vertices}, tipAngle={best.tipSharpness:F0}°, area={best.area:F0}, angle={best.angle:F1}°");
                return best.angle;
            }

            // Fallback: use intensity peak detection
            Log($"  No triangle contour found, using intensity peak detection...");
            return FindRotationByIntensityPeak(foreground, region);
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
        /// Find rotation angle by detecting white pixel density peaks around the ring (OPTIMIZED)
        /// </summary>
        private double FindRotationByIntensityPeak(Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            // Use 24 samples (one per segment) instead of 120
            int[] angleCounts = new int[24];
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;
            float innerR = region.InnerRadius * 0.6f;
            float outerR = region.OuterRadius * 0.95f;

            // Sample each segment
            for (int seg = 0; seg < 24; seg++)
            {
                double centerAngle = seg * 15 * Math.PI / 180; // 15 degrees per segment
                int count = 0;

                // Sample 3 angles within segment, 4 radii
                for (int ai = -1; ai <= 1; ai++)
                {
                    double angle = centerAngle + ai * 5 * Math.PI / 180;
                    double cosA = Math.Cos(angle);
                    double sinA = Math.Sin(angle);

                    for (int ri = 0; ri < 4; ri++)
                    {
                        float r = innerR + (outerR - innerR) * (ri + 0.5f) / 4;
                        int x = (int)(cx + r * cosA);
                        int y = (int)(cy + r * sinA);

                        if (x >= 0 && x < foreground.Width && y >= 0 && y < foreground.Height)
                        {
                            if (foreground.Data[y, x, 0] > 128)
                                count++;
                        }
                    }
                }
                angleCounts[seg] = count;
            }

            // Find peak
            int maxCount = 0, maxSeg = 0;
            int totalCount = 0;
            for (int i = 0; i < 24; i++)
            {
                totalCount += angleCounts[i];
                if (angleCounts[i] > maxCount)
                {
                    maxCount = angleCounts[i];
                    maxSeg = i;
                }
            }

            double avgCount = totalCount / 24.0;
            double resultAngle = maxSeg * 15.0;

            Log($"  Intensity peak at {resultAngle}° (count={maxCount}, avg={avgCount:F0})");
            return resultAngle;
        }

        /// <summary>
        /// Find rotation angle by analyzing intensity variance around the ring
        /// </summary>
        private double FindRotationByVariance(Image<Gray, byte> source, RingImageSegmentation.RingRegion region)
        {
            double maxVariance = 0;
            double bestAngle = 0;
            float sampleRadius = (region.OuterRadius + region.InnerRadius) / 2;

            for (int angle = 0; angle < 360; angle += 5)
            {
                double rad = angle * Math.PI / 180;
                var intensities = new List<int>();

                // Sample radially at this angle
                for (float r = region.InnerRadius * 0.8f; r < region.OuterRadius; r += 3)
                {
                    int x = (int)(region.Center.X + r * Math.Cos(rad));
                    int y = (int)(region.Center.Y + r * Math.Sin(rad));

                    if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                    {
                        intensities.Add(source.Data[y, x, 0]);
                    }
                }

                if (intensities.Count > 5)
                {
                    double avg = intensities.Average();
                    double variance = intensities.Sum(v => Math.Pow(v - avg, 2)) / intensities.Count;

                    if (variance > maxVariance)
                    {
                        maxVariance = variance;
                        bestAngle = angle;
                    }
                }
            }

            return bestAngle;
        }

        /// <summary>
        /// Decode using region intersection method (HALCON Arrow_Decode algorithm)
        /// Divides ring into 24 segments, each with inner and outer parts
        /// </summary>
        private string DecodeWithRegionIntersection(Image<Gray, byte> foreground, PointF center,
            float innerRadius, float centerRadius, float outerRadius, double startAngleDeg)
        {
            var binaryString = new System.Text.StringBuilder();
            double segmentRad = SEGMENT_ANGLE * Math.PI / 180;
            double startRad = startAngleDeg * Math.PI / 180;

            if (EnableDetailedLog)
            {
                Log($"  Decoding with startAngle={startAngleDeg:F1}°, threshold={FILL_THRESHOLD:P0}");
                Log($"  Seg# | Angle°  | Inner% | InBit | Outer% | OutBit");
                Log($"  -----|---------|--------|-------|--------|-------");
            }

            for (int i = 0; i < SEGMENTS; i++)
            {
                // Calculate segment boundaries
                // Note: In HALCON code, they use (hv_Phi * i) - hv_Phi to offset
                double angle1 = startRad + (i * segmentRad) - segmentRad;
                double angle2 = startRad + ((i + 1) * segmentRad) - segmentRad;
                double midAngleDeg = ((angle1 + angle2) / 2) * 180 / Math.PI;

                // Inner ring segment (between innerRadius and centerRadius)
                double innerFillRatio = CalculateSegmentWhiteRatio(foreground, center,
                    innerRadius, centerRadius, angle1, angle2);
                int innerBit = innerFillRatio >= FILL_THRESHOLD ? 1 : 0;

                // Outer ring segment (between centerRadius and outerRadius)
                double outerFillRatio = CalculateSegmentWhiteRatio(foreground, center,
                    centerRadius, outerRadius, angle1, angle2);
                int outerBit = outerFillRatio >= FILL_THRESHOLD ? 1 : 0;

                if (EnableDetailedLog)
                {
                    Log($"  {i,4} | {midAngleDeg,6:F1} | {innerFillRatio,5:P0} |   {innerBit}   | {outerFillRatio,5:P0} |   {outerBit}");
                }

                // Append bits: inner first, then outer (as per HALCON code)
                binaryString.Append(innerBit);
                binaryString.Append(outerBit);
            }

            return binaryString.ToString();
        }

        /// <summary>
        /// Calculate the ratio of WHITE pixels in a segment (OPTIMIZED)
        /// Uses fixed 4x4=16 samples per segment for speed
        /// </summary>
        private double CalculateSegmentWhiteRatio(Image<Gray, byte> foreground, PointF center,
            float innerRadius, float outerRadius, double startAngle, double endAngle)
        {
            int whitePixels = 0;
            int cx = (int)center.X;
            int cy = (int)center.Y;
            int w = foreground.Width;
            int h = foreground.Height;

            // Use fixed 4 radial x 4 angular = 16 samples (fast and accurate enough)
            for (int ri = 0; ri < 4; ri++)
            {
                float radius = innerRadius + (outerRadius - innerRadius) * (ri + 0.5f) / 4;
                for (int ai = 0; ai < 4; ai++)
                {
                    double angle = startAngle + (endAngle - startAngle) * (ai + 0.5) / 4;
                    int x = (int)(cx + radius * Math.Cos(angle));
                    int y = (int)(cy + radius * Math.Sin(angle));

                    if (x >= 0 && x < w && y >= 0 && y < h && foreground.Data[y, x, 0] > 128)
                        whitePixels++;
                }
            }

            return whitePixels / 16.0;
        }

        /// <summary>
        /// Decrypt binary string to readable data
        /// </summary>
        public string DecryptBinaryToLong(string temp)
        {
            if (string.IsNullOrEmpty(temp) || temp.Length != 48)
            {
                return "-1";
            }

            if (temp == "000000000000000000000000000000000000000000000000")
            {
                return "0";
            }

            try
            {
                // Data structure (48 bits):
                // Bits 0-3: Header (4 bits)
                // Bits 4-11: DueDate (8 bits)
                // Bits 12-23: MachineID (12 bits)
                // Bits 24-45: SerialNumber (22 bits)
                // Bits 46-47: Parity or Bits 44-47: BCC

                string data = temp.Substring(4, temp.Length - 6);
                string checkParityNum = temp.Substring(46, 2);

                // Check parity
                if (checkParityNum == "00" || checkParityNum == "11" || !CheckParity(data, checkParityNum))
                {
                    if (!ValidateBCC(temp))
                    {
                        return "-1";
                    }
                }

                // Extract fields
                string dueDateBits = temp.Substring(4, 8);
                string machineIdBits = temp.Substring(12, 12);
                string serialNumberBits = temp.Substring(24, 22);

                int dueDateValue = Convert.ToInt32(dueDateBits, 2);
                string dueDate = DecodeDueDate(dueDateValue);
                string machineId = Convert.ToInt64(machineIdBits, 2).ToString().PadLeft(4, '0');
                string serialNumber = Convert.ToInt64(serialNumberBits, 2).ToString().PadLeft(7, '0');

                return $"{dueDate}{machineId}{serialNumber}";
            }
            catch
            {
                return "-1";
            }
        }

        private bool CheckParity(string data, string checkParityNum)
        {
            int count = data.Count(bit => bit == '1');
            string expected = (count % 2 == 0) ? "10" : "01";
            return expected == checkParityNum;
        }

        private bool ValidateBCC(string binaryString)
        {
            if (binaryString.Length < 48)
                return false;

            int bcc = 0;
            for (int i = 4; i < 44; i += 4)
            {
                if (i + 4 <= binaryString.Length)
                {
                    int val = Convert.ToInt32(binaryString.Substring(i, 4), 2);
                    bcc ^= val;
                }
            }

            int storedBcc = Convert.ToInt32(binaryString.Substring(44, 4), 2);
            return bcc == storedBcc;
        }

        private string DecodeDueDate(int decimalSeq)
        {
            int baseYear = 2024;
            int year = baseYear + (decimalSeq / 12);
            int month = (decimalSeq % 12) + 1;
            return $"{year}{month:D2}";
        }

        /// <summary>
        /// Create detailed visualization image showing decoded ring with foreground contours and sample points
        /// </summary>
        public Image<Bgr, byte> CreateVisualization(Image<Bgr, byte> source, RingCodeResult result)
        {
            var visualization = source.Clone();

            // Draw outer circle (red)
            CvInvoke.Circle(visualization, Point.Round(result.Center), (int)result.OuterRadius,
                new MCvScalar(0, 0, 255), 2);

            // Draw center circle (red)
            CvInvoke.Circle(visualization, Point.Round(result.Center), (int)result.MiddleRadius,
                new MCvScalar(0, 0, 255), 1);

            // Draw inner circle (red)
            CvInvoke.Circle(visualization, Point.Round(result.Center), (int)(result.OuterRadius * INNER_RATIO),
                new MCvScalar(0, 0, 255), 2);

            // Draw center point (green)
            CvInvoke.Circle(visualization, Point.Round(result.Center), 4,
                new MCvScalar(0, 255, 0), -1);

            // Draw segment lines (green)
            double startAngle = result.RotationAngle * Math.PI / 180;
            double segmentRad = SEGMENT_ANGLE * Math.PI / 180;

            for (int i = 0; i < SEGMENTS; i++)
            {
                double angle = startAngle + (i * segmentRad) - segmentRad;

                // Line from center to outer edge
                int x1 = (int)(result.Center.X + result.OuterRadius * INNER_RATIO * 0.8 * Math.Cos(angle));
                int y1 = (int)(result.Center.Y + result.OuterRadius * INNER_RATIO * 0.8 * Math.Sin(angle));
                int x2 = (int)(result.Center.X + result.OuterRadius * Math.Cos(angle));
                int y2 = (int)(result.Center.Y + result.OuterRadius * Math.Sin(angle));

                CvInvoke.Line(visualization, new Point(x1, y1), new Point(x2, y2),
                    new MCvScalar(0, 255, 0), 1);
            }

            // Draw sample points (green X marks) ONLY on WHITE (filled) segments
            float innerR = result.OuterRadius * (float)INNER_RATIO;
            float centerR = result.MiddleRadius;
            float outerR = result.OuterRadius;

            for (int i = 0; i < SEGMENTS; i++)
            {
                double midAngle = startAngle + (i * segmentRad) - segmentRad + segmentRad / 2;

                // Binary string format: [inner0][outer0][inner1][outer1]...
                bool hasInnerBit = result.BinaryString.Length > i * 2 && result.BinaryString[i * 2] == '1';
                bool hasOuterBit = result.BinaryString.Length > i * 2 + 1 && result.BinaryString[i * 2 + 1] == '1';

                // Inner ring sample point - only if bit is 1
                if (hasInnerBit)
                {
                    float rInner = (innerR + centerR) / 2;
                    int xi = (int)(result.Center.X + rInner * Math.Cos(midAngle));
                    int yi = (int)(result.Center.Y + rInner * Math.Sin(midAngle));
                    DrawCross(visualization, xi, yi, 3, new MCvScalar(0, 255, 0));
                }

                // Outer ring sample point - only if bit is 1
                if (hasOuterBit)
                {
                    float rOuter = (centerR + outerR) / 2;
                    int xo = (int)(result.Center.X + rOuter * Math.Cos(midAngle));
                    int yo = (int)(result.Center.Y + rOuter * Math.Sin(midAngle));
                    DrawCross(visualization, xo, yo, 3, new MCvScalar(0, 255, 0));
                }
            }

            // Draw rotation indicator (green arrow)
            int rx = (int)(result.Center.X + result.OuterRadius * 0.95 * Math.Cos(startAngle));
            int ry = (int)(result.Center.Y + result.OuterRadius * 0.95 * Math.Sin(startAngle));
            CvInvoke.ArrowedLine(visualization, Point.Round(result.Center), new Point(rx, ry),
                new MCvScalar(0, 255, 0), 2);

            return visualization;
        }

        /// <summary>
        /// Draw a small cross mark
        /// </summary>
        private void DrawCross(Image<Bgr, byte> img, int x, int y, int size, MCvScalar color)
        {
            CvInvoke.Line(img, new Point(x - size, y - size), new Point(x + size, y + size), color, 1);
            CvInvoke.Line(img, new Point(x - size, y + size), new Point(x + size, y - size), color, 1);
        }

        /// <summary>
        /// Create combined visualization for all decoded rings with segment lines
        /// </summary>
        public Image<Bgr, byte> CreateCombinedVisualization(Image<Bgr, byte> source, List<RingCodeResult> results)
        {
            var visualization = source.Clone();

            foreach (var result in results)
            {
                var mainColor = result.IsValid ? new MCvScalar(0, 255, 0) : new MCvScalar(0, 0, 255);
                var lineColor = new MCvScalar(0, 255, 0); // Green for segment lines

                // Draw white region contours (RED) for debugging foreground extraction
                if (result.ForegroundMask != null)
                {
                    using var contours = new VectorOfVectorOfPoint();
                    using var hierarchy = new Mat();
                    CvInvoke.FindContours(result.ForegroundMask.Clone(), contours, hierarchy,
                        RetrType.External, ChainApproxMethod.ChainApproxSimple);

                    // Draw all white region contours in RED
                    CvInvoke.DrawContours(visualization, contours, -1, new MCvScalar(0, 0, 255), 1);
                }

                // Draw outer circle
                CvInvoke.Circle(visualization, Point.Round(result.Center), (int)result.OuterRadius, mainColor, 2);

                // Draw middle circle
                CvInvoke.Circle(visualization, Point.Round(result.Center), (int)result.MiddleRadius, mainColor, 1);

                // Draw inner circle
                CvInvoke.Circle(visualization, Point.Round(result.Center), (int)(result.OuterRadius * INNER_RATIO), mainColor, 2);

                // Draw center point
                CvInvoke.Circle(visualization, Point.Round(result.Center), 4, new MCvScalar(0, 0, 255), -1);

                // Draw segment lines (green)
                double startAngle = result.RotationAngle * Math.PI / 180;
                double segmentRad = SEGMENT_ANGLE * Math.PI / 180;

                for (int i = 0; i < SEGMENTS; i++)
                {
                    double angle = startAngle + (i * segmentRad) - segmentRad;

                    // Line from inner to outer edge
                    int x1 = (int)(result.Center.X + result.OuterRadius * INNER_RATIO * 0.9 * Math.Cos(angle));
                    int y1 = (int)(result.Center.Y + result.OuterRadius * INNER_RATIO * 0.9 * Math.Sin(angle));
                    int x2 = (int)(result.Center.X + result.OuterRadius * Math.Cos(angle));
                    int y2 = (int)(result.Center.Y + result.OuterRadius * Math.Sin(angle));

                    CvInvoke.Line(visualization, new Point(x1, y1), new Point(x2, y2), lineColor, 1);
                }

                // Draw sample points (green X marks) ONLY on WHITE (filled) segments
                float innerR = result.OuterRadius * (float)INNER_RATIO;
                float centerR = result.MiddleRadius;
                float outerR = result.OuterRadius;

                for (int i = 0; i < SEGMENTS; i++)
                {
                    double midAngle = startAngle + (i * segmentRad) - segmentRad + segmentRad / 2;

                    // Binary string format: [inner0][outer0][inner1][outer1]...
                    // So segment i has: inner bit at i*2, outer bit at i*2+1
                    bool hasInnerBit = result.BinaryString.Length > i * 2 && result.BinaryString[i * 2] == '1';
                    bool hasOuterBit = result.BinaryString.Length > i * 2 + 1 && result.BinaryString[i * 2 + 1] == '1';

                    // Inner ring sample point - only if bit is 1
                    if (hasInnerBit)
                    {
                        float rInner = (innerR + centerR) / 2;
                        int xi = (int)(result.Center.X + rInner * Math.Cos(midAngle));
                        int yi = (int)(result.Center.Y + rInner * Math.Sin(midAngle));
                        DrawCross(visualization, xi, yi, 3, lineColor);
                    }

                    // Outer ring sample point - only if bit is 1
                    if (hasOuterBit)
                    {
                        float rOuter = (centerR + outerR) / 2;
                        int xo = (int)(result.Center.X + rOuter * Math.Cos(midAngle));
                        int yo = (int)(result.Center.Y + rOuter * Math.Sin(midAngle));
                        DrawCross(visualization, xo, yo, 3, lineColor);
                    }
                }

                // Draw arrow indicator to locator point
                if (result.LocatorPoints.Count > 0)
                {
                    var arrowTip = result.LocatorPoints[0];
                    CvInvoke.ArrowedLine(visualization, Point.Round(result.Center), Point.Round(arrowTip),
                        new MCvScalar(0, 255, 0), 2);
                }

                // Draw label
                string label = result.IsValid
                    ? $"#{result.RingIndex}: {result.DecodedData}"
                    : $"#{result.RingIndex}: Err";
                CvInvoke.PutText(visualization, label,
                    new Point((int)result.Center.X - 60, (int)result.Center.Y - (int)result.OuterRadius - 10),
                    FontFace.HersheySimplex, 0.4, mainColor, 1);
            }

            return visualization;
        }
    }
}
