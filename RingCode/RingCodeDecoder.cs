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

        // Arrow template matchers
        private ArrowTemplateMatcher _darkTemplateMatcher;
        private ArrowTemplateMatcher _lightTemplateMatcher;
        private bool _templatesInitialized = false;

        /// <summary>
        /// Initialize arrow template matchers from default paths
        /// </summary>
        public void InitializeTemplates()
        {
            if (_templatesInitialized) return;

            var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var darkPath = Path.Combine(appData, "arrow_template_dark.png");
            var lightPath = Path.Combine(appData, "arrow_template_light.png");

            _darkTemplateMatcher = new ArrowTemplateMatcher();
            _lightTemplateMatcher = new ArrowTemplateMatcher();

            if (File.Exists(darkPath))
            {
                _darkTemplateMatcher.LoadTemplate(darkPath);
                Log($"Dark template loaded from {darkPath}");
            }

            if (File.Exists(lightPath))
            {
                _lightTemplateMatcher.LoadTemplate(lightPath);
                Log($"Light template loaded from {lightPath}");
            }

            _templatesInitialized = true;
        }

        /// <summary>
        /// Check if templates are loaded
        /// </summary>
        public bool HasDarkTemplate => _darkTemplateMatcher?.IsLoaded ?? false;
        public bool HasLightTemplate => _lightTemplateMatcher?.IsLoaded ?? false;
        public bool HasAnyTemplate => HasDarkTemplate || HasLightTemplate;

        /// <summary>
        /// Last template match error message
        /// </summary>
        public string LastTemplateMatchError { get; private set; } = "";

        /// <summary>
        /// Last successful template match result (for visualization)
        /// </summary>
        public ArrowMatchResult LastTemplateMatchResult { get; private set; }

        // Private field to track last match type
        private string _lastMatchType = "";

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
            public string TemplateMatchError { get; set; } = "";  // Template matching specific error
            public Image<Bgr, byte> ProcessedImage { get; set; }
            public List<PointF> LocatorPoints { get; set; } = new();
            public int RingIndex { get; set; }
            public Image<Gray, byte> ForegroundMask { get; set; }  // For debugging

            // Template match result for visualization
            public PointF? TemplateMatchCenter { get; set; }
            public VectorOfPoint TemplateMatchContour { get; set; }
            public double TemplateMatchScore { get; set; }
            public string TemplateMatchType { get; set; } = "";  // "Dark" or "Light"

            // Arrow contour for special marking
            public VectorOfPoint ArrowContour { get; set; }
            public PointF ArrowTip { get; set; }
            public string ArrowDetectionMethod { get; set; } = "";  // "Template", "Contour", "ContourInv", "Intensity"
        }

        /// <summary>
        /// Decode a single ring region with multi-rotation search
        /// Tries multiple rotation angles and picks the one that passes BCC validation
        /// </summary>
        public RingCodeResult DecodeRing(Image<Gray, byte> grayImage, RingImageSegmentation.RingRegion region)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new RingCodeResult
            {
                RingIndex = region.Index,
                Center = region.Center,
                OuterRadius = region.OuterRadius,
                InnerRadius = region.InnerRadius,
                LocatorPoints = region.TrianglePoints
            };

            if (EnableDetailedLog)
            {
                Log($"========== Ring #{region.Index} ==========");
                Log($"Center: ({region.Center.X:F1}, {region.Center.Y:F1}), OuterR: {region.OuterRadius:F1}");
            }

            try
            {
                // Step 1: Extract ring region
                var ringImage = ExtractRingRegion(grayImage, region);
                long t1 = sw.ElapsedMilliseconds;

                // Step 2: Extract white foreground region
                var foregroundMask = ExtractForegroundRegion(ringImage, region);
                result.ForegroundMask = foregroundMask;
                long t2 = sw.ElapsedMilliseconds;

                if (EnableDetailedLog)
                {
                    // Count white pixels for debugging
                    int whiteCount = CvInvoke.CountNonZero(foregroundMask);
                    Log($"Ring#{region.Index}: Extract={t1}ms, Foreground={t2 - t1}ms, whitePixels={whiteCount}");
                }

                // Step 3: Find arrow/triangle and calculate base rotation angle
                double baseAngle = FindArrowAngle(foregroundMask, ringImage, region);
                result.TemplateMatchError = LastTemplateMatchError;  // Capture any template match error

                // Capture template match result for visualization
                if (LastTemplateMatchResult != null && LastTemplateMatchResult.IsFound)
                {
                    result.TemplateMatchCenter = LastTemplateMatchResult.Center;
                    result.TemplateMatchContour = LastTemplateMatchResult.MatchedContour;
                    result.TemplateMatchScore = LastTemplateMatchResult.Score;
                    result.TemplateMatchType = _lastMatchType;
                }

                // Capture arrow contour for visualization
                if (_lastArrowContour != null)
                {
                    result.ArrowContour = _lastArrowContour;
                    result.ArrowTip = _lastArrowTip;
                }
                long t3 = sw.ElapsedMilliseconds;

                // Step 4 & 5: Fast decode with fallback
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
                    double[] angleOffsets = { 7.5, -7.5, 15, -15 };
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
                long t4 = sw.ElapsedMilliseconds;

                result.MiddleRadius = centerRadius;
                result.RotationAngle = bestAngle;
                result.BinaryString = bestBinary;
                result.DecodedData = bestDecoded;
                result.IsValid = found;

                Log($"Ring#{region.Index}: Arrow={t3 - t2}ms, Decode={t4 - t3}ms, Total={t4}ms, Valid={found}, Data={bestDecoded}");

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
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var results = new List<RingCodeResult>();

            var segmentation = new RingImageSegmentation();
            var segResult = segmentation.SegmentImage(sourceImage);
            Log($"DecodeAllRings: Segmentation took {sw.ElapsedMilliseconds}ms");

            if (!segResult.Success)
            {
                return results;
            }

            var grayImage = sourceImage.Convert<Gray, byte>();
            Log($"DecodeAllRings: Gray conversion took {sw.ElapsedMilliseconds}ms");

            long decodeStart = sw.ElapsedMilliseconds;
            foreach (var ring in segResult.DetectedRings)
            {
                var decoded = DecodeRing(grayImage, ring);
                // Skip visualization per ring - will create combined viz later
                results.Add(decoded);
            }
            Log($"DecodeAllRings: Decoded {results.Count} rings in {sw.ElapsedMilliseconds - decodeStart}ms");
            Log($"DecodeAllRings: Total time {sw.ElapsedMilliseconds}ms");

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
        /// Extract foreground region using Otsu + AdaptiveThreshold
        /// Improved: Uses Otsu threshold for light/dark determination and validates result
        /// </summary>
        private Image<Gray, byte> ExtractForegroundRegion(Image<Gray, byte> ringImage, RingImageSegmentation.RingRegion region)
        {
            int innerR = (int)(region.OuterRadius * 0.45);
            int outerR = (int)(region.OuterRadius * 1.02);
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;

            // Create ring mask for the data area (exclude center hole)
            var ringMask = new Image<Gray, byte>(ringImage.Size);
            CvInvoke.Circle(ringMask, new Point(cx, cy), outerR, new MCvScalar(255), -1);
            CvInvoke.Circle(ringMask, new Point(cx, cy), innerR, new MCvScalar(0), -1);

            // Step 1: Apply Gaussian blur for noise reduction
            var smoothed = new Image<Gray, byte>(ringImage.Size);
            CvInvoke.GaussianBlur(ringImage, smoothed, new System.Drawing.Size(7, 7), 0);

            // Step 2: Sample intensities from DATA RING area only (0.5R - 0.85R)
            // This excludes the outer edge and center hole for accurate analysis
            var dataIntensities = new List<int>();
            float[] dataRadii = { 0.55f, 0.65f, 0.75f, 0.82f };
            int samplesPerRadius = 48;

            foreach (float ratioR in dataRadii)
            {
                float r = region.OuterRadius * ratioR;
                for (int i = 0; i < samplesPerRadius; i++)
                {
                    double angle = i * (2 * Math.PI / samplesPerRadius);
                    int x = (int)(cx + r * Math.Cos(angle));
                    int y = (int)(cy + r * Math.Sin(angle));

                    if (x >= 0 && x < smoothed.Width && y >= 0 && y < smoothed.Height)
                    {
                        dataIntensities.Add(smoothed.Data[y, x, 0]);
                    }
                }
            }

            if (dataIntensities.Count == 0)
            {
                Log($"  ERROR: No data samples collected!");
                return new Image<Gray, byte>(ringImage.Size);
            }

            // Step 3: Calculate Otsu threshold on sampled data
            int[] histogram = new int[256];
            foreach (int v in dataIntensities)
                histogram[v]++;

            double otsuThreshold = CalculateOtsuThreshold(histogram, dataIntensities.Count);

            // Count pixels above and below Otsu threshold
            int darkCount = 0, lightCount = 0;
            double darkSum = 0, lightSum = 0;
            foreach (int v in dataIntensities)
            {
                if (v < otsuThreshold)
                {
                    darkCount++;
                    darkSum += v;
                }
                else
                {
                    lightCount++;
                    lightSum += v;
                }
            }

            double darkMean = darkCount > 0 ? darkSum / darkCount : 0;
            double lightMean = lightCount > 0 ? lightSum / lightCount : 255;
            double ringMean = dataIntensities.Average();

            // Step 4: Determine light/dark ring
            // Light ring: background (majority) is bright, marks are dark
            // Dark ring: background (majority) is dark, marks are bright
            bool isLightRing;
            string reason;

            // The background should be the majority of pixels
            // Data marks occupy roughly 30-40% of the ring area
            if (lightCount > darkCount * 1.2)
            {
                isLightRing = true;
                reason = $"lightCount({lightCount}) > darkCount({darkCount})*1.2";
            }
            else if (darkCount > lightCount * 1.2)
            {
                isLightRing = false;
                reason = $"darkCount({darkCount}) > lightCount({lightCount})*1.2";
            }
            else
            {
                // Close counts - use mean intensity
                isLightRing = ringMean > 128;
                reason = $"close counts, mean={ringMean:F0}";
            }

            _lastRingIsLight = isLightRing;

            Log($"  Ring type: {(isLightRing ? "LIGHT" : "DARK")} ({reason})");
            Log($"    Otsu={otsuThreshold:F0}, darkMean={darkMean:F0}, lightMean={lightMean:F0}, " +
                $"darkCount={darkCount}, lightCount={lightCount}");

            // Step 5: Create binary image using GLOBAL Otsu threshold first
            var binaryOtsu = new Image<Gray, byte>(smoothed.Size);
            CvInvoke.Threshold(smoothed, binaryOtsu, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            // Step 6: Also try Adaptive threshold for comparison
            int blockSize = Math.Max(31, (int)(region.OuterRadius / 4) | 1);
            blockSize = Math.Min(blockSize, 101);
            double C = 8;

            var binaryAdaptive = new Image<Gray, byte>(smoothed.Size);
            CvInvoke.AdaptiveThreshold(smoothed, binaryAdaptive, 255,
                AdaptiveThresholdType.GaussianC, ThresholdType.Binary, blockSize, C);

            // Step 7: Choose best binary result based on ring type
            Image<Gray, byte> binary;
            string binaryMethod;

            if (isLightRing)
            {
                // Light ring: we want dark marks to become WHITE in output
                // Otsu Binary gives: bright->white, dark->black
                // So we need to INVERT: dark marks become white
                var binaryOtsuInv = binaryOtsu.Not();

                // Count white pixels in data area for validation
                var maskedOtsu = binaryOtsuInv.Copy(ringMask);
                var maskedAdaptiveInv = binaryAdaptive.Not().Copy(ringMask);

                int whiteOtsu = CvInvoke.CountNonZero(maskedOtsu);
                int whiteAdaptive = CvInvoke.CountNonZero(maskedAdaptiveInv);

                // Expected: ~30-50% of ring area should be white (data marks)
                int ringArea = (int)(Math.PI * (outerR * outerR - innerR * innerR));
                double ratioOtsu = (double)whiteOtsu / ringArea;
                double ratioAdaptive = (double)whiteAdaptive / ringArea;

                Log($"    Light ring binary: Otsu={whiteOtsu}({ratioOtsu:P0}), Adaptive={whiteAdaptive}({ratioAdaptive:P0})");

                // Choose the one closer to expected ratio (30-50%)
                double targetRatio = 0.40;
                if (Math.Abs(ratioOtsu - targetRatio) < Math.Abs(ratioAdaptive - targetRatio))
                {
                    binary = binaryOtsuInv;
                    binaryMethod = "OtsuInv";
                }
                else
                {
                    binary = binaryAdaptive.Not();
                    binaryMethod = "AdaptiveInv";
                }

                maskedOtsu.Dispose();
                maskedAdaptiveInv.Dispose();
            }
            else
            {
                // Dark ring: we want bright marks to become WHITE in output
                // Otsu Binary gives: bright->white, dark->black - already correct!
                var maskedOtsu = binaryOtsu.Copy(ringMask);
                var maskedAdaptive = binaryAdaptive.Copy(ringMask);

                int whiteOtsu = CvInvoke.CountNonZero(maskedOtsu);
                int whiteAdaptive = CvInvoke.CountNonZero(maskedAdaptive);

                int ringArea = (int)(Math.PI * (outerR * outerR - innerR * innerR));
                double ratioOtsu = (double)whiteOtsu / ringArea;
                double ratioAdaptive = (double)whiteAdaptive / ringArea;

                Log($"    Dark ring binary: Otsu={whiteOtsu}({ratioOtsu:P0}), Adaptive={whiteAdaptive}({ratioAdaptive:P0})");

                double targetRatio = 0.40;
                if (Math.Abs(ratioOtsu - targetRatio) < Math.Abs(ratioAdaptive - targetRatio))
                {
                    binary = binaryOtsu;
                    binaryMethod = "Otsu";
                }
                else
                {
                    binary = binaryAdaptive;
                    binaryMethod = "Adaptive";
                }

                maskedOtsu.Dispose();
                maskedAdaptive.Dispose();
            }

            // Step 8: Morphological cleanup
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Open, kernel,
                new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Close, kernel,
                new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            // Apply ring mask
            var foreground = binary.Copy(ringMask);

            int finalWhite = CvInvoke.CountNonZero(foreground);
            Log($"  Foreground: method={binaryMethod}, whitePixels={finalWhite}");

            // Cleanup
            binaryOtsu.Dispose();
            binaryAdaptive.Dispose();
            if (binary != binaryOtsu && binary != binaryAdaptive)
                binary.Dispose();

            return foreground;
        }

        /// <summary>
        /// Calculate Otsu threshold from histogram
        /// </summary>
        private double CalculateOtsuThreshold(int[] histogram, int totalPixels)
        {
            double sum = 0;
            for (int i = 0; i < 256; i++)
                sum += i * histogram[i];

            double sumB = 0;
            int wB = 0;
            double maxVariance = 0;
            double threshold = 128;

            for (int t = 0; t < 256; t++)
            {
                wB += histogram[t];
                if (wB == 0) continue;

                int wF = totalPixels - wB;
                if (wF == 0) break;

                sumB += t * histogram[t];
                double mB = sumB / wB;
                double mF = (sum - sumB) / wF;

                double variance = (double)wB * wF * (mB - mF) * (mB - mF);
                if (variance > maxVariance)
                {
                    maxVariance = variance;
                    threshold = t;
                }
            }

            return threshold;
        }

        /// <summary>
        /// Store ring type for arrow detection
        /// </summary>
        private bool _lastRingIsLight = true;

        /// <summary>
        /// Find arrow angle using BINARY template matching
        /// Simple approach: binarize both image and template, then pattern match
        /// </summary>
        private double FindArrowAngle(Image<Gray, byte> foreground, Image<Gray, byte> original,
            RingImageSegmentation.RingRegion region)
        {
            // Initialize templates if not done yet
            InitializeTemplates();
            _lastMatchType = "";
            _lastArrowContour = null;
            _lastArrowTip = PointF.Empty;

            // Method 1 (PRIMARY): Find arrow by analyzing EDGE contours on ORIGINAL image
            // Arrow is at outer edge, has distinctive triangular/Y shape with LOW solidity
            Log($"  [Method 1] Edge contour analysis on original image...");
            double contourAngle = FindArrowByEdgeContours(original, region);
            if (contourAngle != double.MinValue)
            {
                Log($"  Edge contour analysis found arrow at: {contourAngle:F1}°");
                return contourAngle;
            }

            // Method 2 (FALLBACK): Use triangle TIP points if available (from segmentation)
            // Note: TrianglePoints from segmentation is often unreliable
            if (region.TrianglePoints.Count >= 1)
            {
                var arrowTip = region.TrianglePoints
                    .OrderByDescending(pt =>
                        Math.Sqrt(Math.Pow(pt.X - region.Center.X, 2) + Math.Pow(pt.Y - region.Center.Y, 2)))
                    .First();

                double dist = Math.Sqrt(Math.Pow(arrowTip.X - region.Center.X, 2) + Math.Pow(arrowTip.Y - region.Center.Y, 2));
                double angle = Math.Atan2(arrowTip.Y - region.Center.Y, arrowTip.X - region.Center.X);
                double angleDeg = angle * 180 / Math.PI;

                Log($"  [Method 2 fallback] Arrow TIP at ({arrowTip.X:F1}, {arrowTip.Y:F1}), dist={dist:F1}, angle={angleDeg:F1}°");
                return angleDeg;
            }

            // Method 3: Intensity peak detection as last resort
            Log($"  [Method 3] Using intensity peak detection as last resort...");
            return FindRotationByIntensityPeak(foreground, region);
        }

        /// <summary>
        /// Find arrow using BINARY template matching
        /// Both source and template are binarized, then pattern match with rotation
        /// </summary>
        private (bool found, double angle, double score, PointF center) FindArrowByBinaryTemplateMatch(
            Image<Gray, byte> foreground, Image<Gray, byte> original, RingImageSegmentation.RingRegion region)
        {
            // Create ring ROI for search
            int margin = (int)(region.OuterRadius * 0.15f);
            int roiX = Math.Max(0, (int)(region.Center.X - region.OuterRadius - margin));
            int roiY = Math.Max(0, (int)(region.Center.Y - region.OuterRadius - margin));
            int roiW = Math.Min((int)(region.OuterRadius * 2 + margin * 2), original.Width - roiX);
            int roiH = Math.Min((int)(region.OuterRadius * 2 + margin * 2), original.Height - roiY);

            if (roiW < 50 || roiH < 50)
                return (false, 0, 0, PointF.Empty);

            // Extract and binarize the ring region
            original.ROI = new System.Drawing.Rectangle(roiX, roiY, roiW, roiH);
            var roiImage = original.Clone();
            original.ROI = System.Drawing.Rectangle.Empty;

            // Binarize with Otsu threshold
            var binaryROI = new Image<Gray, byte>(roiImage.Size);
            CvInvoke.Threshold(roiImage, binaryROI, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            // Also try inverted binary (for light arrows on light background)
            var binaryROIInv = new Image<Gray, byte>(roiImage.Size);
            CvInvoke.BitwiseNot(binaryROI, binaryROIInv);

            double bestScore = 0;
            double bestAngle = 0;
            PointF bestCenter = PointF.Empty;
            string bestType = "";

            // Try both dark and light templates
            var templatesToTry = new List<(ArrowTemplateMatcher matcher, string name, Image<Gray, byte> searchImage)>();

            if (_darkTemplateMatcher?.IsLoaded == true)
            {
                templatesToTry.Add((_darkTemplateMatcher, "Dark", binaryROI));
                templatesToTry.Add((_darkTemplateMatcher, "DarkInv", binaryROIInv));
            }
            if (_lightTemplateMatcher?.IsLoaded == true)
            {
                templatesToTry.Add((_lightTemplateMatcher, "Light", binaryROI));
                templatesToTry.Add((_lightTemplateMatcher, "LightInv", binaryROIInv));
            }

            foreach (var (matcher, name, searchImage) in templatesToTry)
            {
                // Get binary template
                var template = GetBinaryTemplate(matcher);
                if (template == null) continue;

                // Calculate expected template scale based on ring size
                double expectedSize = (region.OuterRadius - region.InnerRadius) * 0.5;
                double templateSize = Math.Max(template.Width, template.Height);
                double baseScale = expectedSize / templateSize;

                // Try multiple scales
                double[] scales = { baseScale * 0.6, baseScale * 0.8, baseScale, baseScale * 1.2, baseScale * 1.5 };

                foreach (double scale in scales)
                {
                    if (scale < 0.1 || scale > 3.0) continue;

                    int scaledW = Math.Max(8, (int)(template.Width * scale));
                    int scaledH = Math.Max(8, (int)(template.Height * scale));

                    if (scaledW >= searchImage.Width - 2 || scaledH >= searchImage.Height - 2)
                        continue;

                    var scaledTemplate = new Image<Gray, byte>(scaledW, scaledH);
                    CvInvoke.Resize(template, scaledTemplate, new System.Drawing.Size(scaledW, scaledH));

                    // Try different rotation angles (coarse search: 15° steps)
                    for (double angle = 0; angle < 360; angle += 15)
                    {
                        // Rotate template
                        using var rotMat = new Mat();
                        CvInvoke.GetRotationMatrix2D(new PointF(scaledW / 2f, scaledH / 2f), angle, 1.0, rotMat);

                        var rotatedTemplate = new Image<Gray, byte>(scaledTemplate.Size);
                        CvInvoke.WarpAffine(scaledTemplate, rotatedTemplate, rotMat, scaledTemplate.Size,
                            Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

                        // Template match
                        using var matchResult = new Mat();
                        CvInvoke.MatchTemplate(searchImage, rotatedTemplate, matchResult, TemplateMatchingType.CcoeffNormed);

                        double minVal = 0, maxVal = 0;
                        Point minLoc = new Point(), maxLoc = new Point();
                        CvInvoke.MinMaxLoc(matchResult, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                        if (maxVal > bestScore && maxVal > 0.45)  // Increased threshold for better accuracy
                        {
                            // Verify match is in the ring area
                            float matchX = maxLoc.X + scaledW / 2f + roiX;
                            float matchY = maxLoc.Y + scaledH / 2f + roiY;
                            float distFromCenter = (float)Math.Sqrt(
                                Math.Pow(matchX - region.Center.X, 2) +
                                Math.Pow(matchY - region.Center.Y, 2));

                            // CRITICAL: Arrow must be at OUTER EDGE (not inner ring where code segments are)
                            // Arrow markers are typically at 0.85-1.1 of outer radius
                            if (distFromCenter >= region.OuterRadius * 0.75f && distFromCenter <= region.OuterRadius * 1.15f)
                            {
                                bestScore = maxVal;
                                bestCenter = new PointF(matchX, matchY);
                                bestType = name;

                                // Calculate angle from ring center to match center
                                bestAngle = Math.Atan2(matchY - region.Center.Y, matchX - region.Center.X) * 180 / Math.PI;
                            }
                        }

                        rotatedTemplate.Dispose();
                    }

                    scaledTemplate.Dispose();
                }
            }

            if (bestScore > 0.45)  // Must have good confidence to use template match
            {
                Log($"    Best binary match: {bestType}, score={bestScore:F2}, angle={bestAngle:F1}°");
                _lastMatchType = bestType;
                LastTemplateMatchResult = new ArrowMatchResult
                {
                    IsFound = true,
                    Score = bestScore,
                    Angle = bestAngle,
                    Center = bestCenter
                };
                return (true, bestAngle, bestScore, bestCenter);
            }

            return (false, 0, 0, PointF.Empty);
        }

        /// <summary>
        /// Get binarized version of template
        /// </summary>
        private Image<Gray, byte> GetBinaryTemplate(ArrowTemplateMatcher matcher)
        {
            if (matcher == null || !matcher.IsLoaded || string.IsNullOrEmpty(matcher.TemplatePath))
                return null;

            try
            {
                var template = new Image<Gray, byte>(matcher.TemplatePath);
                var binary = new Image<Gray, byte>(template.Size);
                CvInvoke.Threshold(template, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);
                return binary;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Find arrow using template matching (tries both dark and light templates)
        /// Returns: (angle, error, matchResult, templateType)
        /// </summary>
        private (double? angle, string error, ArrowMatchResult matchResult, string templateType) FindArrowByTemplateMatching(Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            ArrowMatchResult bestResult = null;
            string bestTemplateType = "";
            string errors = "";

            // Try dark template first - use multi-angle matching for better results
            if (_darkTemplateMatcher?.IsLoaded == true)
            {
                // Method 1: Standard template matching
                var darkResult = _darkTemplateMatcher.FindArrowMultiAngle(foreground, minScore: 0.4,
                    angleStart: -180, angleEnd: 180, angleStep: 15);

                // Method 2: Try edge-based matching if standard fails
                if (!darkResult.IsFound)
                {
                    Log($"    Dark template standard failed, trying edge-based...");
                    darkResult = _darkTemplateMatcher.FindArrowEdgeBased(foreground, minScore: 0.3,
                        angleStart: -180, angleEnd: 180, angleStep: 15);
                }

                if (darkResult.IsFound)
                {
                    if (bestResult == null || darkResult.Score > bestResult.Score)
                    {
                        bestResult = darkResult;
                        bestTemplateType = "Dark";
                        Log($"    Dark template matched: score={darkResult.Score:F2}, angle={darkResult.Angle:F1}°");
                    }
                }
                else
                {
                    errors += $"深色範本: {darkResult.ErrorMessage}; ";
                }
            }

            // Try light template - use multi-angle matching
            if (_lightTemplateMatcher?.IsLoaded == true)
            {
                // Method 1: Standard template matching
                var lightResult = _lightTemplateMatcher.FindArrowMultiAngle(foreground, minScore: 0.4,
                    angleStart: -180, angleEnd: 180, angleStep: 15);

                // Method 2: Try edge-based matching if standard fails
                if (!lightResult.IsFound)
                {
                    Log($"    Light template standard failed, trying edge-based...");
                    lightResult = _lightTemplateMatcher.FindArrowEdgeBased(foreground, minScore: 0.3,
                        angleStart: -180, angleEnd: 180, angleStep: 15);
                }

                if (lightResult.IsFound)
                {
                    if (bestResult == null || lightResult.Score > bestResult.Score)
                    {
                        bestResult = lightResult;
                        bestTemplateType = "Light";
                        Log($"    Light template matched: score={lightResult.Score:F2}, angle={lightResult.Angle:F1}°");
                    }
                }
                else
                {
                    errors += $"淺色範本: {lightResult.ErrorMessage}; ";
                }
            }

            if (bestResult != null && bestResult.IsFound)
            {
                LastTemplateMatchResult = bestResult;
                // Calculate angle from center to matched arrow
                double angle = Math.Atan2(bestResult.Center.Y - region.Center.Y, bestResult.Center.X - region.Center.X);
                return (angle * 180 / Math.PI, null, bestResult, bestTemplateType);
            }

            LastTemplateMatchResult = null;
            return (null, errors.TrimEnd(' ', ';'), null, "");
        }

        // Store last detected arrow contour for visualization
        private VectorOfPoint _lastArrowContour;
        private PointF _lastArrowTip;

        /// <summary>
        /// Arrow detection result with score for comparison
        /// </summary>
        private (double angle, double score, VectorOfPoint contour, PointF tipPoint) FindArrowByContourAnalysisWithScore(
            Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            // Find contours in the foreground
            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(foreground.Clone(), contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            // Calculate expected segment area
            double ringArea = Math.PI * (region.OuterRadius * region.OuterRadius - region.InnerRadius * region.InnerRadius);
            double expectedSegmentArea = ringArea / 48.0;
            double minArea = expectedSegmentArea * 0.02;   // Very small for arrow
            double maxArea = expectedSegmentArea * 2.0;

            // Collect valid candidates
            var candidates = new List<(VectorOfPoint contour, double area, double solidity, int vertices,
                PointF centroid, PointF tipPoint, double tipAngle, double tipDist)>();

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);

                if (area < minArea || area > maxArea) continue;

                var moments = CvInvoke.Moments(contour);
                if (moments.M00 <= 0) continue;

                float cx = (float)(moments.M10 / moments.M00);
                float cy = (float)(moments.M01 / moments.M00);

                // Arrow must be INSIDE the data ring (0.50R - 0.98R)
                double centroidDist = Math.Sqrt(Math.Pow(cx - region.Center.X, 2) + Math.Pow(cy - region.Center.Y, 2));
                double centroidDistRatio = centroidDist / region.OuterRadius;
                if (centroidDistRatio < ARROW_MIN_DIST_RATIO || centroidDistRatio > ARROW_MAX_DIST_RATIO)
                    continue;

                // Calculate solidity
                using var hull = new VectorOfPoint();
                CvInvoke.ConvexHull(contour, hull);
                double hullArea = CvInvoke.ContourArea(hull);
                double solidity = hullArea > 0 ? area / hullArea : 1.0;

                // Approximate polygon vertices
                using var approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, approx, CvInvoke.ArcLength(contour, true) * 0.02, true);
                int vertices = approx.Size;

                // Find the TIP - point furthest from ring center
                var points = contour.ToArray();
                PointF tipPoint = new PointF(cx, cy);
                double maxDist = 0;

                foreach (var pt in points)
                {
                    double dist = Math.Sqrt(Math.Pow(pt.X - region.Center.X, 2) + Math.Pow(pt.Y - region.Center.Y, 2));
                    if (dist > maxDist)
                    {
                        maxDist = dist;
                        tipPoint = new PointF(pt.X, pt.Y);
                    }
                }

                double tipAngle = Math.Atan2(tipPoint.Y - region.Center.Y, tipPoint.X - region.Center.X) * 180 / Math.PI;

                var contourCopy = new VectorOfPoint(contour.ToArray());
                candidates.Add((contourCopy, area, solidity, vertices, new PointF(cx, cy), tipPoint, tipAngle, maxDist));
            }

            if (candidates.Count == 0)
                return (0, 0, null, PointF.Empty);

            // Score each candidate - PRIORITIZE shapes near OUTER EDGE with TRIANGULAR characteristics
            double bestScore = 0;
            (VectorOfPoint contour, double area, double solidity, int vertices,
                PointF centroid, PointF tipPoint, double tipAngle, double tipDist) bestArrow = default;

            foreach (var cand in candidates)
            {
                // Key insight: Arrow tip should be at OUTER part of data ring
                double tipDistRatio = cand.tipDist / region.OuterRadius;

                // Arrow tip should be in outer part of data ring (0.70R - 0.98R)
                double edgeScore = 0;
                if (tipDistRatio >= ARROW_TIP_MIN_RATIO && tipDistRatio <= ARROW_TIP_MAX_RATIO)
                    edgeScore = 1.0;  // Perfect - at outer edge of data ring
                else if (tipDistRatio >= 0.65 && tipDistRatio <= 1.02)
                    edgeScore = 0.6;  // Acceptable
                else
                    edgeScore = 0.2;  // Not in expected position

                // Smaller area = more likely to be arrow (arrow is smallest segment)
                double areaScore = 1.0 / (1.0 + cand.area / expectedSegmentArea);

                // Lower solidity = more triangular (rectangles have ~0.9, triangles have ~0.5-0.7)
                double solidityScore = (cand.solidity < ARROW_SOLIDITY_THRESHOLD) ? (0.85 - cand.solidity) : 0.05;

                // Vertices bonus for triangle-like shapes (3-5 vertices)
                double vertexScore = (cand.vertices >= 3 && cand.vertices <= 5) ? 1.0 :
                                     (cand.vertices >= 6 && cand.vertices <= 7) ? 0.7 : 0.3;

                double score = edgeScore * areaScore * solidityScore * vertexScore * 100;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestArrow = cand;
                }
            }

            if (bestArrow.contour != null)
            {
                return (bestArrow.tipAngle, bestScore, bestArrow.contour, bestArrow.tipPoint);
            }

            return (0, 0, null, PointF.Empty);
        }

        /// <summary>
        /// Find arrow angle using HALCON-style approach (legacy method - calls the scored version)
        /// </summary>
        private double FindArrowByContourAnalysis(Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            _lastArrowContour = null;
            _lastArrowTip = PointF.Empty;

            var result = FindArrowByContourAnalysisWithScore(foreground, region);

            if (result.score > 0 && result.contour != null)
            {
                _lastArrowContour = result.contour;
                _lastArrowTip = result.tipPoint;
                Log($"  Arrow found: angle={result.angle:F1}°, score={result.score:F3}");
                return result.angle;
            }

            Log($"  No arrow found, using intensity peak detection...");
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

        // Arrow detection constants - centralized threshold definitions
        // CRITICAL: Arrow is INSIDE the data ring (the white donut area), not at the outer edge!
        // outerR from FindRing is the OUTER EDGE of the white ring, arrow is inside at ~0.7-0.95R
        private const double ARROW_SOLIDITY_THRESHOLD = 0.75;  // Arrow Y-shape typically 0.4-0.65, data marks 0.85+
        private const double ARROW_SOLIDITY_CONFIDENT = 0.60;  // Very confident arrow detection threshold
        private const double ARROW_MIN_DIST_RATIO = 0.50;      // Inner edge of data ring area (after center hole)
        private const double ARROW_MAX_DIST_RATIO = 0.98;      // Outer edge of data ring (before white background)
        private const double ARROW_TIP_MIN_RATIO = 0.70;       // Arrow tip should be in outer part of data ring
        private const double ARROW_TIP_MAX_RATIO = 0.98;       // Arrow tip near outer edge of data

        /// <summary>
        /// Find arrow by analyzing BLOB contours on THRESHOLDED image
        /// CRITICAL: Arrow is INSIDE the data ring (0.50R-0.98R), with tip pointing OUTWARD
        /// Arrow tip should be at outer part of data ring (0.70R-0.98R)
        /// </summary>
        private double FindArrowByEdgeContours(Image<Gray, byte> original, RingImageSegmentation.RingRegion region)
        {
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;
            float outerR = region.OuterRadius;
            float innerR = region.InnerRadius;

            // === RING-BASED PREPROCESSING ===
            // Create mask for DATA RING area (where arrow is located)
            var dataMask = new Image<Gray, byte>(original.Size);
            CvInvoke.Circle(dataMask, new Point(cx, cy), (int)outerR, new MCvScalar(255), -1);
            CvInvoke.Circle(dataMask, new Point(cx, cy), (int)innerR, new MCvScalar(0), -1);

            // Apply CLAHE for uneven lighting correction
            var enhanced = new Mat();
            CvInvoke.CLAHE(original, 2.0, new System.Drawing.Size(8, 8), enhanced);
            var enhancedImg = enhanced.ToImage<Gray, byte>();

            // Collect pixel values from ring area only for statistical thresholding
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

            // Calculate optimal threshold from ring area statistics
            double meanValue = ringPixels.Count > 0 ? ringPixels.Average(p => (double)p) : 128;
            double stdDev = ringPixels.Count > 0 ? Math.Sqrt(ringPixels.Average(p => Math.Pow(p - meanValue, 2))) : 30;
            double calculatedThresh = meanValue - 1.5 * stdDev;
            byte optimalThresh = (byte)Math.Max(80, Math.Min(180, calculatedThresh));

            Log($"    Ring-based threshold: {optimalThresh} (mean={meanValue:F0}, std={stdDev:F0})");

            // Apply binary threshold to find dark marks
            var binaryResult = new Image<Gray, byte>(enhancedImg.Size);
            CvInvoke.Threshold(enhancedImg, binaryResult, optimalThresh, 255, ThresholdType.BinaryInv);

            // Apply mask
            var maskedBinary = new Image<Gray, byte>(original.Size);
            CvInvoke.BitwiseAnd(binaryResult, dataMask, maskedBinary);

            // Morphological cleanup
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(maskedBinary, maskedBinary, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
            var largerKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new System.Drawing.Size(5, 5), new Point(-1, -1));
            CvInvoke.MorphologyEx(maskedBinary, maskedBinary, MorphOp.Close, largerKernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));

            // Collect candidates from preprocessed binary
            var allCandidates = new List<(double score, double solidity, double angle, double area,
                double centroidDistRatio, double tipDistRatio, double elongation, VectorOfPoint contour,
                PointF centroid, PointF basePoint, bool tipPointsOutward, string method)>();

            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(maskedBinary.Clone(), contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            // Arrow characteristics - reasonable area based on ring size
            double minArea = outerR * outerR * 0.005;  // ~0.5% of ring area
            double maxArea = outerR * outerR * 0.10;   // ~10% of ring area

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);
                if (area < minArea || area > maxArea) continue;

                var moments = CvInvoke.Moments(contour);
                if (moments.M00 < 1) continue;

                float ctrX = (float)(moments.M10 / moments.M00);
                float ctrY = (float)(moments.M01 / moments.M00);
                PointF centroid = new PointF(ctrX, ctrY);

                double centroidDist = Math.Sqrt(Math.Pow(ctrX - cx, 2) + Math.Pow(ctrY - cy, 2));
                double centroidDistRatio = centroidDist / outerR;

                // Find TIP point (furthest from center) and BASE point (closest to center)
                var points = contour.ToArray();
                PointF tipPoint = centroid;
                PointF basePoint = centroid;
                double maxTipDist = 0;
                double minBaseDist = double.MaxValue;

                foreach (var pt in points)
                {
                    double d = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                    if (d > maxTipDist)
                    {
                        maxTipDist = d;
                        tipPoint = new PointF(pt.X, pt.Y);
                    }
                    if (d < minBaseDist)
                    {
                        minBaseDist = d;
                        basePoint = new PointF(pt.X, pt.Y);
                    }
                }

                double tipDistRatio = maxTipDist / outerR;

                // Arrow tip should point OUTWARD
                bool tipPointsOutward = maxTipDist > minBaseDist && (maxTipDist - minBaseDist) > 3;

                // Calculate solidity - arrow (Y/triangle) has LOW solidity
                using var hull = new VectorOfPoint();
                CvInvoke.ConvexHull(contour, hull);
                double hullArea = CvInvoke.ContourArea(hull);
                double solidity = hullArea > 0 ? area / hullArea : 1.0;

                // Calculate elongation ratio (tip-base distance vs sqrt(area))
                double elongation = (maxTipDist - minBaseDist) / Math.Sqrt(area);

                // Calculate angle from center to BASE point (closest point) - this is the arrow direction
                double angle = Math.Atan2(basePoint.Y - cy, basePoint.X - cx) * 180.0 / Math.PI;
                if (angle < 0) angle += 360;

                // Multi-feature scoring with emphasis on solidity and elongation
                double score = CalculateArrowScore(solidity, centroidDistRatio, tipDistRatio,
                    area, outerR, tipPointsOutward, elongation);

                if (score > 0.1)
                {
                    var contourCopy = new VectorOfPoint(contour.ToArray());
                    allCandidates.Add((score, solidity, angle, area, centroidDistRatio, tipDistRatio,
                        elongation, contourCopy, centroid, basePoint, tipPointsOutward, "Preprocessed"));
                }
            }

            // === Y-SHAPE DETECTION: Find pairs of nearby contours ===
            // Y-shaped arrow may be split into two branches - detect by finding adjacent pairs
            // STRICT CONDITIONS to avoid false positives
            var yShapeCandidates = new List<(double score, double angle, double combinedArea,
                VectorOfPoint contour1, VectorOfPoint contour2, PointF basePoint)>();

            for (int i = 0; i < allCandidates.Count; i++)
            {
                for (int j = i + 1; j < allCandidates.Count; j++)
                {
                    var c1 = allCandidates[i];
                    var c2 = allCandidates[j];

                    // Both contours must have reasonably HIGH solidity (individual branches are solid)
                    if (c1.solidity < 0.82 || c2.solidity < 0.82) continue;

                    // STRICT: Both should have similar area (arrow branches are symmetric)
                    double areaRatio = Math.Min(c1.area, c2.area) / Math.Max(c1.area, c2.area);
                    if (areaRatio < 0.4) continue;

                    // Both should be reasonably small (arrow branches are medium sized)
                    double maxBranchArea = outerR * outerR * 0.06;  // Each branch < 6% of ring
                    if (c1.area > maxBranchArea || c2.area > maxBranchArea) continue;

                    // Check angle difference (Y-shape branches are ~15-22° apart)
                    // STRICT: Narrower range to avoid false positives from adjacent data marks
                    double angleDiff = Math.Abs(c1.angle - c2.angle);
                    if (angleDiff > 180) angleDiff = 360 - angleDiff;

                    if (angleDiff >= 15 && angleDiff <= 24)
                    {
                        // Check if both are at similar distance from center (within 0.15R - STRICT)
                        // Real Y-arrow branches are at SAME distance, data marks are scattered
                        double distDiff = Math.Abs(c1.centroidDistRatio - c2.centroidDistRatio);
                        if (distDiff > 0.15) continue;

                        // BOTH branches must be in inner-middle position (0.55-0.78R)
                        // Data marks at outer ring (0.80R+) should NOT be matched as Y-pair
                        if (c1.centroidDistRatio < 0.55 || c1.centroidDistRatio > 0.78) continue;
                        if (c2.centroidDistRatio < 0.55 || c2.centroidDistRatio > 0.78) continue;

                        // STRICT: If BOTH branches have high solidity (>0.92), it's likely data marks
                        // Real arrow branches have lower solidity due to Y-shape gaps
                        if (c1.solidity > 0.92 && c2.solidity > 0.92) continue;

                        double avgCentroidDist = (c1.centroidDistRatio + c2.centroidDistRatio) / 2;

                        // Combined area should be reasonable for arrow
                        double combinedArea = c1.area + c2.area;
                        double expectedArea = outerR * outerR * 0.03;
                        if (combinedArea < expectedArea * 0.3 || combinedArea > expectedArea * 3) continue;

                        // Calculate average angle
                        double avgAngle = (c1.angle + c2.angle) / 2;
                        if (Math.Abs(c1.angle - c2.angle) > 180)
                            avgAngle = (avgAngle + 180) % 360;

                        // Y-shape score - lower base score so shape-matched single contours can win
                        double pairScore = 0.85;  // Lower to let shape match compete
                        if (angleDiff >= 17 && angleDiff <= 21) pairScore += 0.08;  // Ideal angle ~18-20°
                        if (avgCentroidDist >= 0.62 && avgCentroidDist <= 0.70) pairScore += 0.05;

                        // Use midpoint as base
                        var midBase = new PointF(
                            (c1.basePoint.X + c2.basePoint.X) / 2,
                            (c1.basePoint.Y + c2.basePoint.Y) / 2);

                        Log($"    Y-pair: {c1.angle:F0}° + {c2.angle:F0}° = {avgAngle:F0}°, " +
                            $"diff={angleDiff:F0}°, dist={avgCentroidDist:F2}R, score={pairScore:F2}");

                        yShapeCandidates.Add((pairScore, avgAngle, combinedArea,
                            c1.contour, c2.contour, midBase));
                    }
                }
            }

            // Only add Y-shape if it scores higher than best single contour
            var bestSingleScore = allCandidates.Count > 0 ? allCandidates.Max(c => c.score) : 0;
            foreach (var yc in yShapeCandidates)
            {
                if (yc.score > bestSingleScore)
                {
                    allCandidates.Add((yc.score, 0.5, yc.angle, yc.combinedArea, 0.7, 0.85,
                        1.5, yc.contour1, new PointF(0, 0), yc.basePoint, true, "Y-Shape"));
                }
            }

            // Cleanup
            dataMask.Dispose();
            enhanced.Dispose();
            enhancedImg.Dispose();
            binaryResult.Dispose();
            maskedBinary.Dispose();
            kernel.Dispose();
            largerKernel.Dispose();

            Log($"    Found {allCandidates.Count} arrow candidates");

            if (allCandidates.Count > 0)
            {
                var sortedCandidates = allCandidates.OrderByDescending(c => c.score).ToList();

                // Log top candidates
                for (int i = 0; i < Math.Min(5, sortedCandidates.Count); i++)
                {
                    var c = sortedCandidates[i];
                    Log($"    #{i+1}: score={c.score:F2}, solidity={c.solidity:F3}, " +
                        $"centroid={c.centroidDistRatio:F2}R, tip={c.tipDistRatio:F2}R, " +
                        $"elongation={c.elongation:F2}, angle={c.angle:F0}°, method={c.method}");
                }

                var best = sortedCandidates[0];

                if (best.score >= 0.20)
                {
                    int sector = (int)Math.Round(best.angle / 15.0) % 24;
                    double snappedAngle = sector * 15.0;

                    _lastArrowContour = best.contour;
                    _lastArrowTip = best.basePoint;  // Store base point for visualization

                    Log($"  Arrow found: score={best.score:F2}, angle={best.angle:F1}° -> sector {sector} ({snappedAngle}°)");
                    return snappedAngle;
                }
            }

            Log($"  No arrow found");
            return double.MinValue;
        }

        /// <summary>
        /// Calculate arrow confidence score using multiple features
        /// CRITICAL: Arrow has LOW solidity (Y/triangle shape) and is elongated radially
        /// </summary>
        private double CalculateArrowScore(double solidity, double centroidDistRatio, double tipDistRatio,
            double area, float outerR, bool tipPointsOutward, double elongation)
        {
            // Feature 1: Direction - tip should point outward
            double directionScore = tipPointsOutward ? 1.0 : 0.3;

            // Feature 2: Tip position - should be at OUTER part of data ring (0.80R - 0.98R)
            double tipPositionScore;
            if (tipDistRatio >= 0.80 && tipDistRatio <= 0.98)
                tipPositionScore = 1.0;
            else if (tipDistRatio >= 0.70 && tipDistRatio <= 1.02)
                tipPositionScore = 0.6;
            else
                tipPositionScore = 0.2;

            // Feature 3: Centroid position - should be in data ring (0.70R - 0.92R)
            double centroidScore;
            if (centroidDistRatio >= 0.70 && centroidDistRatio <= 0.92)
                centroidScore = 1.0;
            else if (centroidDistRatio >= 0.60 && centroidDistRatio <= 0.98)
                centroidScore = 0.6;
            else
                centroidScore = 0.3;

            // Feature 4: Solidity - arrow (Y/triangle) has LOW solidity (0.4-0.78)
            // Data marks are more rectangular and have HIGH solidity (0.90-1.0)
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

            // Feature 5: Elongation - arrow is elongated (radially stretched)
            double elongationScore;
            if (elongation >= 1.5 && elongation <= 4.0)
                elongationScore = 1.0;
            else if (elongation >= 1.0 && elongation <= 5.0)
                elongationScore = 0.6;
            else
                elongationScore = 0.2;

            // Feature 6: Area - arrow has specific size range
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

        /// <summary>
        /// Create a Y-shaped arrow template contour for shape matching
        /// </summary>
        private VectorOfPoint CreateArrowTemplate()
        {
            // Create Y-shaped arrow pointing RIGHT (0°)
            // The Y-arrow has: stem pointing outward, two branches at the base
            var points = new List<Point>
            {
                // Tip (outer point)
                new Point(100, 50),
                // Right branch
                new Point(30, 20),
                new Point(40, 35),
                // Center junction
                new Point(50, 50),
                // Left branch
                new Point(40, 65),
                new Point(30, 80),
            };

            return new VectorOfPoint(points.ToArray());
        }

        /// <summary>
        /// Calculate shape match score between a contour and the arrow template
        /// Uses Hu Moments for rotation-invariant matching
        /// </summary>
        private double CalculateShapeMatchScore(VectorOfPoint contour, VectorOfPoint template)
        {
            try
            {
                // matchShapes returns 0 for perfect match, higher for worse match
                // Use method I1 (Hu moments)
                double matchValue = CvInvoke.MatchShapes(contour, template, ContoursMatchType.I1, 0);

                // Convert to score (0-1 range, higher is better)
                // matchValue < 0.1 = very good match
                // matchValue > 1.0 = poor match
                if (matchValue < 0.05)
                    return 1.0;  // Excellent match
                else if (matchValue < 0.1)
                    return 0.9;
                else if (matchValue < 0.2)
                    return 0.7;
                else if (matchValue < 0.3)
                    return 0.5;
                else if (matchValue < 0.5)
                    return 0.3;
                else
                    return 0.1;  // Poor match
            }
            catch
            {
                return 0.0;
            }
        }

        /// <summary>
        /// Find arrow by detecting the UNIQUE sector
        /// Arrow sector has a different pattern than data sectors
        /// Compute edge density at outer ring edge - arrow has distinctive edge pattern
        /// </summary>
        private double FindArrowByUniqueSector(Image<Gray, byte> source, RingImageSegmentation.RingRegion region)
        {
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;
            float outerR = region.OuterRadius;
            float innerR = region.InnerRadius;

            // Compute edges using Canny
            var edges = new Image<Gray, byte>(source.Size);
            CvInvoke.Canny(source, edges, 50, 150);

            // Sample the OUTER EDGE region (where arrow is located)
            // Arrow is typically at the very outer edge, slightly outside the data ring
            float sampleR1 = outerR * 0.92f;  // Inner sample radius
            float sampleR2 = outerR * 1.08f;  // Outer sample radius (beyond ring)

            double[] edgeCounts = new double[24];
            double[] intensities = new double[24];

            for (int s = 0; s < 24; s++)
            {
                double sectorAngleDeg = s * 15.0;
                double angleStart = (sectorAngleDeg - 7.5) * Math.PI / 180.0;
                double angleEnd = (sectorAngleDeg + 7.5) * Math.PI / 180.0;

                int edgeCount = 0;
                int pixelCount = 0;
                double intensitySum = 0;

                // Sample pixels in this sector's outer edge region
                for (double r = sampleR1; r <= sampleR2; r += 2)
                {
                    for (double a = angleStart; a <= angleEnd; a += 0.05)
                    {
                        int x = (int)(cx + r * Math.Cos(a));
                        int y = (int)(cy + r * Math.Sin(a));

                        if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                        {
                            if (edges.Data[y, x, 0] > 0) edgeCount++;
                            intensitySum += source.Data[y, x, 0];
                            pixelCount++;
                        }
                    }
                }

                edgeCounts[s] = pixelCount > 0 ? (double)edgeCount / pixelCount : 0;
                intensities[s] = pixelCount > 0 ? intensitySum / pixelCount : 0;
            }

            edges.Dispose();

            // Find the sector with HIGHEST edge density (arrow has more edges than empty space)
            // But also consider: arrow might be the sector with LOWEST edge density if it's a solid shape

            // Compute mean and std of edge counts
            double meanEdge = edgeCounts.Average();
            double stdEdge = Math.Sqrt(edgeCounts.Select(e => Math.Pow(e - meanEdge, 2)).Average());

            // Find outlier sectors (significantly different from mean)
            int bestSector = -1;
            double maxDeviation = 0;

            for (int s = 0; s < 24; s++)
            {
                double deviation = Math.Abs(edgeCounts[s] - meanEdge);
                if (deviation > maxDeviation && deviation > stdEdge * 1.5)
                {
                    maxDeviation = deviation;
                    bestSector = s;
                }
            }

            // Log edge counts
            var edgeStr = string.Join(",", edgeCounts.Select((e, i) => $"{i * 15}°:{e:F3}"));
            Log($"  Edge densities: {edgeStr}");
            Log($"  Mean={meanEdge:F3}, Std={stdEdge:F3}");

            if (bestSector >= 0)
            {
                double resultAngle = bestSector * 15.0;
                Log($"  Unique sector: {bestSector} ({resultAngle}°), edge={edgeCounts[bestSector]:F3}, deviation={maxDeviation:F3}");
                return resultAngle;
            }

            Log($"  No unique sector found");
            return double.MinValue;
        }

        /// <summary>
        /// Find arrow by shape analysis - arrow (Y-shape) has lower solidity than rectangular data marks
        /// Solidity = Area / ConvexHullArea
        /// Data marks (rectangles): solidity ~0.85-0.95
        /// Arrow (Y-shape): solidity ~0.4-0.7
        /// </summary>
        private double FindArrowByShapeAnalysis(Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;
            float outerR = region.OuterRadius;
            float innerR = region.InnerRadius;

            // Find contours in foreground mask
            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(foreground.Clone(), contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            double lowestSolidity = 1.0;
            double arrowAngle = double.MinValue;
            VectorOfPoint arrowContour = null;
            PointF arrowCenter = PointF.Empty;

            // Minimum area threshold (to filter noise)
            double minArea = (outerR - innerR) * (outerR - innerR) * 0.05;
            double maxArea = (outerR - innerR) * (outerR - innerR) * 2.0;

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);

                // Skip too small or too large contours
                if (area < minArea || area > maxArea) continue;

                // Get centroid
                var moments = CvInvoke.Moments(contour);
                if (moments.M00 < 1) continue;
                float ctrX = (float)(moments.M10 / moments.M00);
                float ctrY = (float)(moments.M01 / moments.M00);

                // Arrow must be inside the data ring (0.50R to 0.98R from center)
                double dist = Math.Sqrt(Math.Pow(ctrX - cx, 2) + Math.Pow(ctrY - cy, 2));
                double distRatio = dist / outerR;
                if (distRatio < ARROW_MIN_DIST_RATIO || distRatio > ARROW_MAX_DIST_RATIO) continue;

                // Calculate solidity = Area / ConvexHullArea
                using var hull = new VectorOfPoint();
                CvInvoke.ConvexHull(contour, hull);
                double hullArea = CvInvoke.ContourArea(hull);
                if (hullArea < 1) continue;

                double solidity = area / hullArea;

                // Arrow (Y-shape) has lower solidity than rectangular marks
                // Look for contours with solidity < threshold
                if (solidity < lowestSolidity && solidity < ARROW_SOLIDITY_THRESHOLD)
                {
                    lowestSolidity = solidity;
                    arrowCenter = new PointF(ctrX, ctrY);

                    // Calculate angle from center to this contour
                    double dx = ctrX - cx;
                    double dy = ctrY - cy;
                    arrowAngle = Math.Atan2(dy, dx) * 180.0 / Math.PI;

                    // Store contour for visualization
                    _lastArrowContour = contour;
                    _lastArrowTip = arrowCenter;
                }
            }

            if (arrowAngle != double.MinValue)
            {
                // Snap to nearest 15° sector
                if (arrowAngle < 0) arrowAngle += 360;
                int sector = (int)Math.Round(arrowAngle / 15.0) % 24;
                double snappedAngle = sector * 15.0;

                Log($"  Shape analysis: found contour with solidity={lowestSolidity:F3} at {arrowAngle:F1}° -> sector {sector} ({snappedAngle}°)");
                return snappedAngle;
            }

            Log($"  Shape analysis: no low-solidity contour found");
            return double.MinValue;
        }

        /// <summary>
        /// Discrete sector template matching - extract patch at each of 24 positions
        /// and find which sector best matches the arrow template
        /// </summary>
        private double FindArrowByFastTemplateMatch(Image<Gray, byte> source, RingImageSegmentation.RingRegion region)
        {
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;
            float outerR = region.OuterRadius;

            // Get the template based on ring type
            Image<Gray, byte> template = null;
            if (_lastRingIsLight && _lightTemplateMatcher?.IsLoaded == true)
                template = _lightTemplateMatcher.Template;
            else if (!_lastRingIsLight && _darkTemplateMatcher?.IsLoaded == true)
                template = _darkTemplateMatcher.Template;
            else
                template = _lightTemplateMatcher?.Template ?? _darkTemplateMatcher?.Template;

            if (template == null) return double.MinValue;

            Log($"  Using {(_lastRingIsLight ? "LIGHT" : "DARK")} template ({template.Width}x{template.Height})");

            // Calculate patch size based on ring dimensions
            // Arrow marker is roughly 1-1.5 sectors wide
            float ringWidth = outerR - region.InnerRadius;
            int patchSize = (int)(ringWidth * 0.8f);  // Patch should cover arrow area
            patchSize = Math.Max(40, Math.Min(patchSize, 120));

            // Scale template to match patch size
            var scaledTemplate = template.Resize(patchSize, patchSize, Inter.Linear);

            // Also create INVERTED template (in case colors are reversed)
            var invertedTemplate = new Image<Gray, byte>(scaledTemplate.Size);
            CvInvoke.BitwiseNot(scaledTemplate, invertedTemplate);

            double bestScore = -1;
            int bestSector = 0;
            bool usedInverted = false;
            double[] allScores = new double[24];
            double[] allScoresInv = new double[24];

            // Sample each of 24 sectors
            for (int s = 0; s < 24; s++)
            {
                double sectorAngleDeg = s * 15.0;
                double sectorAngleRad = sectorAngleDeg * Math.PI / 180.0;

                // Position on outer edge at this sector
                float extractR = outerR * 0.85f;  // Slightly inside outer edge
                int px = (int)(cx + extractR * Math.Cos(sectorAngleRad));
                int py = (int)(cy + extractR * Math.Sin(sectorAngleRad));

                // Extract patch centered at this position
                int x1 = px - patchSize / 2;
                int y1 = py - patchSize / 2;

                // Bounds check
                if (x1 < 0 || y1 < 0 || x1 + patchSize >= source.Width || y1 + patchSize >= source.Height)
                    continue;

                source.ROI = new Rectangle(x1, y1, patchSize, patchSize);
                var patch = source.Clone();
                source.ROI = Rectangle.Empty;

                // Try NORMAL template
                using (var result = new Mat())
                {
                    CvInvoke.MatchTemplate(patch, scaledTemplate, result, TemplateMatchingType.CcoeffNormed);
                    double minVal = 0, maxVal = 0;
                    Point minLoc = new Point(), maxLoc = new Point();
                    CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc);
                    allScores[s] = maxVal;
                }

                // Try INVERTED template
                using (var resultInv = new Mat())
                {
                    CvInvoke.MatchTemplate(patch, invertedTemplate, resultInv, TemplateMatchingType.CcoeffNormed);
                    double minVal = 0, maxVal = 0;
                    Point minLoc = new Point(), maxLoc = new Point();
                    CvInvoke.MinMaxLoc(resultInv, ref minVal, ref maxVal, ref minLoc, ref maxLoc);
                    allScoresInv[s] = maxVal;
                }

                // Use the better of the two
                double scoreNormal = allScores[s];
                double scoreInv = allScoresInv[s];
                double bestForSector = Math.Max(scoreNormal, scoreInv);

                if (bestForSector > bestScore)
                {
                    bestScore = bestForSector;
                    bestSector = s;
                    usedInverted = (scoreInv > scoreNormal);
                }

                patch.Dispose();
            }

            scaledTemplate.Dispose();
            invertedTemplate.Dispose();

            // Log all scores for debugging
            var scoreStr = string.Join(",", allScores.Select((sc, i) => $"{i * 15}°:{sc:F2}"));
            var scoreStrInv = string.Join(",", allScoresInv.Select((sc, i) => $"{i * 15}°:{sc:F2}"));
            Log($"  Normal scores: {scoreStr}");
            Log($"  Inverted scores: {scoreStrInv}");

            double resultAngle = bestSector * 15.0;
            Log($"  Template match: best sector={bestSector} ({resultAngle}°), score={bestScore:F3}, inverted={usedInverted}");

            if (bestScore < 0.3)
            {
                Log($"  Score too low, falling back...");
                return double.MinValue;
            }

            return resultAngle;
        }

        /// <summary>
        /// Rotate image by given angle (degrees)
        /// </summary>
        private Image<Gray, byte> RotateImage(Image<Gray, byte> source, double angleDegrees)
        {
            var center = new PointF(source.Width / 2f, source.Height / 2f);
            var rotMatrix = new Mat();
            CvInvoke.GetRotationMatrix2D(center, -angleDegrees, 1.0, rotMatrix);

            var rotated = new Image<Gray, byte>(source.Size);
            CvInvoke.WarpAffine(source, rotated, rotMatrix, source.Size, Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));
            rotMatrix.Dispose();

            return rotated;
        }

        /// <summary>
        /// HALCON-style gradient similarity: S = Σ [(Sx·Tx + Sy·Ty) / (|T|·|S|)]
        /// Normalized dot product of gradient vectors - illumination invariant
        /// </summary>
        private double ComputeGradientSimilarity(Image<Gray, float> patchGx, Image<Gray, float> patchGy,
                                                  Image<Gray, float> templateGx, Image<Gray, float> templateGy)
        {
            double sum = 0;
            int count = 0;

            for (int y = 0; y < templateGx.Height; y++)
            {
                for (int x = 0; x < templateGx.Width; x++)
                {
                    float tx = templateGx.Data[y, x, 0];
                    float ty = templateGy.Data[y, x, 0];
                    float tMag = (float)Math.Sqrt(tx * tx + ty * ty);

                    // Skip low gradient points (no edge)
                    if (tMag < 10) continue;

                    float sx = patchGx.Data[y, x, 0];
                    float sy = patchGy.Data[y, x, 0];
                    float sMag = (float)Math.Sqrt(sx * sx + sy * sy);

                    if (sMag < 1) continue;

                    // Normalized dot product
                    double dotProduct = (sx * tx + sy * ty) / (sMag * tMag);
                    sum += dotProduct;
                    count++;
                }
            }

            return count > 0 ? sum / count : 0;
        }

        /// <summary>
        /// FAST arrow detection by intensity variance analysis
        /// Arrow sector has DIFFERENT pattern than code sectors
        /// For light rings: arrow is typically a small distinct mark (low variance in outer area)
        /// </summary>
        private double FindArrowByIntensityVariance(Image<Gray, byte> source, RingImageSegmentation.RingRegion region)
        {
            int cx = (int)region.Center.X;
            int cy = (int)region.Center.Y;
            float outerR = region.OuterRadius;

            // Sample 24 sectors (15° each) - fast!
            const int NUM_SECTORS = 24;
            double[] sectorScores = new double[NUM_SECTORS];

            // For each sector, calculate how "different" it is from typical code pattern
            for (int s = 0; s < NUM_SECTORS; s++)
            {
                double sectorAngle = s * 15 * Math.PI / 180;
                var outerIntensities = new List<int>();

                // Sample OUTER EDGE area (0.85R to 1.0R) - where arrow marker typically is
                for (int ai = -2; ai <= 2; ai++)  // 5 angle samples within sector
                {
                    double angle = sectorAngle + ai * 3 * Math.PI / 180;
                    double cosA = Math.Cos(angle);
                    double sinA = Math.Sin(angle);

                    for (float r = outerR * 0.85f; r <= outerR * 1.0f; r += outerR * 0.05f)
                    {
                        int x = (int)(cx + r * cosA);
                        int y = (int)(cy + r * sinA);

                        if (x >= 0 && x < source.Width && y >= 0 && y < source.Height)
                        {
                            outerIntensities.Add(source.Data[y, x, 0]);
                        }
                    }
                }

                if (outerIntensities.Count < 5) continue;

                // Calculate statistics
                double mean = outerIntensities.Average();
                double variance = outerIntensities.Sum(v => Math.Pow(v - mean, 2)) / outerIntensities.Count;

                // Arrow sector characteristics:
                // - For light ring with dark arrow: outer area has LOW mean (dark arrow mark)
                // - For light ring with light arrow: outer area has HIGH mean but LOW variance
                // Key: arrow sector has more UNIFORM intensity at outer edge
                sectorScores[s] = variance;
            }

            // Find sector with LOWEST variance at outer edge (most uniform = likely arrow)
            double minVariance = double.MaxValue;
            int arrowSector = 0;

            for (int s = 0; s < NUM_SECTORS; s++)
            {
                if (sectorScores[s] > 0 && sectorScores[s] < minVariance)
                {
                    minVariance = sectorScores[s];
                    arrowSector = s;
                }
            }

            double avgVariance = sectorScores.Where(v => v > 0).DefaultIfEmpty(1).Average();

            // Only accept if this sector is significantly more uniform than average
            if (minVariance < avgVariance * 0.7)
            {
                double arrowAngle = arrowSector * 15.0;
                Log($"  Arrow found at {arrowAngle}° (variance={minVariance:F0}, avg={avgVariance:F0})");
                return arrowAngle;
            }

            Log($"  No clear arrow by variance (min={minVariance:F0}, avg={avgVariance:F0})");
            return double.MinValue;
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
        /// Decode using fast sampling method (optimized from HALCON Arrow_Decode algorithm)
        /// Divides ring into 24 segments, each with inner and outer parts
        /// Uses dense sampling instead of mask operations for speed
        /// </summary>
        private string DecodeWithRegionIntersection(Image<Gray, byte> foreground, PointF center,
            float innerRadius, float centerRadius, float outerRadius, double startAngleDeg)
        {
            var binaryString = new System.Text.StringBuilder(48);
            double segmentRad = SEGMENT_ANGLE * Math.PI / 180;
            double startRad = startAngleDeg * Math.PI / 180;
            int cx = (int)center.X;
            int cy = (int)center.Y;
            int w = foreground.Width;
            int h = foreground.Height;

            if (EnableDetailedLog)
            {
                Log($"  Decoding with startAngle={startAngleDeg:F1}°, threshold={FILL_THRESHOLD:P0}");
                Log($"  Seg# | Angle°  | Inner% | InBit | Outer% | OutBit");
                Log($"  -----|---------|--------|-------|--------|-------");
            }

            // Pre-compute sin/cos tables for 8 angular samples per segment
            const int ANGULAR_SAMPLES = 8;
            const int RADIAL_SAMPLES = 6;
            double[] cosTable = new double[ANGULAR_SAMPLES];
            double[] sinTable = new double[ANGULAR_SAMPLES];

            for (int i = 0; i < SEGMENTS; i++)
            {
                // Calculate segment angular boundaries
                double angle1 = startRad + (i * segmentRad) - segmentRad;
                double angle2 = startRad + ((i + 1) * segmentRad) - segmentRad;
                double midAngleDeg = ((angle1 + angle2) / 2) * 180 / Math.PI;

                // Pre-compute angles for this segment (avoid edge lines)
                double angleMargin = segmentRad * 0.1; // 10% margin from edges
                double angleStep = (angle2 - angle1 - 2 * angleMargin) / (ANGULAR_SAMPLES - 1);
                for (int a = 0; a < ANGULAR_SAMPLES; a++)
                {
                    double angle = angle1 + angleMargin + a * angleStep;
                    cosTable[a] = Math.Cos(angle);
                    sinTable[a] = Math.Sin(angle);
                }

                // Sample inner ring (innerRadius to centerRadius)
                int innerWhite = 0, innerTotal = 0;
                float innerStep = (centerRadius - innerRadius) / RADIAL_SAMPLES;
                for (int r = 0; r < RADIAL_SAMPLES; r++)
                {
                    float radius = innerRadius + innerStep * (r + 0.5f);
                    for (int a = 0; a < ANGULAR_SAMPLES; a++)
                    {
                        int x = (int)(cx + radius * cosTable[a]);
                        int y = (int)(cy + radius * sinTable[a]);
                        if (x >= 0 && x < w && y >= 0 && y < h)
                        {
                            innerTotal++;
                            if (foreground.Data[y, x, 0] > 128)
                                innerWhite++;
                        }
                    }
                }
                double innerFillRatio = innerTotal > 0 ? (double)innerWhite / innerTotal : 0;
                int innerBit = innerFillRatio >= FILL_THRESHOLD ? 1 : 0;

                // Sample outer ring (centerRadius to outerRadius)
                int outerWhite = 0, outerTotal = 0;
                float outerStep = (outerRadius - centerRadius) / RADIAL_SAMPLES;
                for (int r = 0; r < RADIAL_SAMPLES; r++)
                {
                    float radius = centerRadius + outerStep * (r + 0.5f);
                    for (int a = 0; a < ANGULAR_SAMPLES; a++)
                    {
                        int x = (int)(cx + radius * cosTable[a]);
                        int y = (int)(cy + radius * sinTable[a]);
                        if (x >= 0 && x < w && y >= 0 && y < h)
                        {
                            outerTotal++;
                            if (foreground.Data[y, x, 0] > 128)
                                outerWhite++;
                        }
                    }
                }
                double outerFillRatio = outerTotal > 0 ? (double)outerWhite / outerTotal : 0;
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
        /// Calculate the ratio of WHITE pixels in a segment (OPTIMIZED - legacy method)
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

                // Draw white region contours with SOLIDITY labels for debugging
                if (result.ForegroundMask != null)
                {
                    using var contours = new VectorOfVectorOfPoint();
                    using var hierarchy = new Mat();
                    CvInvoke.FindContours(result.ForegroundMask.Clone(), contours, hierarchy,
                        RetrType.External, ChainApproxMethod.ChainApproxSimple);

                    int blobCx = (int)result.Center.X;
                    int blobCy = (int)result.Center.Y;
                    float blobOuterR = result.OuterRadius;

                    // Analyze and draw each blob
                    for (int bi = 0; bi < contours.Size; bi++)
                    {
                        var contour = contours[bi];
                        double blobArea = CvInvoke.ContourArea(contour);
                        if (blobArea < 100) continue;  // Skip tiny contours

                        // Get centroid
                        var moments = CvInvoke.Moments(contour);
                        if (moments.M00 < 1) continue;
                        float blobX = (float)(moments.M10 / moments.M00);
                        float blobY = (float)(moments.M01 / moments.M00);

                        // Check distance from center
                        double blobDist = Math.Sqrt(Math.Pow(blobX - blobCx, 2) + Math.Pow(blobY - blobCy, 2));
                        double blobDistRatio = blobDist / blobOuterR;
                        if (blobDistRatio < 0.4 || blobDistRatio > 1.2) continue;  // Not in ring area

                        // Calculate solidity
                        using var hull = new VectorOfPoint();
                        CvInvoke.ConvexHull(contour, hull);
                        double hullArea = CvInvoke.ContourArea(hull);
                        double solidity = hullArea > 0 ? blobArea / hullArea : 1.0;

                        // Calculate angle from center
                        double blobAngle = Math.Atan2(blobY - blobCy, blobX - blobCx) * 180.0 / Math.PI;
                        if (blobAngle < 0) blobAngle += 360;

                        // Color based on solidity: LOW solidity = CYAN (potential arrow), HIGH = RED
                        MCvScalar blobColor;
                        if (solidity < 0.70)
                            blobColor = new MCvScalar(255, 255, 0);  // Cyan - low solidity (arrow candidate)
                        else if (solidity < 0.85)
                            blobColor = new MCvScalar(0, 255, 255);  // Yellow - medium solidity
                        else
                            blobColor = new MCvScalar(0, 0, 255);    // Red - high solidity (data mark)

                        // Draw contour
                        using var singleContour = new VectorOfVectorOfPoint();
                        singleContour.Push(contour);
                        CvInvoke.DrawContours(visualization, singleContour, 0, blobColor, 2);

                        // Label with solidity and angle
                        string blobLabel = $"S:{solidity:F2} A:{blobAngle:F0}";
                        CvInvoke.PutText(visualization, blobLabel, new Point((int)blobX - 30, (int)blobY - 5),
                            FontFace.HersheySimplex, 0.35, blobColor, 1);
                    }
                }

                // *** SPECIAL ARROW MARKING - Draw arrow with FILLED RED color ***
                if (result.ArrowContour != null && result.ArrowContour.Size > 0)
                {
                    using var arrowContourVector = new VectorOfVectorOfPoint();
                    arrowContourVector.Push(result.ArrowContour);

                    // Draw filled red arrow
                    CvInvoke.DrawContours(visualization, arrowContourVector, 0, new MCvScalar(0, 0, 255), -1);  // -1 = filled

                    // Draw thick red outline
                    CvInvoke.DrawContours(visualization, arrowContourVector, 0, new MCvScalar(0, 0, 200), 3);

                    // Draw arrow tip marker
                    if (result.ArrowTip != PointF.Empty)
                    {
                        CvInvoke.Circle(visualization, Point.Round(result.ArrowTip), 5, new MCvScalar(255, 255, 0), -1);  // Yellow dot
                        CvInvoke.Circle(visualization, Point.Round(result.ArrowTip), 5, new MCvScalar(0, 0, 0), 2);  // Black outline
                    }
                }
                else
                {
                    // *** ARROW LINE INDICATOR (fallback when no contour) ***
                    // Draw thick MAGENTA line from center to arrow position using RotationAngle
                    double arrowAngleRad = result.RotationAngle * Math.PI / 180;
                    int arrowEndX = (int)(result.Center.X + result.OuterRadius * 1.1 * Math.Cos(arrowAngleRad));
                    int arrowEndY = (int)(result.Center.Y + result.OuterRadius * 1.1 * Math.Sin(arrowAngleRad));

                    // Thick magenta line
                    CvInvoke.Line(visualization, Point.Round(result.Center), new Point(arrowEndX, arrowEndY),
                        new MCvScalar(255, 0, 255), 4);  // Magenta, thickness 4

                    // Arrow head circle at tip
                    CvInvoke.Circle(visualization, new Point(arrowEndX, arrowEndY), 8, new MCvScalar(255, 0, 255), -1);  // Filled magenta
                    CvInvoke.Circle(visualization, new Point(arrowEndX, arrowEndY), 8, new MCvScalar(0, 0, 0), 2);  // Black outline

                    // Draw "ARROW" text near the tip
                    CvInvoke.PutText(visualization, "ARROW", new Point(arrowEndX + 10, arrowEndY - 10),
                        FontFace.HersheySimplex, 0.5, new MCvScalar(255, 0, 255), 2);
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

                // *** MAIN ARROW INDICATOR - THICK MAGENTA line showing detected arrow angle ***
                {
                    double arrowAngleRad = result.RotationAngle * Math.PI / 180;
                    int arrowX = (int)(result.Center.X + result.OuterRadius * 1.1 * Math.Cos(arrowAngleRad));
                    int arrowY = (int)(result.Center.Y + result.OuterRadius * 1.1 * Math.Sin(arrowAngleRad));

                    // Draw thick magenta arrow showing detected arrow direction
                    var arrowColor = new MCvScalar(255, 0, 255);  // Magenta (BGR)
                    CvInvoke.ArrowedLine(visualization, Point.Round(result.Center), new Point(arrowX, arrowY),
                        arrowColor, 4, LineType.AntiAlias, 0, 0.2);

                    // Draw circle at arrow endpoint
                    CvInvoke.Circle(visualization, new Point(arrowX, arrowY), 10, arrowColor, 3);

                    // Label showing angle
                    CvInvoke.PutText(visualization, $"Arrow: {result.RotationAngle:F1}°",
                        new Point(arrowX - 40, arrowY + 25),
                        FontFace.HersheySimplex, 0.4, arrowColor, 1);
                }

                // Draw arrow indicator to locator point (green, thinner)
                if (result.LocatorPoints.Count > 0)
                {
                    var arrowTip = result.LocatorPoints[0];
                    CvInvoke.ArrowedLine(visualization, Point.Round(result.Center), Point.Round(arrowTip),
                        new MCvScalar(0, 255, 0), 2);
                }

                // Draw template match result (CYAN for visibility)
                if (result.TemplateMatchCenter.HasValue)
                {
                    var matchColor = new MCvScalar(255, 255, 0);  // Cyan (BGR)

                    // Draw matched contour
                    if (result.TemplateMatchContour != null && result.TemplateMatchContour.Size > 0)
                    {
                        using var contourVector = new VectorOfVectorOfPoint();
                        contourVector.Push(result.TemplateMatchContour);
                        CvInvoke.DrawContours(visualization, contourVector, 0, matchColor, 2);
                    }

                    // Draw center of matched arrow
                    var matchCenter = Point.Round(result.TemplateMatchCenter.Value);
                    CvInvoke.Circle(visualization, matchCenter, 8, matchColor, 2);
                    CvInvoke.Circle(visualization, matchCenter, 3, matchColor, -1);

                    // Draw arrow from ring center to matched arrow center
                    CvInvoke.ArrowedLine(visualization, Point.Round(result.Center), matchCenter,
                        matchColor, 2, LineType.AntiAlias, 0, 0.3);

                    // Draw label for template match
                    string matchLabel = $"{result.TemplateMatchType} ({result.TemplateMatchScore:F2})";
                    CvInvoke.PutText(visualization, matchLabel,
                        new Point(matchCenter.X - 30, matchCenter.Y - 15),
                        FontFace.HersheySimplex, 0.35, matchColor, 1);
                }

                // Draw label
                string label = result.IsValid
                    ? $"#{result.RingIndex}: {result.DecodedData}"
                    : $"#{result.RingIndex}: Err";
                CvInvoke.PutText(visualization, label,
                    new Point((int)result.Center.X - 60, (int)result.Center.Y - (int)result.OuterRadius - 10),
                    FontFace.HersheySimplex, 0.4, mainColor, 1);

                // Draw rotated thumbnail in corner (similar to HALCON - arrow points right)
                try
                {
                    var rotatedThumb = CreateRotatedThumbnail(source, result, 120);
                    if (rotatedThumb != null)
                    {
                        // Position thumbnail near the ring (offset to bottom-right)
                        int thumbX = (int)(result.Center.X + result.OuterRadius * 0.5);
                        int thumbY = (int)(result.Center.Y + result.OuterRadius * 0.5);

                        // Ensure thumbnail fits within visualization bounds
                        thumbX = Math.Min(thumbX, visualization.Width - rotatedThumb.Width - 5);
                        thumbY = Math.Min(thumbY, visualization.Height - rotatedThumb.Height - 5);
                        thumbX = Math.Max(5, thumbX);
                        thumbY = Math.Max(5, thumbY);

                        // Draw border around thumbnail
                        var thumbRect = new Rectangle(thumbX - 2, thumbY - 2, rotatedThumb.Width + 4, rotatedThumb.Height + 4);
                        CvInvoke.Rectangle(visualization, thumbRect, new MCvScalar(255, 255, 0), 2);

                        // Copy thumbnail to visualization
                        var roi = new Rectangle(thumbX, thumbY, rotatedThumb.Width, rotatedThumb.Height);
                        visualization.ROI = roi;
                        rotatedThumb.CopyTo(visualization);
                        visualization.ROI = Rectangle.Empty;

                        // Add "→" indicator to show arrow direction
                        CvInvoke.PutText(visualization, "->",
                            new Point(thumbX + rotatedThumb.Width / 2 - 10, thumbY - 5),
                            FontFace.HersheySimplex, 0.4, new MCvScalar(255, 255, 0), 1);
                    }
                }
                catch { /* Ignore thumbnail errors */ }
            }

            return visualization;
        }

        /// <summary>
        /// Create a rotated thumbnail of the ring code with arrow pointing right
        /// Similar to HALCON's Affine_Trans_CorrectPosition
        /// </summary>
        private Image<Bgr, byte> CreateRotatedThumbnail(Image<Bgr, byte> source, RingCodeResult result, int size)
        {
            if (result.OuterRadius < 10) return null;

            try
            {
                // Extract ring region
                int margin = (int)(result.OuterRadius * 1.2);
                int x = Math.Max(0, (int)(result.Center.X - margin));
                int y = Math.Max(0, (int)(result.Center.Y - margin));
                int w = Math.Min(margin * 2, source.Width - x);
                int h = Math.Min(margin * 2, source.Height - y);

                if (w <= 0 || h <= 0) return null;

                source.ROI = new Rectangle(x, y, w, h);
                var cropped = source.Clone();
                source.ROI = Rectangle.Empty;

                // Calculate rotation angle to make arrow point right (0 degrees)
                // Arrow is at RotationAngle, we want it at 0
                double rotationNeeded = -result.RotationAngle;

                // Rotate around center
                float cx = cropped.Width / 2f;
                float cy = cropped.Height / 2f;

                var rotationMat = new Mat();
                CvInvoke.GetRotationMatrix2D(new PointF(cx, cy), rotationNeeded, 1.0, rotationMat);

                var rotated = new Image<Bgr, byte>(cropped.Size);
                CvInvoke.WarpAffine(cropped, rotated, rotationMat, cropped.Size);

                // Resize to thumbnail size
                var thumbnail = new Image<Bgr, byte>(size, size);
                CvInvoke.Resize(rotated, thumbnail, new System.Drawing.Size(size, size));

                // Draw arrow direction indicator on thumbnail
                int arrowLen = size / 3;
                int centerX = size / 2;
                int centerY = size / 2;
                CvInvoke.ArrowedLine(thumbnail,
                    new Point(centerX, centerY),
                    new Point(centerX + arrowLen, centerY),
                    new MCvScalar(0, 255, 255), 2, LineType.AntiAlias, 0, 0.3);

                return thumbnail;
            }
            catch
            {
                return null;
            }
        }
    }
}
