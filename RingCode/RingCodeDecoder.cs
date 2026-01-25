#if ANDROID || WINDOWS
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using PointF = System.Drawing.PointF;
using Point = System.Drawing.Point;
using Rectangle = System.Drawing.Rectangle;

#if WINDOWS
using CameraMaui.ShapeMatcher;
#endif

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

        // Debug output - set to a valid directory path to save arrow detection debug images
        public static string DebugOutputDir { get; set; } = null;

        // Arrow template matchers
        private ArrowTemplateMatcher _darkTemplateMatcher;
        private ArrowTemplateMatcher _lightTemplateMatcher;
        private bool _templatesInitialized = false;

#if WINDOWS
        // Shape-based matcher (higher accuracy, Windows only)
        private IShapeBasedMatcher? _shapeMatcher;
        private bool _shapeMatcherInitialized = false;
        private const string SHAPE_MATCHER_CLASS_ID = "y_arrow";
#endif

        /// <summary>
        /// Initialize arrow template matchers from default paths
        /// </summary>
        // Cached template paths (found once, reused)
        private static string _cachedDarkTemplatePath = null;
        private static string _cachedLightTemplatePath = null;
        private static bool _templatePathsSearched = false;

        public void InitializeTemplates()
        {
            if (_templatesInitialized) return;

            _darkTemplateMatcher = new ArrowTemplateMatcher();
            _lightTemplateMatcher = new ArrowTemplateMatcher();

            // Only search for paths once (static cache)
            if (!_templatePathsSearched)
            {
                _templatePathsSearched = true;
                FindTemplatePaths();
            }

            if (_cachedDarkTemplatePath != null)
            {
                _darkTemplateMatcher.LoadTemplate(_cachedDarkTemplatePath);
                Log($"[Template] Dark loaded: {_darkTemplateMatcher.Template?.Width}x{_darkTemplateMatcher.Template?.Height}");
            }

            if (_cachedLightTemplatePath != null)
            {
                _lightTemplateMatcher.LoadTemplate(_cachedLightTemplatePath);
                Log($"[Template] Light loaded: {_lightTemplateMatcher.Template?.Width}x{_lightTemplateMatcher.Template?.Height}");
            }

            if (_cachedDarkTemplatePath == null && _cachedLightTemplatePath == null)
            {
                Log($"[Template] WARNING: No templates found!");
            }

            _templatesInitialized = true;

#if WINDOWS
            // Initialize shape-based matcher for higher accuracy arrow detection
            InitializeShapeMatcher();
#endif
        }

#if WINDOWS
        /// <summary>
        /// Initialize the shape-based matcher with Y-arrow templates
        /// </summary>
        private void InitializeShapeMatcher()
        {
            if (_shapeMatcherInitialized) return;

            try
            {
                _shapeMatcher = ShapeMatcherFactory.Create();
                Log($"[ShapeMatcher] Using {(ShapeMatcherFactory.IsNativeAvailable ? "native" : "managed")} implementation");
                if (!_shapeMatcher.IsReady)
                {
                    Log($"[ShapeMatcher] Failed to initialize");
                    _shapeMatcher = null;
                    return;
                }

                // Try to load pre-saved templates
                var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                var templatesPath = Path.Combine(appData, "y_arrow_templates.yml");

                if (File.Exists(templatesPath))
                {
                    int loaded = _shapeMatcher.LoadTemplates(templatesPath);
                    if (loaded > 0)
                    {
                        Log($"[ShapeMatcher] Loaded {loaded} templates from {templatesPath}");
                        _shapeMatcherInitialized = true;
                        return;
                    }
                }

                // Try to load REAL Y-arrow template from file (extracted from actual images)
                var realTemplatePath = FindRealYArrowTemplate();
                if (realTemplatePath != null && File.Exists(realTemplatePath))
                {
                    Log($"[ShapeMatcher] Loading real Y-arrow template from: {realTemplatePath}");
                    using var templateImage = new Image<Gray, byte>(realTemplatePath);
                    int added = _shapeMatcher.AddTemplateWithRotations(templateImage, SHAPE_MATCHER_CLASS_ID,
                        angleStart: 0f, angleEnd: 360f, angleStep: 10f);

                    if (added > 0)
                    {
                        Log($"[ShapeMatcher] Created {added} Y-arrow templates from real image");
                        _shapeMatcher.SaveTemplates(templatesPath);
                        _shapeMatcherInitialized = true;
                        return;
                    }
                }

                // Fallback: Create programmatic Y-arrow template
                Log($"[ShapeMatcher] WARNING: Real template not found, using programmatic template");
                using var fallbackTemplate = EmguCvExtensions.CreateYArrowTemplate(80);
                int fallbackAdded = _shapeMatcher.AddTemplateWithRotations(fallbackTemplate, SHAPE_MATCHER_CLASS_ID,
                    angleStart: 0f, angleEnd: 360f, angleStep: 15f);

                if (fallbackAdded > 0)
                {
                    Log($"[ShapeMatcher] Created {fallbackAdded} Y-arrow templates (programmatic)");
                    _shapeMatcher.SaveTemplates(templatesPath);
                    _shapeMatcherInitialized = true;
                }
                else
                {
                    Log($"[ShapeMatcher] Failed to create templates");
                    _shapeMatcher.Dispose();
                    _shapeMatcher = null;
                }
            }
            catch (Exception ex)
            {
                Log($"[ShapeMatcher] Initialization error: {ex.Message}");
                _shapeMatcher?.Dispose();
                _shapeMatcher = null;
            }
        }

        /// <summary>
        /// Find the real Y-arrow template file (extracted from actual ring code images)
        /// </summary>
        private static string FindRealYArrowTemplate()
        {
            // Search locations for the real Y-arrow template
            var searchPaths = new List<string>();

            // 1. ArrowDetectionTest output folder (where debug images are saved)
            var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            searchPaths.Add(Path.Combine(desktopPath, "ArrowDetectionTest", "arrow_template_active.png"));

            // 2. LocalApplicationData (where TemplateCreatorPage saves templates)
            var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            searchPaths.Add(Path.Combine(appData, "arrow_template_active.png"));
            searchPaths.Add(Path.Combine(appData, "arrow_template_dark.png"));
            searchPaths.Add(Path.Combine(appData, "arrow_template_light.png"));

            // 3. DebugOutputDir if set
            if (!string.IsNullOrEmpty(DebugOutputDir))
            {
                searchPaths.Add(Path.Combine(DebugOutputDir, "arrow_template_active.png"));
                searchPaths.Add(Path.Combine(DebugOutputDir, "arrow_template.png"));
            }

            // 4. Current directory
            searchPaths.Add("arrow_template_active.png");
            searchPaths.Add("arrow_template.png");

            foreach (var path in searchPaths)
            {
                if (File.Exists(path))
                {
                    Log($"[ShapeMatcher] Found real template at: {path}");
                    return path;
                }
            }

            Log($"[ShapeMatcher] Real template not found in any search path");
            return null;
        }
#endif

        private static void FindTemplatePaths()
        {
            var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            Log($"[Template] LocalApplicationData = {appData}");

            // Check 1: Direct LocalApplicationData (for packaged MAUI apps, this IS the correct path)
            var darkPath = Path.Combine(appData, "arrow_template_dark.png");
            var lightPath = Path.Combine(appData, "arrow_template_light.png");

            Log($"[Template] Check: dark exists={File.Exists(darkPath)}, light exists={File.Exists(lightPath)}");

            if (File.Exists(darkPath)) _cachedDarkTemplatePath = darkPath;
            if (File.Exists(lightPath)) _cachedLightTemplatePath = lightPath;

            if (_cachedDarkTemplatePath != null || _cachedLightTemplatePath != null)
            {
                Log($"[Template] Found in direct path");
                return;
            }

            // Check 2: For console apps, search in Packages folder
            var packagesPath = Path.Combine(appData, "Packages");
            if (!Directory.Exists(packagesPath))
            {
                Log($"[Template] No Packages folder, templates NOT FOUND");
                return;
            }

            try
            {
                foreach (var dir in Directory.GetDirectories(packagesPath))
                {
                    CheckPackageDir(dir, ref darkPath, ref lightPath);
                    if (_cachedDarkTemplatePath != null && _cachedLightTemplatePath != null)
                    {
                        Log($"[Template] Found in package dir");
                        return;
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"[Template] Error: {ex.Message}");
            }

            if (_cachedDarkTemplatePath == null && _cachedLightTemplatePath == null)
            {
                Log($"[Template] WARNING: No templates found!");
            }
        }

        private static void CheckPackageDir(string packageDir, ref string darkPath, ref string lightPath)
        {
            var localPath = Path.Combine(packageDir, "LocalCache", "Local");
            if (!Directory.Exists(localPath)) return;

            if (_cachedDarkTemplatePath == null)
            {
                var testDark = Path.Combine(localPath, "arrow_template_dark.png");
                if (File.Exists(testDark)) _cachedDarkTemplatePath = testDark;
            }
            if (_cachedLightTemplatePath == null)
            {
                var testLight = Path.Combine(localPath, "arrow_template_light.png");
                if (File.Exists(testLight)) _cachedLightTemplatePath = testLight;
            }
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

        // Green line angle (apex → baseMid) for accurate rotation
        private double? _lastGreenLineAngle = null;

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

            // Green line angle (apex → baseMid) for accurate rotation
            public double? GreenLineAngle { get; set; }
            public string TemplateMatchType { get; set; } = "";  // "Dark" or "Light"

            // Arrow contour for special marking
            public VectorOfPoint ArrowContour { get; set; }
            public VectorOfPoint ArrowContour2 { get; set; }  // Second contour for Y-shape arrows
            public PointF ArrowTip { get; set; }
            public string ArrowDetectionMethod { get; set; } = "";  // "Template", "Contour", "ContourInv", "Intensity"
            public bool IsYShapeArrow { get; set; } = false;  // True if arrow detected as Y-shape pair
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

                // Step 2: Extract white foreground region (also refines OuterRadius)
                var foregroundMask = ExtractForegroundRegion(ringImage, region);
                result.ForegroundMask = foregroundMask;
                result.OuterRadius = region.OuterRadius;  // Update with refined radius
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
                    result.GreenLineAngle = _lastGreenLineAngle;  // Store green line angle for rotation
                }

                // Capture arrow contour for visualization
                if (_lastArrowContour != null)
                {
                    result.ArrowContour = _lastArrowContour;
                    result.ArrowContour2 = _lastArrowContour2;  // Second contour for Y-shape
                    result.IsYShapeArrow = _lastIsYShapeArrow;
                    result.ArrowTip = _lastArrowTip;
                }
                long t3 = sw.ElapsedMilliseconds;

                // Step 4 & 5: Fast decode with fallback
                // Use actual InnerRadius from segmentation instead of fixed ratio
                float bigRadius = region.OuterRadius;
                float innerRadius = region.InnerRadius;  // Use actual inner boundary from radial scan

                // CenterRadius = midpoint between inner and outer boundaries
                float centerRadius = (innerRadius + bigRadius) / 2f;

                Log($"Sampling radii: inner={innerRadius:F0}, center={centerRadius:F0}, outer={bigRadius:F0}");

                string bestBinary = "";
                string bestDecoded = "-1";
                double bestAngle = baseAngle;
                bool bestHasBothValid = false;  // Both parity and BCC valid
                bool found = false;

                // If ShapeMatcher found the arrow, use its angle DIRECTLY (no offset search)
                // ShapeMatcher uses contour base point which is precise
                if (_lastMatchType == "ShapeMatcher")
                {
                    // Direct decode using ShapeMatcher angle - no offset search
                    bestBinary = DecodeWithRegionIntersection(
                        foregroundMask, region.Center, innerRadius, centerRadius, bigRadius, baseAngle);
                    var (decoded, parityValid, bccValid) = DecryptBinaryWithValidation(bestBinary);
                    bestDecoded = decoded;
                    bestAngle = baseAngle;
                    bestHasBothValid = parityValid && bccValid;
                    found = decoded != "-1" && decoded != "0";

                    Log($"  [ShapeMatcher] Direct decode: angle={baseAngle:F1}°, parity={parityValid}, BCC={bccValid}");
                }
                else
                {
                    // Fallback: Try multiple angles for other detection methods
                    // Extended search range to handle arrow detection errors up to ±37.5°
                    double[] angleOffsets = { 0, 7.5, -7.5, 15, -15, 22.5, -22.5, 30, -30, 37.5, -37.5 };

                    // Collect all valid decode attempts
                    var validDecodes = new List<(double angle, string binary, string decoded, bool parityValid, bool bccValid, double absOffset)>();

                    foreach (double offset in angleOffsets)
                    {
                        double testAngle = baseAngle + offset;
                        string binary = DecodeWithRegionIntersection(
                            foregroundMask, region.Center, innerRadius, centerRadius, bigRadius, testAngle);
                        var (decoded, parityValid, bccValid) = DecryptBinaryWithValidation(binary);

                        if (decoded != "-1" && decoded != "0")
                        {
                            validDecodes.Add((testAngle, binary, decoded, parityValid, bccValid, Math.Abs(offset)));
                        }
                    }

                    // Select best decode: prefer both valid, then smaller offset
                    if (validDecodes.Count > 0)
                    {
                        var ranked = validDecodes
                            .OrderByDescending(d => d.parityValid && d.bccValid)  // Both valid first
                            .ThenBy(d => d.absOffset)  // Smaller offset preferred
                            .First();

                        bestBinary = ranked.binary;
                        bestDecoded = ranked.decoded;
                        bestAngle = ranked.angle;
                        bestHasBothValid = ranked.parityValid && ranked.bccValid;
                        found = true;

                        Log($"  Selected decode: angle={ranked.angle:F1}°, parity={ranked.parityValid}, BCC={ranked.bccValid}, offset={ranked.absOffset:F1}°");
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

            // Step 1: Apply CLAHE for local contrast enhancement, then Gaussian blur
            // CLAHE enhances local contrast, helping detect faint marks
            Image<Gray, byte> claheEnhanced;
            try
            {
                var matClahe = new Mat();
                CvInvoke.CLAHE(ringImage, 3.0, new System.Drawing.Size(8, 8), matClahe);
                claheEnhanced = matClahe.ToImage<Gray, byte>();
                matClahe.Dispose();
            }
            catch
            {
                // Fallback if CLAHE fails
                claheEnhanced = ringImage.Clone();
            }

            var smoothed = new Image<Gray, byte>(ringImage.Size);
            CvInvoke.GaussianBlur(claheEnhanced, smoothed, new System.Drawing.Size(5, 5), 0);
            claheEnhanced.Dispose();

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

            // Step 4: Determine if marks are BRIGHTER or DARKER than background
            // Key insight: background is the majority, marks are the minority (~30-40%)
            // We need to detect:
            //   - Light ring with dark marks: bright background, dark marks → invert binary
            //   - Dark ring with bright marks: dark background, bright marks → keep binary
            //   - Gray ring with white marks: gray background, white marks → keep binary (NOT invert!)
            bool isLightRing;
            bool marksAreBrighter;  // New flag: true if data marks are brighter than background
            string reason;

            // Use percentile analysis to determine mark vs background brightness
            dataIntensities.Sort();
            int p25 = dataIntensities[dataIntensities.Count / 4];
            int p50 = dataIntensities[dataIntensities.Count / 2];  // median = background
            int p75 = dataIntensities[3 * dataIntensities.Count / 4];
            int p95 = dataIntensities[(int)(dataIntensities.Count * 0.95)];
            int p05 = dataIntensities[(int)(dataIntensities.Count * 0.05)];

            // The spread above median vs below median tells us which direction marks go
            int spreadAbove = p95 - p50;  // How much brighter are the bright outliers
            int spreadBelow = p50 - p05;  // How much darker are the dark outliers

            // Marks are brighter if there's more spread above median than below
            // This handles the gray-ring-with-white-marks case correctly
            marksAreBrighter = spreadAbove > spreadBelow * 1.3;

            // Traditional light/dark classification (for logging/template selection)
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
                isLightRing = ringMean > 128;
                reason = $"close counts, mean={ringMean:F0}";
            }

            // Override: if marks are brighter, treat as dark ring processing (no inversion)
            if (isLightRing && marksAreBrighter)
            {
                Log($"  OVERRIDE: Gray ring with WHITE marks detected (spreadAbove={spreadAbove} > spreadBelow={spreadBelow}*1.3)");
                isLightRing = false;  // Process as dark ring (no binary inversion)
                reason = $"gray ring with white marks, p50={p50}, p95={p95}";
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

            // Step 6b: LOCAL_THRESHOLD approach (similar to HALCON)
            // HALCON: mean_image + dyn_threshold
            // Compute local mean using box filter, then threshold based on difference
            int localBlockSize = Math.Max(41, (int)(region.OuterRadius / 3) | 1);
            localBlockSize = Math.Min(localBlockSize, 151);

            // CRITICAL FIX: Mask out areas outside the ring BEFORE computing localMean
            // Otherwise, the black background pulls down localMean at ring edges,
            // causing edge pixels to be falsely detected as "brighter than mean"
            var maskedSmoothed = new Image<Gray, byte>(smoothed.Size);
            // Fill with ring's mean intensity (so edges don't get pulled by black background)
            maskedSmoothed.SetValue(new MCvScalar(ringMean));
            // Copy only the ring area from smoothed image
            smoothed.Copy(maskedSmoothed, ringMask);

            var localMean = new Image<Gray, byte>(smoothed.Size);
            CvInvoke.Blur(maskedSmoothed, localMean, new System.Drawing.Size(localBlockSize, localBlockSize), new Point(-1, -1));
            maskedSmoothed.Dispose();

            // Dynamic threshold: compare pixel to local mean
            // For light rings: pixel < localMean - offset means dark mark (foreground)
            // For dark rings: pixel > localMean + offset means bright mark (foreground)
            // Higher offset = better separation between regions
            double localOffset = 18;  // Increased for better region separation
            var binaryLocal = new Image<Gray, byte>(smoothed.Size);

            // Compute difference: localMean - smoothed (for finding dark regions)
            // or smoothed - localMean (for finding bright regions)
            using var diffDark = new Image<Gray, byte>(smoothed.Size);
            using var diffBright = new Image<Gray, byte>(smoothed.Size);
            CvInvoke.Subtract(localMean, smoothed, diffDark);  // Positive where original is darker than mean
            CvInvoke.Subtract(smoothed, localMean, diffBright);  // Positive where original is brighter than mean

            // Threshold the difference
            if (isLightRing)
            {
                // Light ring: find dark marks (where original < localMean - offset)
                CvInvoke.Threshold(diffDark, binaryLocal, localOffset, 255, ThresholdType.Binary);
            }
            else
            {
                // Dark ring: find bright marks (where original > localMean + offset)
                CvInvoke.Threshold(diffBright, binaryLocal, localOffset, 255, ThresholdType.Binary);
            }

            // Apply erosion to break apart connected regions in Local threshold result
            using var erodeKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.Erode(binaryLocal, binaryLocal, erodeKernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            localMean.Dispose();

            // Step 7: Choose best binary result from Otsu, Adaptive, and Local methods
            Image<Gray, byte> binary;
            string binaryMethod;

            // Expected: ~30-50% of ring area should be white (data marks)
            int ringArea = (int)(Math.PI * (outerR * outerR - innerR * innerR));
            double targetRatio = 0.40;

            // Prepare candidates for comparison
            var candidates = new List<(Image<Gray, byte> img, string name, double ratio)>();

            if (isLightRing)
            {
                // Light ring: we want dark marks to become WHITE in output
                var binaryOtsuInv = binaryOtsu.Not();
                var binaryAdaptiveInv = binaryAdaptive.Not();

                // Count white pixels in data area for each method
                var maskedOtsu = binaryOtsuInv.Copy(ringMask);
                var maskedAdaptive = binaryAdaptiveInv.Copy(ringMask);
                var maskedLocal = binaryLocal.Copy(ringMask);

                int whiteOtsu = CvInvoke.CountNonZero(maskedOtsu);
                int whiteAdaptive = CvInvoke.CountNonZero(maskedAdaptive);
                int whiteLocal = CvInvoke.CountNonZero(maskedLocal);

                double ratioOtsu = (double)whiteOtsu / ringArea;
                double ratioAdaptive = (double)whiteAdaptive / ringArea;
                double ratioLocal = (double)whiteLocal / ringArea;

                Log($"    Light ring binary: Otsu={ratioOtsu:P0}, Adaptive={ratioAdaptive:P0}, Local={ratioLocal:P0}");

                candidates.Add((binaryOtsuInv, "OtsuInv", ratioOtsu));
                candidates.Add((binaryAdaptiveInv, "AdaptiveInv", ratioAdaptive));
                candidates.Add((binaryLocal, "Local", ratioLocal));

                maskedOtsu.Dispose();
                maskedAdaptive.Dispose();
                maskedLocal.Dispose();
            }
            else
            {
                // Dark ring: we want bright marks to become WHITE in output
                // NOTE: AdaptiveThreshold with ThresholdType.Binary makes gray background WHITE (wrong!)
                // because threshold = localMean - C, and gray(130) > 130-8=122 → WHITE
                // Solution: Skip binaryAdaptive for dark rings, only use Otsu and Local

                var maskedOtsu = binaryOtsu.Copy(ringMask);
                var maskedLocal = binaryLocal.Copy(ringMask);

                int whiteOtsu = CvInvoke.CountNonZero(maskedOtsu);
                int whiteLocal = CvInvoke.CountNonZero(maskedLocal);

                double ratioOtsu = (double)whiteOtsu / ringArea;
                double ratioLocal = (double)whiteLocal / ringArea;

                Log($"    Dark ring binary: Otsu={ratioOtsu:P0}, Local={ratioLocal:P0} (Adaptive skipped)");

                candidates.Add((binaryOtsu, "Otsu", ratioOtsu));
                candidates.Add((binaryLocal, "Local", ratioLocal));
                // Skip binaryAdaptive for dark rings - it incorrectly makes gray background white
                binaryAdaptive.Dispose();

                maskedOtsu.Dispose();
                maskedLocal.Dispose();
            }

            // IMPROVED: Combine multiple methods (UNION) instead of picking just one
            // This captures marks detected by ANY method, reducing missed detections
            binary = new Image<Gray, byte>(smoothed.Size);
            var usedMethods = new List<string>();

            foreach (var c in candidates)
            {
                // Include methods with ratio between 5% and 60%
                // Too low = noise, too high = merged marks
                if (c.ratio >= 0.05 && c.ratio <= 0.60)
                {
                    // OR operation - union of all valid methods
                    CvInvoke.BitwiseOr(binary, c.img, binary);
                    usedMethods.Add($"{c.name}({c.ratio:P0})");
                }
                c.img.Dispose();
            }

            binaryMethod = usedMethods.Count > 0 ? string.Join("+", usedMethods) : "None";
            Log($"    Combined methods: {binaryMethod}");

            // Step 8: Morphological cleanup (like HALCON fill_up + select_shape)
            // Small kernel for noise removal (OPEN)
            var kernelSmall = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Open, kernelSmall,
                new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            // Larger kernel for hole filling (CLOSE) - like HALCON fill_up
            var kernelLarge = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                new System.Drawing.Size(7, 7), new Point(-1, -1));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Close, kernelLarge,
                new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));

            // Apply ring mask
            var maskedBinaryTemp = binary.Copy(ringMask);

            // Step 9: Area filtering - remove small noise regions (like HALCON select_shape Area >= 300)
            // Reduced from 500 to 300 to keep smaller valid marks
            int minArea = 300;
            var foreground = new Image<Gray, byte>(binary.Size);
            using var contoursFilter = new VectorOfVectorOfPoint();
            using var hierarchyFilter = new Mat();
            CvInvoke.FindContours(maskedBinaryTemp.Clone(), contoursFilter, hierarchyFilter,
                RetrType.External, ChainApproxMethod.ChainApproxSimple);

            int keptCount = 0;
            int removedCount = 0;
            for (int i = 0; i < contoursFilter.Size; i++)
            {
                double area = CvInvoke.ContourArea(contoursFilter[i]);
                if (area >= minArea)
                {
                    CvInvoke.DrawContours(foreground, contoursFilter, i, new MCvScalar(255), -1);
                    keptCount++;
                }
                else
                {
                    removedCount++;
                }
            }

            int finalWhite = CvInvoke.CountNonZero(foreground);
            Log($"  Foreground: method={binaryMethod}, whitePixels={finalWhite}, " +
                $"contours: kept={keptCount}, removed={removedCount} (minArea={minArea})");

            // Refine outer radius using HALCON-style metrology with edge detection
            // Pass both gray image (for edge detection) and binary mask (for validation)
            float refinedOuterR = RefineOuterRadiusMetrology(ringImage, foreground, region.Center, region.OuterRadius, region.InnerRadius);
            // Always apply refinement - the function already clamps to reasonable range
            Log($"  Refined outerR: {region.OuterRadius:F0} -> {refinedOuterR:F0}");
            region.OuterRadius = refinedOuterR;

            // If least squares fitting found a better center, update region.Center
            if (_lastFitCenter != PointF.Empty && _lastFitRadius > 0)
            {
                double centerShift = Math.Sqrt(Math.Pow(_lastFitCenter.X - region.Center.X, 2) +
                                               Math.Pow(_lastFitCenter.Y - region.Center.Y, 2));
                if (centerShift > 5)  // Only update if shift is significant (> 5 pixels)
                {
                    Log($"  Center corrected: ({region.Center.X:F0},{region.Center.Y:F0}) -> ({_lastFitCenter.X:F0},{_lastFitCenter.Y:F0}), shift={centerShift:F1}");
                    region.Center = _lastFitCenter;
                }
            }

            // Cleanup - binary is already a clone, candidates are disposed above
            // binaryOtsu, binaryAdaptive, binaryLocal were already disposed via candidates loop
            binary.Dispose();
            ringMask.Dispose();
            smoothed.Dispose();

            return foreground;
        }

        /// <summary>
        /// Refine outer radius using HALCON-style metrology with edge detection
        /// Reference: add_metrology_object_circle_measure from HALCON documentation
        ///
        /// Key differences from previous approach:
        /// 1. Samples at regular angular intervals around the FULL circle (not just where data marks exist)
        /// 2. Uses 1D gradient edge detection perpendicular to the circle (like HALCON measure regions)
        /// 3. Finds edges on BOTH sides of data marks, giving evenly distributed fit points
        /// </summary>
        private float RefineOuterRadiusMetrology(Image<Gray, byte> grayImage, Image<Gray, byte> foreground,
            PointF center, float currentOuterR, float innerR)
        {
            int cx = (int)center.X;
            int cy = (int)center.Y;

            // HALCON-style: Sample at regular angular intervals around the FULL circle
            // 120 points = every 3 degrees (like HALCON's default metrology density)
            int numAngles = 120;
            var edgePoints = new List<(Point pt, double angle, float radius)>();
            _lastFitPoints.Clear();

            // Store center for later verification
            _lastFitCenter = new PointF(cx, cy);
            Log($"  [Metrology] Using center=({cx},{cy}), currentOuterR={currentOuterR:F1}");

            // Define the scan range around the expected outer radius
            // MeasureLength1 in HALCON = half-width of measure region perpendicular to boundary
            float measureLength = currentOuterR * 0.15f;  // Scan ±15% of outer radius
            float scanStart = currentOuterR - measureLength;
            float scanEnd = currentOuterR + measureLength;

            // Don't scan inside the data region
            scanStart = Math.Max(scanStart, innerR * 1.1f);

            // Make sure we don't scan outside image bounds
            float maxAllowedR = Math.Min(
                Math.Min(cx, grayImage.Width - cx),
                Math.Min(cy, grayImage.Height - cy)
            ) - 5;
            scanEnd = Math.Min(scanEnd, maxAllowedR);

            // Step 1: For each angle, find the OUTERMOST data mark boundary
            // Only sample where binary mask has white pixels (data marks) - not in empty "air"
            for (int a = 0; a < numAngles; a++)
            {
                double angle = a * 2 * Math.PI / numAngles;
                float cosA = (float)Math.Cos(angle);
                float sinA = (float)Math.Sin(angle);

                // Find the outermost white pixel in binary mask at this angle
                float outermostWhiteR = 0;
                bool foundWhite = false;

                for (float r = scanStart; r <= scanEnd; r += 1.0f)
                {
                    int x = (int)(cx + r * cosA);
                    int y = (int)(cy + r * sinA);

                    if (x < 0 || x >= foreground.Width || y < 0 || y >= foreground.Height)
                        break;

                    if (foreground.Data[y, x, 0] > 128)  // White pixel = data mark
                    {
                        outermostWhiteR = r;
                        foundWhite = true;
                    }
                }

                // Only record edge point if we found a data mark at this angle
                if (foundWhite && outermostWhiteR > scanStart)
                {
                    int epx = (int)(cx + outermostWhiteR * cosA);
                    int epy = (int)(cy + outermostWhiteR * sinA);
                    edgePoints.Add((new Point(epx, epy), angle, outermostWhiteR));
                }
            }

            Log($"  [Metrology] Found {edgePoints.Count}/{numAngles} edge points using gradient detection");

            if (edgePoints.Count < 30)  // Need at least 25% coverage
            {
                Log($"  [Metrology] Not enough edge points, falling back to mask-based method");
                return RefineOuterRadiusFromMaskFallback(foreground, center, currentOuterR, innerR);
            }

            // Step 2: Store all edge points for visualization
            foreach (var (pt, angle, radius) in edgePoints)
            {
                _lastFitPoints.Add(pt);
            }

            // Step 3: Use PERCENTILE-based radius calculation (simpler and more robust than ellipse fitting)
            // Ellipse fitting doesn't work well with sparse/partial data
            var radii = edgePoints.Select(e => e.radius).OrderBy(r => r).ToList();

            float minR = radii.First();
            float maxR = radii.Last();
            float medianR = radii[radii.Count / 2];
            float p90R = radii[(int)(radii.Count * 0.90)];  // 90th percentile
            float p95R = radii[(int)(radii.Count * 0.95)];  // 95th percentile
            float meanR = radii.Average();

            // Use 90th percentile as the outer radius
            // This is robust to outliers while still capturing most data marks
            float finalRadius = p90R;

            // Update stored fit center and radius
            // Keep original center (don't change it based on sparse data)
            _lastFitCenter = center;
            _lastFitRadius = finalRadius;

            Log($"  [Metrology] Radius stats: min={minR:F1}, median={medianR:F1}, mean={meanR:F1}, " +
                $"p90={p90R:F1}, p95={p95R:F1}, max={maxR:F1}");
            Log($"  [Metrology] Using p90 radius: {finalRadius:F1} (center unchanged at {cx},{cy})");

            // Add small margin (2px) to ensure we don't cut into data marks
            return finalRadius + 2;
        }

        /// <summary>
        /// Fallback method: Refine outer radius by scanning binary mask
        /// Used when edge detection doesn't find enough points
        /// </summary>
        private float RefineOuterRadiusFromMaskFallback(Image<Gray, byte> foreground, PointF center, float currentOuterR, float innerR)
        {
            int cx = (int)center.X;
            int cy = (int)center.Y;
            int numAngles = 360;
            var radiusData = new List<(float radius, double angle)>();

            float minDataR = innerR * 1.1f;
            float maxScanR = currentOuterR * 1.25f;

            float maxAllowedR = Math.Min(
                Math.Min(cx, foreground.Width - cx),
                Math.Min(cy, foreground.Height - cy)
            ) - 5;
            maxScanR = Math.Min(maxScanR, maxAllowedR);

            for (int a = 0; a < numAngles; a++)
            {
                double angle = a * 2 * Math.PI / numAngles;
                float cosA = (float)Math.Cos(angle);
                float sinA = (float)Math.Sin(angle);

                float maxR = 0;
                bool foundDataMark = false;

                for (float r = minDataR; r < maxScanR; r += 1.5f)
                {
                    int x = (int)(cx + r * cosA);
                    int y = (int)(cy + r * sinA);

                    if (x < 0 || x >= foreground.Width || y < 0 || y >= foreground.Height)
                        break;

                    if (foreground.Data[y, x, 0] > 128)
                    {
                        maxR = r;
                        foundDataMark = true;
                    }
                }

                if (foundDataMark && maxR > minDataR)
                {
                    radiusData.Add((maxR, angle));
                    float px = cx + maxR * cosA;
                    float py = cy + maxR * sinA;
                    _lastFitPoints.Add(new Point((int)px, (int)py));
                }
            }

            if (radiusData.Count < 20)
                return currentOuterR;

            // Use robust max-based radius
            var radii = radiusData.Select(d => d.radius).ToList();
            radii.Sort();
            float maxRadius = radii.Max();

            Log($"  [Metrology Fallback] Found {radiusData.Count} points, maxR={maxRadius:F1}");
            return maxRadius + 2;
        }

        // Store the fitted center and radius for debug visualization
        private float _lastFitRadius = 0;

        /// <summary>
        /// Fit a circle to a set of points using least squares (Kasa method)
        /// Returns (centerX, centerY, radius) or null if fitting fails
        /// </summary>
        private (double cx, double cy, double r)? FitCircleToPoints(List<Point> points)
        {
            if (points.Count < 10)
                return null;

            // Kasa method for circle fitting
            // Minimizes algebraic distance: Σ(xi² + yi² - 2a*xi - 2b*yi - c)²
            // where center = (a, b) and c = r² - a² - b²

            double sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0;
            double sumXY = 0, sumX3 = 0, sumY3 = 0, sumX2Y = 0, sumXY2 = 0;
            int n = points.Count;

            foreach (var pt in points)
            {
                double x = pt.X;
                double y = pt.Y;
                double x2 = x * x;
                double y2 = y * y;

                sumX += x;
                sumY += y;
                sumX2 += x2;
                sumY2 += y2;
                sumXY += x * y;
                sumX3 += x2 * x;
                sumY3 += y2 * y;
                sumX2Y += x2 * y;
                sumXY2 += x * y2;
            }

            // Solve the normal equations using Cramer's rule
            double A = n * sumX2 - sumX * sumX;
            double B = n * sumXY - sumX * sumY;
            double C = n * sumY2 - sumY * sumY;
            double D = 0.5 * (n * (sumX3 + sumXY2) - sumX * (sumX2 + sumY2));
            double E = 0.5 * (n * (sumX2Y + sumY3) - sumY * (sumX2 + sumY2));

            double denom = A * C - B * B;
            if (Math.Abs(denom) < 1e-10)
                return null;  // Points are collinear

            double cx = (D * C - B * E) / denom;
            double cy = (A * E - B * D) / denom;

            // Calculate radius as mean distance from center to points
            double sumR = 0;
            foreach (var pt in points)
            {
                double r = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                sumR += r;
            }
            double radius = sumR / n;

            // Validate result
            if (double.IsNaN(cx) || double.IsNaN(cy) || double.IsNaN(radius))
                return null;
            if (radius < 50 || radius > 1000)  // Sanity check
                return null;

            return (cx, cy, radius);
        }

        // Store fit points for debug visualization
        private List<Point> _lastFitPoints = new List<Point>();
        private PointF _lastFitCenter = PointF.Empty;  // Center used when calculating fit points
        public IReadOnlyList<Point> LastFitPoints => _lastFitPoints;

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
        private PointF _lastArrowMatchCenter = PointF.Empty;  // Arrow center position for CenterRadius calculation

        // Debug image references for drawing arrow after arrowTip calculation
        private Image<Bgr, byte> _debugImgForArrow = null;
        private int _debugImgCx = 0;
        private int _debugImgCy = 0;
        private float _debugImgOuterR = 0;

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
            _lastArrowContour2 = null;
            _lastIsYShapeArrow = false;
            _lastArrowTip = PointF.Empty;
            _lastArrowMatchCenter = PointF.Empty;

#if WINDOWS
            // Method -1 (PRIMARY): Shape-Based Matching using real Y-arrow template
            if (_shapeMatcher != null && _shapeMatcher.TemplateCount > 0)
            {
                Log($"  [ShapeMatcher] Shape-based arrow detection...");
                var sw = System.Diagnostics.Stopwatch.StartNew();

                var shapeResult = _shapeMatcher.FindArrowInRing(
                    foreground,
                    region.Center,
                    region.InnerRadius * 0.5f,  // Search from 50% of inner radius
                    region.OuterRadius,
                    threshold: 0.5f,
                    classId: SHAPE_MATCHER_CLASS_ID);

                sw.Stop();

                if (shapeResult.IsFound && shapeResult.Score >= 0.5f)
                {
                    _lastMatchType = "ShapeMatcher";
                    _lastArrowMatchCenter = shapeResult.Center;
                    Log($"  [ShapeMatcher] Template match at {shapeResult.Angle:F1}° (score={shapeResult.Score:F3}) in {sw.ElapsedMilliseconds}ms");

                    float cx = region.Center.X;
                    float cy = region.Center.Y;
                    double finalAngle = shapeResult.Angle;

                    // Load template image and get its contour
                    VectorOfPoint templateContour = null;
                    var templatePath = Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                        "ArrowDetectionTest", "arrow_template_active.png");

                    if (File.Exists(templatePath))
                    {
                        using var templateImg = new Image<Gray, byte>(templatePath);
                        using var templateBinary = new Image<Gray, byte>(templateImg.Size);
                        CvInvoke.Threshold(templateImg, templateBinary, 127, 255, ThresholdType.Binary);

                        using var templateContours = new VectorOfVectorOfPoint();
                        using var hierarchy = new Mat();
                        CvInvoke.FindContours(templateBinary, templateContours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                        // Find largest contour in template
                        double maxArea = 0;
                        int maxIdx = -1;
                        for (int i = 0; i < templateContours.Size; i++)
                        {
                            double area = CvInvoke.ContourArea(templateContours[i]);
                            if (area > maxArea) { maxArea = area; maxIdx = i; }
                        }

                        if (maxIdx >= 0)
                        {
                            // Transform template contour: scale, rotate, translate
                            var srcPoints = templateContours[maxIdx].ToArray();
                            var templateCenterX = templateImg.Width / 2.0;
                            var templateCenterY = templateImg.Height / 2.0;
                            double angleRad = shapeResult.TemplateAngle * Math.PI / 180.0;
                            double scale = shapeResult.Scale;  // Use matched scale

                            var transformedPoints = new Point[srcPoints.Length];
                            for (int i = 0; i < srcPoints.Length; i++)
                            {
                                // Translate to origin
                                double px = srcPoints[i].X - templateCenterX;
                                double py = srcPoints[i].Y - templateCenterY;
                                // Scale
                                px *= scale;
                                py *= scale;
                                // Rotate (OpenCV style: x' = cos*x + sin*y, y' = -sin*x + cos*y)
                                double rx = px * Math.Cos(angleRad) + py * Math.Sin(angleRad);
                                double ry = -px * Math.Sin(angleRad) + py * Math.Cos(angleRad);
                                // Translate to match position
                                transformedPoints[i] = new Point(
                                    (int)(rx + shapeResult.Center.X),
                                    (int)(ry + shapeResult.Center.Y));
                            }
                            templateContour = new VectorOfPoint(transformedPoints);
                        }
                    }

                    // Calculate green line angle from templateContour (ALWAYS, not just in debug)
                    Point apex = Point.Empty;
                    Point baseMid = Point.Empty;
                    if (templateContour != null)
                    {
                        var pts = templateContour.ToArray();

                        // Find apex = closest to ring center
                        double minDist = double.MaxValue;
                        foreach (var pt in pts)
                        {
                            double d = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                            if (d < minDist) { minDist = d; apex = pt; }
                        }

                        // Find base midpoint = average of furthest points from apex
                        var byDist = pts.OrderByDescending(pt =>
                            Math.Sqrt(Math.Pow(pt.X - apex.X, 2) + Math.Pow(pt.Y - apex.Y, 2))).ToList();
                        int baseCount = Math.Max(5, pts.Length / 10);
                        double bx = byDist.Take(baseCount).Average(p => p.X);
                        double by = byDist.Take(baseCount).Average(p => p.Y);
                        baseMid = new Point((int)bx, (int)by);

                        // Calculate and store green line angle (apex → baseMid)
                        double greenLineAngle = Math.Atan2(baseMid.Y - apex.Y, baseMid.X - apex.X) * 180.0 / Math.PI;
                        if (greenLineAngle < 0) greenLineAngle += 360;
                        _lastGreenLineAngle = greenLineAngle;
                        Log($"  [ShapeMatcher] Green line angle: {greenLineAngle:F1}° (apex→baseMid)");
                    }

                    // DEBUG: Save visualization with TEMPLATE contour overlay
                    if (!string.IsNullOrEmpty(DebugOutputDir))
                    {
                        try
                        {
                            using var debugImg = original.Convert<Bgr, byte>();

                            // Draw TEMPLATE contour (transformed) - YELLOW
                            if (templateContour != null)
                            {
                                using var overlay = debugImg.Clone();
                                using var contourArray = new VectorOfVectorOfPoint(templateContour);
                                CvInvoke.DrawContours(overlay, contourArray, 0, new MCvScalar(0, 255, 255), -1); // Yellow fill
                                CvInvoke.AddWeighted(debugImg, 0.6, overlay, 0.4, 0, debugImg);
                                CvInvoke.DrawContours(debugImg, contourArray, 0, new MCvScalar(0, 255, 255), 2); // Yellow outline

                                // Draw GREEN axis line: apex → base midpoint (using pre-calculated values)
                                if (apex != Point.Empty && baseMid != Point.Empty)
                                {
                                    CvInvoke.Line(debugImg, apex, baseMid, new MCvScalar(0, 255, 0), 3);
                                    // Draw apex (cyan) and base midpoint (green) dots
                                    CvInvoke.Circle(debugImg, apex, 8, new MCvScalar(255, 255, 0), -1);
                                    CvInvoke.Circle(debugImg, baseMid, 6, new MCvScalar(0, 255, 0), -1);
                                }
                            }

                            // Draw match center - MAGENTA dot
                            CvInvoke.Circle(debugImg, new Point((int)shapeResult.Center.X, (int)shapeResult.Center.Y),
                                10, new MCvScalar(255, 0, 255), -1);

                            // Draw ring center - RED dot
                            CvInvoke.Circle(debugImg, new Point((int)cx, (int)cy), 5, new MCvScalar(0, 0, 255), -1);

                            CvInvoke.PutText(debugImg, $"Angle={shapeResult.TemplateAngle:F0}, Scale={shapeResult.Scale:F2}, score={shapeResult.Score:F3}",
                                new Point(10, 30), FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);
                            var debugPath = Path.Combine(DebugOutputDir, $"shapematcher_debug_{DateTime.Now:HHmmss}.png");
                            debugImg.Save(debugPath);
                            Log($"  [ShapeMatcher] Debug image saved: {debugPath}");

                            // === Save rotated image (arrow pointing to 3 o'clock / 0°) ===
                            if (_lastGreenLineAngle.HasValue)
                            {
                                double rotationAngle = _lastGreenLineAngle.Value;

                                // Create rotation matrix centered on ring center
                                var ringCenter = new PointF((float)cx, (float)cy);
                                using var rotMat = new Mat();
                                CvInvoke.GetRotationMatrix2D(ringCenter, rotationAngle, 1.0, rotMat);

                                // Apply rotation
                                using var rotatedImg = new Mat();
                                CvInvoke.WarpAffine(debugImg, rotatedImg, rotMat, debugImg.Size,
                                    Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0, 0, 0));

                                // Add text showing original angle and rotation
                                CvInvoke.PutText(rotatedImg, $"Rotated to 3 o'clock (green line was at {rotationAngle:F1} deg)",
                                    new Point(10, 60), FontFace.HersheySimplex, 0.7, new MCvScalar(0, 255, 255), 2);

                                // Draw reference line at 0° (3 o'clock direction)
                                int refLength = (int)(region.OuterRadius * 0.8);
                                Point refEnd = new Point((int)(cx + refLength), (int)cy);
                                CvInvoke.Line(rotatedImg, new Point((int)cx, (int)cy), refEnd,
                                    new MCvScalar(0, 255, 255), 2); // Yellow reference line

                                var rotatedPath = Path.Combine(DebugOutputDir, $"shapematcher_rotated_{DateTime.Now:HHmmss}.png");
                                rotatedImg.Save(rotatedPath);
                                Log($"  [ShapeMatcher] Rotated image saved: {rotatedPath}");
                            }
                        }
                        catch (Exception ex) { Log($"  [ShapeMatcher] Debug error: {ex.Message}"); }
                    }

                    // Set LastTemplateMatchResult for shape matcher (so result.GreenLineAngle gets set)
                    _lastMatchType = "ShapeMatcher";
                    LastTemplateMatchResult = new ArrowMatchResult
                    {
                        IsFound = true,
                        Score = shapeResult.Score,
                        Angle = (float)finalAngle,
                        Center = shapeResult.Center,
                        MatchedContour = templateContour
                    };

                    return finalAngle;
                }
                else
                {
                    Log($"  [ShapeMatcher] No match (score={shapeResult.Score:F3}, error={shapeResult.ErrorMessage}) in {sw.ElapsedMilliseconds}ms");
                }
            }
#endif

            // Method 0 (BACKUP): Multi-angle Rotated Template Matching
            // 對 24 個旋轉角度的 Y-arrow template 做 matchTemplate，找到最高分數
            Log($"  [Method 0] Multi-angle Rotated Template Matching...");
            var rotatedResult = FindArrowByRotatedTemplateMatch(foreground, region);
            if (rotatedResult.found && rotatedResult.score >= 0.5)
            {
                _lastMatchType = "RotatedTemplate";
                _lastArrowMatchCenter = rotatedResult.matchCenter;
                Log($"  RotatedTemplate found arrow at: {rotatedResult.angle:F1}° (score={rotatedResult.score:F3})");
                return rotatedResult.angle;
            }
            else
            {
                Log($"  RotatedTemplate failed: score={rotatedResult.score:F3} (need >= 0.5), found={rotatedResult.found}");
            }

            // Method 1 (BACKUP): Hu Moments + matchShapes
            Log($"  [Method 1] Hu Moments matchShapes...");
            var huResult = FindArrowByHuMoments(foreground, region);
            if (huResult.found && huResult.score < 0.08)
            {
                _lastMatchType = "HuMoments";
                _lastArrowContour = huResult.contour;
                _lastArrowTip = huResult.tip;
                Log($"  HuMoments found arrow at: {huResult.angle:F1}° (combinedScore={huResult.score:F3})");
                return huResult.angle;
            }
            else
            {
                Log($"  HuMoments failed: score={huResult.score:F3} (need < 0.08), found={huResult.found}");
            }

            // Method 2 (BACKUP): Template matching using user-created arrow templates
            Log($"  [Method 1] Template matching (ring type: {(_lastRingIsLight ? "LIGHT" : "DARK")})...");
            var templateResult = FindArrowByTemplateMatching(foreground, region);
            if (templateResult.angle.HasValue && templateResult.matchResult?.Score >= 0.5)
            {
                double templateAngle = templateResult.angle.Value;

                // Check: find the HuMoments candidate closest to the template angle
                // If close enough AND at outer part of ring (where Y-arrow should be), use the HuMoments angle
                // This handles cases where template matching finds approximately right area but HuMoments has exact position
                if (_huMomentsCandidates.Count > 0)
                {
                    // Find candidate closest to template angle that's at OUTER part of ring
                    // Y-arrow should be at 0.75-0.90R, data marks can be anywhere
                    (double angle, double score, VectorOfPoint contour, PointF tip) closestCandidate = default;
                    double minAngleDiff = double.MaxValue;
                    double closestDistRatio = 0;

                    float cx = region.Center.X;
                    float cy = region.Center.Y;
                    float R = region.OuterRadius;

                    foreach (var candidate in _huMomentsCandidates)
                    {
                        // Calculate CENTROID distance ratio for this candidate
                        // (Y-arrow centroid is at 0.75-0.85R, data marks are at 0.55-0.70R)
                        var moments = CvInvoke.Moments(candidate.contour);
                        double contourCentroidX = moments.M10 / moments.M00;
                        double contourCentroidY = moments.M01 / moments.M00;
                        double centroidDist = Math.Sqrt(Math.Pow(contourCentroidX - cx, 2) + Math.Pow(contourCentroidY - cy, 2));
                        double distRatio = centroidDist / R;

                        double diff = Math.Abs(templateAngle - candidate.angle);
                        if (diff > 180) diff = 360 - diff;
                        if (diff < minAngleDiff)
                        {
                            minAngleDiff = diff;
                            closestCandidate = candidate;
                            closestDistRatio = distRatio;
                        }
                    }

                    Log($"  [Verify] Template={templateAngle:F1}°, closest HuMoments={closestCandidate.angle:F1}° (diff={minAngleDiff:F1}°, score={closestCandidate.score:F3}, distR={closestDistRatio:F2})");

                    // If closest candidate is within 30° of template angle AND at outer part of ring (>= 0.70R)
                    // use the HuMoments angle - it's likely the actual Y-arrow
                    // If candidate is at inner part (< 0.70R), it's likely a data mark, not Y-arrow
                    if (minAngleDiff <= 30 && closestCandidate.contour != null && closestDistRatio >= 0.70)
                    {
                        _lastMatchType = $"HuMoments (template verified)";
                        _lastArrowContour = closestCandidate.contour;
                        _lastArrowTip = closestCandidate.tip;
                        Log($"  Using HuMoments angle {closestCandidate.angle:F1}° (verified template {templateAngle:F1}°, diff={minAngleDiff:F1}°, at outer ring)");
                        return closestCandidate.angle;
                    }
                    else if (minAngleDiff <= 30 && closestDistRatio < 0.70)
                    {
                        Log($"  [Verify] HuMoments at {closestCandidate.angle:F1}° is at inner ring ({closestDistRatio:F2}R < 0.70R), using template instead");
                    }
                    // If no suitable HuMoments contour, just use template
                    // (template is generally reliable when it finds a match)
                }

                _lastMatchType = $"Template ({templateResult.templateType})";
                Log($"  Template match found arrow at: {templateAngle:F1}° (score={templateResult.matchResult.Score:F3})");
                return templateAngle;
            }
            else
            {
                LastTemplateMatchError = templateResult.error ?? "Score too low";
                Log($"  Template match failed: {LastTemplateMatchError}");

                // If template failed but HuMoments found something, use HuMoments as fallback
                if (huResult.found && huResult.score < 0.7)
                {
                    _lastMatchType = "HuMoments (template fallback)";
                    _lastArrowContour = huResult.contour;
                    _lastArrowTip = huResult.tip;
                    Log($"  Using HuMoments angle {huResult.angle:F1}° as fallback (template failed)");
                    return huResult.angle;
                }
            }

            // Method 2 (FALLBACK): Find arrow by analyzing EDGE contours on ORIGINAL image
            // Arrow is at outer edge, has distinctive triangular/Y shape with LOW solidity
            Log($"  [Method 2] Edge contour analysis on original image...");
            double contourAngle = FindArrowByEdgeContours(original, region);
            if (contourAngle != double.MinValue)
            {
                _lastMatchType = "EdgeContour";
                Log($"  Edge contour analysis found arrow at: {contourAngle:F1}°");
                return contourAngle;
            }

            // Method 3 (FALLBACK): Use triangle TIP points if available (from segmentation)
            // Note: TrianglePoints from segmentation is often unreliable
            if (region.TrianglePoints.Count >= 1)
            {
                var arrowTip = region.TrianglePoints
                    .OrderByDescending(pt =>
                        Math.Sqrt(Math.Pow(pt.X - region.Center.X, 2) + Math.Pow(pt.Y - region.Center.Y, 2)))
                    .First();

                double dist = Math.Sqrt(Math.Pow(arrowTip.X - region.Center.X, 2) + Math.Pow(arrowTip.Y - region.Center.Y, 2));
                // Image coords: Y-down, angles clockwise from right (0°=right, 90°=down)
                double angle = Math.Atan2(arrowTip.Y - region.Center.Y, arrowTip.X - region.Center.X);
                double angleDeg = angle * 180 / Math.PI;

                _lastMatchType = "TriangleTip";
                Log($"  [Method 3 fallback] Arrow TIP at ({arrowTip.X:F1}, {arrowTip.Y:F1}), dist={dist:F1}, angle={angleDeg:F1}°");
                return angleDeg;
            }

            // Method 4: Intensity peak detection as last resort
            _lastMatchType = "IntensityPeak";
            Log($"  [Method 4] Using intensity peak detection as last resort...");
            return FindRotationByIntensityPeak(foreground, region);
        }

        // ============================================================
        // Multi-angle Template Matching (方案A: 暴力旋轉搜尋)
        // ============================================================

        // Cached rotated templates (generated once)
        private static Image<Gray, byte>[] _rotatedTemplates = null;
        private static int[] _rotatedTemplateAngles = null;
        private const int ROTATION_STEP = 15;  // 每 15° 一個 template
        private const int TEMPLATE_SIZE = 80;  // Template 大小 (pixels)

        /// <summary>
        /// 重置旋轉 template 快取（當 template 設計變更時需要呼叫）
        /// </summary>
        public static void ResetRotatedTemplates()
        {
            if (_rotatedTemplates != null)
            {
                foreach (var t in _rotatedTemplates)
                    t?.Dispose();
                _rotatedTemplates = null;
            }
            _rotatedTemplateAngles = null;
        }

        /// <summary>
        /// 建立 Y-arrow 的二值化 template 圖片
        /// 根據實際前處理圖片中的 Y-arrow 形狀設計：
        /// - 三個粗分支呈 Y 形
        /// - Tip 指向圓心方向
        /// - 兩個分支向外張開
        /// </summary>
        private static Image<Gray, byte> CreateYArrowTemplateImage()
        {
            // 建立 80x80 的 template
            int size = 80;
            var template = new Image<Gray, byte>(size, size);
            template.SetZero();

            int cx = size / 2;  // 40
            int cy = size / 2;  // 40

            // Y-arrow 形狀：三個粗分支
            // Tip 指向下方（圓心方向），兩個分支向上張開

            // 分支1：Tip（向下，指向圓心）
            Point[] tip = new Point[]
            {
                new Point(cx - 6, cy),          // 左上
                new Point(cx + 6, cy),          // 右上
                new Point(cx + 4, cy + 30),     // 右下
                new Point(cx - 4, cy + 30),     // 左下
            };

            // 分支2：左上分支
            Point[] leftBranch = new Point[]
            {
                new Point(cx - 4, cy - 5),      // 中心右
                new Point(cx - 8, cy + 5),      // 中心左
                new Point(5, 5),                // 外端左
                new Point(12, 0),               // 外端右
            };

            // 分支3：右上分支
            Point[] rightBranch = new Point[]
            {
                new Point(cx + 4, cy - 5),      // 中心左
                new Point(cx + 8, cy + 5),      // 中心右
                new Point(size - 12, 0),        // 外端左
                new Point(size - 5, 5),         // 外端右
            };

            // 中心連接區域（讓三個分支連接起來）
            Point[] center = new Point[]
            {
                new Point(cx - 10, cy - 5),
                new Point(cx + 10, cy - 5),
                new Point(cx + 10, cy + 10),
                new Point(cx - 10, cy + 10),
            };

            // 繪製所有部分
            using (var tipContour = new VectorOfPoint(tip))
            using (var leftContour = new VectorOfPoint(leftBranch))
            using (var rightContour = new VectorOfPoint(rightBranch))
            using (var centerContour = new VectorOfPoint(center))
            {
                CvInvoke.FillPoly(template, centerContour, new MCvScalar(255));
                CvInvoke.FillPoly(template, tipContour, new MCvScalar(255));
                CvInvoke.FillPoly(template, leftContour, new MCvScalar(255));
                CvInvoke.FillPoly(template, rightContour, new MCvScalar(255));
            }

            return template;
        }

        /// <summary>
        /// 產生所有旋轉角度的 templates
        /// </summary>
        private static void GenerateRotatedTemplates()
        {
            if (_rotatedTemplates != null) return;

            var baseTemplate = CreateYArrowTemplateImage();
            int numAngles = 360 / ROTATION_STEP;  // 24 個角度

            _rotatedTemplates = new Image<Gray, byte>[numAngles];
            _rotatedTemplateAngles = new int[numAngles];

            int cx = baseTemplate.Width / 2;
            int cy = baseTemplate.Height / 2;

            // 計算旋轉後需要的畫布大小（確保不會裁切）
            int diagonal = (int)Math.Ceiling(Math.Sqrt(baseTemplate.Width * baseTemplate.Width + baseTemplate.Height * baseTemplate.Height));

            for (int i = 0; i < numAngles; i++)
            {
                int angle = i * ROTATION_STEP;
                _rotatedTemplateAngles[i] = angle;

                // 建立旋轉矩陣
                var rotMat = new Mat();
                CvInvoke.GetRotationMatrix2D(new PointF(cx, cy), -angle, 1.0, rotMat);  // 負角度因為圖像座標系

                // 旋轉 template
                var rotated = new Image<Gray, byte>(baseTemplate.Size);
                CvInvoke.WarpAffine(baseTemplate, rotated, rotMat, baseTemplate.Size,
                    Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0));

                _rotatedTemplates[i] = rotated;
            }

            baseTemplate.Dispose();
        }

        /// <summary>
        /// 使用骨架化 (Skeletonization/Thinning) 找到 Y-arrow 的真實軸線
        /// </summary>
        private (double angle, PointF tip, PointF axisEnd)? GetSkeletonAxis(
            VectorOfPoint contour, int imageWidth, int imageHeight, float ringCx, float ringCy)
        {
            try
            {
                // 1. Create a binary mask from the contour
                using var mask = new Image<Gray, byte>(imageWidth, imageHeight);
                mask.SetZero();
                using var contourArray = new VectorOfVectorOfPoint(contour);
                CvInvoke.DrawContours(mask, contourArray, 0, new MCvScalar(255), -1);

                // Get bounding rect + margin for ROI
                var rect = CvInvoke.BoundingRectangle(contour);
                int margin = 10;
                int x = Math.Max(0, rect.X - margin);
                int y = Math.Max(0, rect.Y - margin);
                int w = Math.Min(rect.Width + margin * 2, imageWidth - x);
                int h = Math.Min(rect.Height + margin * 2, imageHeight - y);
                var roi = new Rectangle(x, y, w, h);

                using var roiMask = new Mat(mask.Mat, roi);

                // 2. Apply morphological thinning (Zhang-Suen algorithm)
                using var skeleton = new Mat();
                CvInvoke.Threshold(roiMask, skeleton, 127, 255, ThresholdType.Binary);

                // Iterative thinning using morphological operations
                using var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new System.Drawing.Size(3, 3), new Point(-1, -1));
                using var temp = new Mat();
                using var eroded = new Mat();
                using var opened = new Mat();
                using var skelPart = new Mat();
                skeleton.CopyTo(temp);

                var skel = Mat.Zeros(skeleton.Rows, skeleton.Cols, DepthType.Cv8U, 1);

                bool done = false;
                int iterations = 0;
                while (!done && iterations < 100)
                {
                    CvInvoke.MorphologyEx(temp, opened, MorphOp.Open, element, new Point(-1, -1), 1, BorderType.Constant, new MCvScalar(0));
                    CvInvoke.Subtract(temp, opened, skelPart);
                    CvInvoke.BitwiseOr(skel, skelPart, skel);
                    CvInvoke.Erode(temp, eroded, element, new Point(-1, -1), 1, BorderType.Constant, new MCvScalar(0));
                    eroded.CopyTo(temp);

                    done = CvInvoke.CountNonZero(temp) == 0;
                    iterations++;
                }

                // 3. Find skeleton points
                var skelPoints = new List<Point>();
                using var skelImg = skel.ToImage<Gray, byte>();
                for (int py = 0; py < skelImg.Height; py++)
                {
                    for (int px = 0; px < skelImg.Width; px++)
                    {
                        if (skelImg.Data[py, px, 0] > 0)
                        {
                            // Convert back to original image coordinates
                            skelPoints.Add(new Point(px + x, py + y));
                        }
                    }
                }

                if (skelPoints.Count < 5)
                {
                    Log($"  [Skeleton] Too few points: {skelPoints.Count}");
                    skel.Dispose();
                    return null;
                }

                // 4. Find the tip = skeleton point closest to ring center
                Point tipPoint = skelPoints[0];
                double minDist = double.MaxValue;
                foreach (var pt in skelPoints)
                {
                    double d = Math.Sqrt(Math.Pow(pt.X - ringCx, 2) + Math.Pow(pt.Y - ringCy, 2));
                    if (d < minDist)
                    {
                        minDist = d;
                        tipPoint = pt;
                    }
                }

                // 5. For Y-arrow: use skeleton CENTROID for axis direction
                // The centroid represents the center of mass, which is along the main stem
                double sumX = 0, sumY = 0;
                foreach (var pt in skelPoints)
                {
                    sumX += pt.X;
                    sumY += pt.Y;
                }
                var skelCentroid = new PointF((float)(sumX / skelPoints.Count), (float)(sumY / skelPoints.Count));

                // 6. Calculate axis angle: tip → centroid (outward direction, away from ring center)
                double angle = Math.Atan2(skelCentroid.Y - tipPoint.Y, skelCentroid.X - tipPoint.X) * 180.0 / Math.PI;
                if (angle < 0) angle += 360;

                Log($"  [Skeleton] Found axis: tip=({tipPoint.X},{tipPoint.Y}), centroid=({skelCentroid.X:F0},{skelCentroid.Y:F0}), angle={angle:F1}°, skelPoints={skelPoints.Count}");

                skel.Dispose();
                return (angle, new PointF(tipPoint.X, tipPoint.Y), skelCentroid);
            }
            catch (Exception ex)
            {
                Log($"  [Skeleton] Error: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 使用多角度 matchTemplate 找到 Y-arrow
        /// </summary>
        private (bool found, double angle, double score, PointF matchCenter) FindArrowByRotatedTemplateMatch(
            Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // 確保 templates 已生成
            GenerateRotatedTemplates();

            float ringCx = region.Center.X;
            float ringCy = region.Center.Y;
            float outerR = region.OuterRadius;

            // 建立搜尋區域 (只在環形區域內搜尋)
            int margin = 20;
            int roiX = Math.Max(0, (int)(ringCx - outerR - margin));
            int roiY = Math.Max(0, (int)(ringCy - outerR - margin));
            int roiW = Math.Min((int)(outerR * 2 + margin * 2), foreground.Width - roiX);
            int roiH = Math.Min((int)(outerR * 2 + margin * 2), foreground.Height - roiY);

            if (roiW < TEMPLATE_SIZE || roiH < TEMPLATE_SIZE)
                return (false, 0, 0, PointF.Empty);

            // 提取 ROI
            foreground.ROI = new Rectangle(roiX, roiY, roiW, roiH);
            var searchRegion = foreground.Clone();
            foreground.ROI = Rectangle.Empty;

            double bestScore = 0;
            int bestAngle = 0;
            Point bestLocation = Point.Empty;
            int bestTemplateIdx = -1;

            // 建立環形 mask，只在外環區域 (0.55R - 0.95R) 搜尋
            var ringMask = new Image<Gray, byte>(searchRegion.Size);
            ringMask.SetZero();
            float localCx = ringCx - roiX;  // ROI 內的圓心座標
            float localCy = ringCy - roiY;
            float innerSearchR = outerR * 0.55f;
            float outerSearchR = outerR * 0.95f;
            CvInvoke.Circle(ringMask, new Point((int)localCx, (int)localCy), (int)outerSearchR, new MCvScalar(255), -1);
            CvInvoke.Circle(ringMask, new Point((int)localCx, (int)localCy), (int)innerSearchR, new MCvScalar(0), -1);

            // 對每個旋轉角度的 template 做 matchTemplate
            for (int i = 0; i < _rotatedTemplates.Length; i++)
            {
                var template = _rotatedTemplates[i];

                // 確保 template 小於搜尋區域
                if (template.Width >= searchRegion.Width || template.Height >= searchRegion.Height)
                    continue;

                using var result = new Mat();
                CvInvoke.MatchTemplate(searchRegion, template, result, TemplateMatchingType.CcoeffNormed);

                // 將 result 與 ringMask 相乘，只保留環形區域的分數
                // 注意：ringMask 需要縮小到 result 的大小
                int resultW = result.Width;
                int resultH = result.Height;
                using var maskResized = new Image<Gray, byte>(resultW, resultH);
                CvInvoke.Resize(ringMask, maskResized, new System.Drawing.Size(resultW, resultH));

                // 將 mask 外的區域設為 0
                using var resultImg = result.ToImage<Gray, float>();
                for (int y = 0; y < resultH; y++)
                {
                    for (int x = 0; x < resultW; x++)
                    {
                        if (maskResized.Data[y, x, 0] == 0)
                            resultImg.Data[y, x, 0] = 0;
                    }
                }

                double minVal = 0, maxVal = 0;
                Point minLoc = new Point(), maxLoc = new Point();
                CvInvoke.MinMaxLoc(resultImg, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                if (maxVal > bestScore)
                {
                    bestScore = maxVal;
                    bestAngle = _rotatedTemplateAngles[i];
                    bestLocation = maxLoc;
                    bestTemplateIdx = i;
                }
            }

            ringMask.Dispose();

            searchRegion.Dispose();

            if (bestScore < 0.3)  // 最低門檻
            {
                Log($"    [RotatedTemplate] No good match, bestScore={bestScore:F3}");
                return (false, 0, 0, PointF.Empty);
            }

            // 計算匹配中心在原始圖片的位置
            var matchedTemplate = _rotatedTemplates[bestTemplateIdx];
            float matchCenterX = roiX + bestLocation.X + matchedTemplate.Width / 2f;
            float matchCenterY = roiY + bestLocation.Y + matchedTemplate.Height / 2f;

            // 計算從環心到匹配位置的角度（用於驗證位置是否在環形區域）
            double angleFromCenter = Math.Atan2(matchCenterY - ringCy, matchCenterX - ringCx) * 180.0 / Math.PI;
            if (angleFromCenter < 0) angleFromCenter += 360;

            // 計算匹配位置到環心的距離比例（用於驗證）
            double distFromCenter = Math.Sqrt(Math.Pow(matchCenterX - ringCx, 2) + Math.Pow(matchCenterY - ringCy, 2));
            double distRatio = distFromCenter / outerR;

            // 關鍵：templateAngle 是 Y-arrow 的旋轉方向
            // 在我們的設計中，0° template 的 tip 指向下方
            // 所以實際的 arrow 方向 = templateAngle + 調整值
            // Y-arrow 的 tip 指向圓心，所以 arrow 方向 = 匹配位置相對於圓心的角度
            // 因此使用 angleFromCenter 是正確的，但要確保匹配位置在合理範圍內

            Log($"    [RotatedTemplate] Best match: templateAngle={bestAngle}°, score={bestScore:F3}, " +
                $"matchCenter=({matchCenterX:F0},{matchCenterY:F0}), angleFromCenter={angleFromCenter:F1}°, " +
                $"distRatio={distRatio:F2}, time={sw.ElapsedMilliseconds}ms");

            // 驗證匹配位置在環形區域內 (0.5R - 1.0R)
            if (distRatio < 0.5 || distRatio > 1.1)
            {
                Log($"    [RotatedTemplate] Match position outside ring area (distRatio={distRatio:F2}), rejected");
                return (false, 0, 0, PointF.Empty);
            }

            // 儲存 debug 圖片
            if (!string.IsNullOrEmpty(DebugOutputDir))
            {
                try
                {
                    string timestamp = DateTime.Now.ToString("HHmmss_fff");
                    _rotatedTemplates[bestTemplateIdx].Save(
                        System.IO.Path.Combine(DebugOutputDir, $"matched_template_{timestamp}.png"));
                }
                catch { }
            }

            return (true, angleFromCenter, bestScore, new PointF(matchCenterX, matchCenterY));
        }

        // Y-arrow template contour for Hu Moments matching
        private static VectorOfPoint _yArrowTemplate = null;

        /// <summary>
        /// Create Y-arrow template contour for Hu Moments matching
        /// Y-arrow shape: tip pointing inward, two branches extending outward
        /// </summary>
        private static VectorOfPoint GetYArrowTemplate()
        {
            if (_yArrowTemplate != null) return _yArrowTemplate;

            // Define Y-arrow shape (tip at origin, branches extending upward)
            // Scale: approximately 50x60 pixels
            Point[] yPoints = new Point[]
            {
                new Point(25, 60),   // Tip (points toward center)
                new Point(15, 40),   // Left branch start
                new Point(0, 0),     // Left branch end
                new Point(15, 15),   // Left inner
                new Point(25, 25),   // Center
                new Point(35, 15),   // Right inner
                new Point(50, 0),    // Right branch end
                new Point(35, 40),   // Right branch start
            };

            _yArrowTemplate = new VectorOfPoint();
            _yArrowTemplate.Push(yPoints);
            return _yArrowTemplate;
        }

        /// <summary>
        /// Find arrow using Hu Moments + matchShapes (HALCON recommended approach)
        /// Rotation-invariant shape matching using 7 Hu Moments
        /// </summary>
        private (bool found, double angle, double score, VectorOfPoint contour, PointF tip) FindArrowByHuMoments(
            Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            float cx = region.Center.X;
            float cy = region.Center.Y;
            float outerR = region.OuterRadius;
            float innerR = region.InnerRadius > 0 ? region.InnerRadius : outerR * 0.45f;

            // Get Y-arrow template
            var template = GetYArrowTemplate();

            // Find all contours in foreground
            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(foreground.Clone(), contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            if (contours.Size == 0)
                return (false, 0, double.MaxValue, null, PointF.Empty);

            // Expected Y-arrow area range (based on ring size)
            double minArea = outerR * outerR * 0.005;  // 0.5% of R²
            double maxArea = outerR * outerR * 0.06;   // 6% of R²

            // Find best matching contour
            double bestScore = double.MaxValue;
            VectorOfPoint bestContour = null;
            PointF bestTip = PointF.Empty;
            double bestAngle = 0;

            var candidates = new List<(int idx, double matchScore, double solidity, double area, PointF centroid, double distRatio)>();

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);

                // Filter by area
                if (area < minArea || area > maxArea) continue;

                // Calculate centroid
                var moments = CvInvoke.Moments(contour);
                if (moments.M00 < 1) continue;
                PointF centroid = new PointF((float)(moments.M10 / moments.M00), (float)(moments.M01 / moments.M00));

                // Filter by position (must be in data ring area)
                double distFromCenter = Math.Sqrt(Math.Pow(centroid.X - cx, 2) + Math.Pow(centroid.Y - cy, 2));
                double distRatio = distFromCenter / outerR;
                if (distRatio < 0.50 || distRatio > 0.95) continue;

                // Calculate solidity
                using var hull = new VectorOfPoint();
                CvInvoke.ConvexHull(contour, hull);
                double hullArea = CvInvoke.ContourArea(hull);
                double solidity = hullArea > 0 ? area / hullArea : 1.0;

                // Use Hu Moments matchShapes to compare with Y-arrow template
                // ContoursMatchType.I1, I2, I3 use different comparison methods
                // Lower score = more similar
                double matchScore = CvInvoke.MatchShapes(contour, template, ContoursMatchType.I1, 0);

                // Check for convexity defects (Y-shape has 1-2 significant defects)
                using var hullIndices = new Mat();
                CvInvoke.ConvexHull(contour, hullIndices, false, false);
                using var defects = new Mat();
                int defectCount = 0;
                double maxDefectDepth = 0;

                if (hullIndices.Rows >= 3)
                {
                    CvInvoke.ConvexityDefects(contour, hullIndices, defects);
                    if (!defects.IsEmpty && defects.Rows > 0)
                    {
                        // defects is N x 1 x 4 int array: [start_idx, end_idx, far_idx, depth]
                        var defectsData = new int[defects.Rows * 4];
                        defects.CopyTo(defectsData);

                        for (int d = 0; d < defects.Rows; d++)
                        {
                            double depth = defectsData[d * 4 + 3] / 256.0;  // depth is in fixed point
                            if (depth > Math.Sqrt(area) * 0.08)  // significant defect (lowered from 0.15)
                            {
                                defectCount++;
                                maxDefectDepth = Math.Max(maxDefectDepth, depth);
                            }
                        }
                    }
                }

                // Calculate angle from center to centroid for logging
                double centroidAngle = Math.Atan2(centroid.Y - cy, centroid.X - cx) * 180.0 / Math.PI;
                if (centroidAngle < 0) centroidAngle += 360;

                // Log all contours that pass area/position filters (for debugging)
                Log($"    [HuMoments] #{i}: area={area:F0}, sol={solidity:F3}, dist={distRatio:F2}R, angle={centroidAngle:F0}°, match={matchScore:F3}, defects={defectCount}");

                // Y-arrow criteria:
                // 1. Solidity 0.35-0.95 (Y-shape has concave regions, but not too hollow)
                //    - Ideal: 0.40-0.65, but allow up to 0.95 with heavy penalty
                // 2. Has at least 1 convexity defect (indicates branching) - soft requirement
                // 3. Good matchShapes score with template
                bool isYShapeCandidate = false;
                string rejectReason = "";

                if (solidity < 0.35)
                {
                    rejectReason = $"solidity {solidity:F3} < 0.35 (too hollow)";
                }
                else if (solidity > 0.97)
                {
                    // Only reject truly rectangular shapes (>0.97)
                    rejectReason = $"solidity {solidity:F3} > 0.97 (definitely rectangular)";
                }
                else if (defectCount == 0 && matchScore > 2.0 && solidity > 0.85)
                {
                    // No defects, poor shape match, AND high solidity - almost certainly not Y-arrow
                    rejectReason = $"no defects, poor match ({matchScore:F3}), high solidity ({solidity:F3})";
                }
                else
                {
                    isYShapeCandidate = true;
                }

                if (!isYShapeCandidate)
                {
                    Log($"    [HuMoments] #{i}: REJECTED ({rejectReason})");
                    continue;
                }

                // Create combined score based on Y-shape characteristics
                // TRUE Y-arrow has: solidity ~0.40-0.55, exactly 1-2 defects
                // But some images may have higher solidity due to lighting/threshold
                // Use soft penalties instead of hard rejection

                // Solidity score: optimal at 0.50, penalize deviation heavily above 0.80
                double solidityScore;
                if (solidity <= 0.65)
                    solidityScore = Math.Abs(solidity - 0.50) * 1.5;  // small penalty for ideal range
                else if (solidity <= 0.80)
                    solidityScore = 0.2 + (solidity - 0.65) * 2;  // moderate penalty
                else
                    solidityScore = 0.5 + (solidity - 0.80) * 3;  // heavy penalty for >0.80, but not rejected

                // Defect score: optimal at 1 defect, penalize 0 or >2
                double defectScore;
                if (defectCount == 1)
                    defectScore = 0;  // best
                else if (defectCount == 2)
                    defectScore = 0.1;  // good
                else if (defectCount == 0)
                    defectScore = 0.25;  // no defects is suspicious but possible
                else
                    defectScore = 0.2 + 0.05 * (defectCount - 2);  // too many defects

                // Combined: prioritize solidity and defects, then matchScore
                // Y-shape characteristics are more reliable than Hu Moments for this specific case
                double combinedScore = solidityScore * 0.5 + defectScore * 0.3 + Math.Min(matchScore, 5) * 0.02;

                Log($"    [HuMoments] #{i}: solidityScore={solidityScore:F3}, defectScore={defectScore:F3}, combinedScore={combinedScore:F3}");

                candidates.Add((i, combinedScore, solidity, area, centroid, distRatio));
            }

            // Sort ALL candidates by solidity (lowest first)
            // Y-arrow should have the lowest solidity among all marks (not limited to outer ring)
            // The Y-arrow centroid can be at 0.55-0.85R depending on shape
            candidates = candidates.OrderBy(c => c.solidity).ToList();

            Log($"    [HuMoments] Selection: {candidates.Count} candidates (sorted by solidity, lowest first)");

            // Clear and populate _huMomentsCandidates for conflict resolution
            _huMomentsCandidates.Clear();
            foreach (var cand in candidates)
            {
                var contour = contours[cand.idx];
                var points = contour.ToArray();

                // Find tip (closest point to center)
                PointF tip = points[0];
                double minDist = double.MaxValue;
                foreach (var p in points)
                {
                    double dist = Math.Sqrt(Math.Pow(p.X - cx, 2) + Math.Pow(p.Y - cy, 2));
                    if (dist < minDist)
                    {
                        minDist = dist;
                        tip = p;
                    }
                }

                // Calculate angle
                double angle = Math.Atan2(tip.Y - cy, tip.X - cx) * 180.0 / Math.PI;
                if (angle < 0) angle += 360;

                // Copy contour
                var contourCopy = new VectorOfPoint();
                contourCopy.Push(points);

                _huMomentsCandidates.Add((angle, cand.matchScore, contourCopy, tip));
            }

            // Select best candidate
            foreach (var cand in candidates)
            {
                var contour = contours[cand.idx];

                // Find Tip: point closest to ring center
                var points = contour.ToArray();
                PointF tip = points[0];
                double minDist = double.MaxValue;

                foreach (var p in points)
                {
                    double dist = Math.Sqrt(Math.Pow(p.X - cx, 2) + Math.Pow(p.Y - cy, 2));
                    if (dist < minDist)
                    {
                        minDist = dist;
                        tip = p;
                    }
                }

                // Calculate angle from center to Tip
                double angle = Math.Atan2(tip.Y - cy, tip.X - cx) * 180.0 / Math.PI;
                if (angle < 0) angle += 360;

                // Copy the contour for later use
                var contourCopy = new VectorOfPoint();
                contourCopy.Push(points);

                Log($"    [HuMoments] Best: idx={cand.idx}, combinedScore={cand.matchScore:F3}, sol={cand.solidity:F3}, tip=({tip.X:F0},{tip.Y:F0}), angle={angle:F1}°");

                return (true, angle, cand.matchScore, contourCopy, tip);
            }

            return (false, 0, double.MaxValue, null, PointF.Empty);
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

            // Save preprocessing debug images
            if (!string.IsNullOrEmpty(DebugOutputDir))
            {
                try
                {
                    string timestamp = DateTime.Now.ToString("HHmmss_fff");
                    roiImage.Save(Path.Combine(DebugOutputDir, $"preprocess_1_gray_{timestamp}.png"));
                    binaryROI.Save(Path.Combine(DebugOutputDir, $"preprocess_2_binary_{timestamp}.png"));
                    binaryROIInv.Save(Path.Combine(DebugOutputDir, $"preprocess_3_binaryInv_{timestamp}.png"));
                    Log($"  [DEBUG] Saved preprocessing images: preprocess_*_{timestamp}.png");
                }
                catch (Exception ex)
                {
                    Log($"  [DEBUG] Failed to save preprocessing images: {ex.Message}");
                }
            }

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

            // Save template images once
            if (!string.IsNullOrEmpty(DebugOutputDir))
            {
                try
                {
                    string timestamp = DateTime.Now.ToString("HHmmss_fff");
                    if (_darkTemplateMatcher?.IsLoaded == true)
                    {
                        var darkTemplate = GetBinaryTemplate(_darkTemplateMatcher);
                        darkTemplate?.Save(Path.Combine(DebugOutputDir, $"template_dark_{timestamp}.png"));
                    }
                    if (_lightTemplateMatcher?.IsLoaded == true)
                    {
                        var lightTemplate = GetBinaryTemplate(_lightTemplateMatcher);
                        lightTemplate?.Save(Path.Combine(DebugOutputDir, $"template_light_{timestamp}.png"));
                    }
                    Log($"  [DEBUG] Saved template images");
                }
                catch { }
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
        /// Verify if a position in the foreground image contains a Y-shape structure.
        /// Y-shape characteristics: low solidity (< 0.75), has branch-like structure
        /// </summary>
        /// <param name="foreground">Binary foreground mask</param>
        /// <param name="center">Position to verify</param>
        /// <param name="roiSize">Size of ROI to extract</param>
        /// <param name="ringCenter">Center of the ring (for direction analysis)</param>
        /// <param name="outerRadius">Outer radius of ring (for area validation)</param>
        /// <returns>(isYShape, solidity, contourArea)</returns>
        private (bool isYShape, double solidity, double area) VerifyYShapeAtPosition(
            Image<Gray, byte> foreground, PointF center, int roiSize, PointF ringCenter, float outerRadius = 0)
        {
            try
            {
                // Extract ROI around the match position
                int halfSize = roiSize / 2;
                int x = Math.Max(0, (int)center.X - halfSize);
                int y = Math.Max(0, (int)center.Y - halfSize);
                int w = Math.Min(roiSize, foreground.Width - x);
                int h = Math.Min(roiSize, foreground.Height - y);

                if (w < roiSize / 2 || h < roiSize / 2)
                    return (false, 1.0, 0);

                foreground.ROI = new Rectangle(x, y, w, h);
                var roi = foreground.Clone();
                foreground.ROI = Rectangle.Empty;

                // Find contours in ROI
                using var contours = new VectorOfVectorOfPoint();
                using var hierarchy = new Mat();
                CvInvoke.FindContours(roi, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                if (contours.Size == 0)
                    return (false, 1.0, 0);

                // Find the largest contour (should be the Y-shape)
                double maxArea = 0;
                int maxIdx = -1;
                for (int i = 0; i < contours.Size; i++)
                {
                    double area = CvInvoke.ContourArea(contours[i]);
                    if (area > maxArea)
                    {
                        maxArea = area;
                        maxIdx = i;
                    }
                }

                if (maxIdx < 0 || maxArea < 100)
                    return (false, 1.0, maxArea);

                var contour = contours[maxIdx];

                // === AREA RANGE CHECK ===
                // Y-arrow area should be 0.5% - 5% of ring area
                // For R=335: minArea = 560, maxArea = 5610
                // This allows for slightly larger Y-arrows while rejecting merged marks (>6000)
                double minExpectedArea = outerRadius > 0 ? outerRadius * outerRadius * 0.005 : 500;
                double maxExpectedArea = outerRadius > 0 ? outerRadius * outerRadius * 0.05 : 10000;
                bool hasValidArea = maxArea >= minExpectedArea && maxArea <= maxExpectedArea;

                if (!hasValidArea)
                {
                    Log($"    [YShapeVerify] REJECTED - area {maxArea:F0} outside range [{minExpectedArea:F0}, {maxExpectedArea:F0}]");
                    return (false, 1.0, maxArea);
                }

                // Calculate solidity = area / convex hull area
                using var hull = new VectorOfPoint();
                CvInvoke.ConvexHull(contour, hull);
                double hullArea = CvInvoke.ContourArea(hull);
                double solidity = hullArea > 0 ? maxArea / hullArea : 1.0;

                // Y-shape has LOW solidity (gaps between branches)
                // Typical Y-shape solidity: 0.4 - 0.7
                // Rectangular data marks: 0.85 - 0.98
                // BUT: Some images have Y-arrows with higher solidity due to lighting/threshold
                // Allow up to 0.96 to handle edge cases (rectangular marks are typically >0.97)
                bool hasLowSolidity = solidity < 0.96;

                // Additional check: bounding rect aspect ratio
                var boundRect = CvInvoke.BoundingRectangle(contour);
                double aspectRatio = (double)boundRect.Width / Math.Max(1, boundRect.Height);
                // Y-shape should be roughly square-ish to slightly elongated
                bool hasGoodAspect = aspectRatio > 0.5 && aspectRatio < 2.0;

                // Check convexity defects (Y-shape has significant defects)
                bool hasDefects = false;
                int deepDefectCount = 0;
                double maxDefectDepth = 0;
                if (contour.Size >= 5)
                {
                    using var hullIndices = new VectorOfInt();
                    CvInvoke.ConvexHull(contour, hullIndices);
                    if (hullIndices.Size >= 3)
                    {
                        using var defects = new Mat();
                        CvInvoke.ConvexityDefects(contour, hullIndices, defects);
                        if (!defects.IsEmpty && defects.Rows > 0)
                        {
                            // Count significant defects (depth > threshold)
                            // Y-arrow has 1-2 DEEP defects (the V-shaped gap between branches)
                            var defectData = new int[defects.Rows * 4];
                            defects.CopyTo(defectData);
                            double contourPerimeter = CvInvoke.ArcLength(contour, true);
                            double depthThreshold = contourPerimeter * 0.08;  // 8% of perimeter

                            for (int i = 0; i < defects.Rows; i++)
                            {
                                float depth = defectData[i * 4 + 3] / 256f;
                                if (depth > depthThreshold)
                                {
                                    deepDefectCount++;
                                    if (depth > maxDefectDepth) maxDefectDepth = depth;
                                }
                            }
                            // Y-shape should have exactly 1-2 deep defects (the center gap)
                            // Noise/fragments typically have 0 or many small defects
                            hasDefects = deepDefectCount >= 1 && deepDefectCount <= 3;
                        }
                    }
                }

                // === POLYGON APPROXIMATION CHECK ===
                // Y-arrow simplified to polygon should have 5-10 vertices
                // (three branches = ~6-8 vertices after simplification)
                using var approxCurve = new VectorOfPoint();
                double epsilon = 0.03 * CvInvoke.ArcLength(contour, true);
                CvInvoke.ApproxPolyDP(contour, approxCurve, epsilon, true);
                int vertexCount = approxCurve.Size;
                bool hasValidVertexCount = vertexCount >= 4 && vertexCount <= 12;

                // === Y-ARROW DIRECTION CHECK ===
                // The TIP of Y-arrow (closest point to ring center) should point toward ring center
                // Find the point on contour closest to ring center
                var points = contour.ToArray();
                double minDistToCenter = double.MaxValue;
                PointF closestPoint = center;
                double maxDistToCenter = 0;

                foreach (var pt in points)
                {
                    // Convert ROI coordinates to image coordinates
                    PointF imgPt = new PointF(pt.X + x, pt.Y + y);
                    double dist = Math.Sqrt(Math.Pow(imgPt.X - ringCenter.X, 2) + Math.Pow(imgPt.Y - ringCenter.Y, 2));
                    if (dist < minDistToCenter)
                    {
                        minDistToCenter = dist;
                        closestPoint = imgPt;
                    }
                    if (dist > maxDistToCenter)
                    {
                        maxDistToCenter = dist;
                    }
                }

                // The Y-arrow should span from inner ring (~0.55R) to outer ring (~0.95R)
                // Check if the closest point is toward inner part of ring
                double minDistRatio = minDistToCenter / (outerRadius > 0 ? outerRadius : 300);
                double maxDistRatio = maxDistToCenter / (outerRadius > 0 ? outerRadius : 300);
                bool hasValidDirection = minDistRatio >= 0.45 && minDistRatio <= 0.80 && maxDistRatio >= 0.70 && maxDistRatio <= 1.05;

                // === FINAL Y-SHAPE DETERMINATION ===
                // Must have: low solidity + valid defects + valid vertex count + valid direction
                // Keep strict criteria to avoid false positives
                bool isYShape = hasLowSolidity && hasDefects && hasValidVertexCount && hasValidDirection;

                Log($"    [YShapeVerify] pos=({center.X:F0},{center.Y:F0}), sol={solidity:F3}, area={maxArea:F0}, vtx={vertexCount}, defects={deepDefectCount}, minD={minDistRatio:F2}R, maxD={maxDistRatio:F2}R, isY={isYShape}");

                return (isYShape, solidity, maxArea);
            }
            catch (Exception ex)
            {
                Log($"    [YShapeVerify] Error: {ex.Message}");
                return (false, 1.0, 0);
            }
        }

        /// <summary>
        /// Find arrow using template matching - selects template based on ring type
        /// For LIGHT rings: binary image has WHITE marks, so we invert the LIGHT template
        /// For DARK rings: binary image has WHITE marks, use DARK template directly
        /// Returns: (angle, error, matchResult, templateType)
        /// </summary>
        private (double? angle, string error, ArrowMatchResult matchResult, string templateType) FindArrowByTemplateMatching(Image<Gray, byte> foreground, RingImageSegmentation.RingRegion region)
        {
            // Select template based on ring type
            Image<Gray, byte> activeTemplate = null;
            string templateType = "";

            if (_lastRingIsLight && _lightTemplateMatcher?.IsLoaded == true)
            {
                // LIGHT ring: binary has WHITE marks (dark arrows), invert LIGHT template
                activeTemplate = _lightTemplateMatcher.Template.Clone();
                CvInvoke.BitwiseNot(activeTemplate, activeTemplate);
                templateType = "Light (inverted)";
                Log($"  [TemplateMatch] Using LIGHT (inverted) template ({activeTemplate.Width}x{activeTemplate.Height})");
            }
            else if (!_lastRingIsLight && _darkTemplateMatcher?.IsLoaded == true)
            {
                // DARK ring: use DARK template directly
                activeTemplate = _darkTemplateMatcher.Template.Clone();
                templateType = "Dark";
                Log($"  [TemplateMatch] Using DARK template ({activeTemplate.Width}x{activeTemplate.Height})");
            }
            else if (_darkTemplateMatcher?.IsLoaded == true)
            {
                // Fallback to dark template
                activeTemplate = _darkTemplateMatcher.Template.Clone();
                templateType = "Dark (fallback)";
                Log($"  [TemplateMatch] Using DARK template as fallback ({activeTemplate.Width}x{activeTemplate.Height})");
            }
            else if (_lightTemplateMatcher?.IsLoaded == true)
            {
                // Fallback to light template (inverted)
                activeTemplate = _lightTemplateMatcher.Template.Clone();
                CvInvoke.BitwiseNot(activeTemplate, activeTemplate);
                templateType = "Light (inverted fallback)";
                Log($"  [TemplateMatch] Using LIGHT (inverted) template as fallback ({activeTemplate.Width}x{activeTemplate.Height})");
            }

            if (activeTemplate == null)
            {
                return (null, "No template loaded", null, "");
            }

            try
            {
                // Calculate scale based on ring size (single scale, not multi-scale for speed)
                float outerR = region.OuterRadius;
                float innerR = region.InnerRadius;
                float expectedArrowSize = (outerR - innerR) * 0.7f;
                float scale = expectedArrowSize / activeTemplate.Width;
                int scaledSize = Math.Max(15, (int)(activeTemplate.Width * scale));

                // Pre-scale template once
                using var scaledTemplate = new Image<Gray, byte>(scaledSize, scaledSize);
                CvInvoke.Resize(activeTemplate, scaledTemplate, new System.Drawing.Size(scaledSize, scaledSize));

                // Save preprocessing debug images
                if (!string.IsNullOrEmpty(DebugOutputDir))
                {
                    try
                    {
                        string timestamp = DateTime.Now.ToString("HHmmss_fff");
                        foreground.Save(Path.Combine(DebugOutputDir, $"preprocess_1_foreground_{timestamp}.png"));
                        activeTemplate.Save(Path.Combine(DebugOutputDir, $"preprocess_2_template_{timestamp}.png"));
                        scaledTemplate.Save(Path.Combine(DebugOutputDir, $"preprocess_3_scaledTemplate_{timestamp}.png"));
                        Log($"  [DEBUG] Saved preprocessing: preprocess_*_{timestamp}.png");
                    }
                    catch (Exception ex)
                    {
                        Log($"  [DEBUG] Failed to save preprocessing: {ex.Message}");
                    }
                }

                Log($"  [TemplateMatch] Coarse-to-fine search with Y-shape verification");

                // Store multiple candidates for Y-shape verification
                var candidates = new List<(double score, double rotation, PointF center)>();
                double minScoreThreshold = 0.45;  // Lower threshold to collect more candidates
                float minArrowDist = outerR * 0.55f;
                float maxArrowDist = outerR * 0.85f;

                // === PHASE 1: Coarse search (30° steps = 12 iterations) ===
                var coarseCandidates = new List<(double score, double angle)>();
                for (double angle = 0; angle < 360; angle += 30)
                {
                    var scaledCenter = new PointF(scaledSize / 2f, scaledSize / 2f);
                    using var rotMat = new Mat();
                    CvInvoke.GetRotationMatrix2D(scaledCenter, -angle, 1.0, rotMat);

                    using var rotated = new Image<Gray, byte>(scaledSize, scaledSize);
                    CvInvoke.WarpAffine(scaledTemplate, rotated, rotMat, new System.Drawing.Size(scaledSize, scaledSize));

                    if (rotated.Width >= foreground.Width || rotated.Height >= foreground.Height)
                        continue;

                    using var result = new Mat();
                    CvInvoke.MatchTemplate(foreground, rotated, result, TemplateMatchingType.CcoeffNormed);

                    double minVal = 0, maxVal = 0;
                    Point minLoc = Point.Empty, maxLoc = Point.Empty;
                    CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                    if (maxVal > minScoreThreshold)
                    {
                        float matchX = maxLoc.X + rotated.Width / 2f;
                        float matchY = maxLoc.Y + rotated.Height / 2f;
                        float distFromCenter = (float)Math.Sqrt(
                            Math.Pow(matchX - region.Center.X, 2) +
                            Math.Pow(matchY - region.Center.Y, 2));

                        if (distFromCenter >= minArrowDist && distFromCenter <= maxArrowDist)
                        {
                            candidates.Add((maxVal, angle, new PointF(matchX, matchY)));
                            coarseCandidates.Add((maxVal, angle));
                        }
                    }
                }

                // === PHASE 2: Fine search around top coarse candidates ===
                // Sort coarse candidates and do fine search on top 3
                var topCoarse = coarseCandidates.OrderByDescending(c => c.score).Take(3).ToList();
                foreach (var (coarseScore, baseAngle) in topCoarse)
                {
                    for (double offset = -20; offset <= 20; offset += 5)
                    {
                        if (offset == 0) continue; // Already tested in coarse phase
                        double fineAngle = baseAngle + offset;
                        if (fineAngle < 0) fineAngle += 360;
                        if (fineAngle >= 360) fineAngle -= 360;

                        var scaledCenter = new PointF(scaledSize / 2f, scaledSize / 2f);
                        using var rotMat = new Mat();
                        CvInvoke.GetRotationMatrix2D(scaledCenter, -fineAngle, 1.0, rotMat);

                        using var rotated = new Image<Gray, byte>(scaledSize, scaledSize);
                        CvInvoke.WarpAffine(scaledTemplate, rotated, rotMat, new System.Drawing.Size(scaledSize, scaledSize));

                        if (rotated.Width >= foreground.Width || rotated.Height >= foreground.Height)
                            continue;

                        using var result = new Mat();
                        CvInvoke.MatchTemplate(foreground, rotated, result, TemplateMatchingType.CcoeffNormed);

                        double minVal = 0, maxVal = 0;
                        Point minLoc = Point.Empty, maxLoc = Point.Empty;
                        CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                        if (maxVal > minScoreThreshold)
                        {
                            float matchX = maxLoc.X + rotated.Width / 2f;
                            float matchY = maxLoc.Y + rotated.Height / 2f;
                            float distFromCenter = (float)Math.Sqrt(
                                Math.Pow(matchX - region.Center.X, 2) +
                                Math.Pow(matchY - region.Center.Y, 2));

                            if (distFromCenter >= minArrowDist && distFromCenter <= maxArrowDist)
                            {
                                candidates.Add((maxVal, fineAngle, new PointF(matchX, matchY)));
                            }
                        }
                    }
                }

                Log($"  [TemplateMatch] Found {candidates.Count} candidates, verifying Y-shape...");

                // === PHASE 3: Verify candidates with Y-shape check ===
                // Sort by score (descending) and verify each
                double bestScore = 0;
                double bestRotation = 0;
                PointF bestMatchCenter = PointF.Empty;
                int roiSize = (int)((outerR - innerR) * 1.2);  // ROI size based on ring width

                foreach (var (score, rotation, center) in candidates.OrderByDescending(c => c.score))
                {
                    // Verify Y-shape at this position
                    var (isYShape, solidity, area) = VerifyYShapeAtPosition(foreground, center, roiSize, region.Center, region.OuterRadius);

                    if (isYShape)
                    {
                        // Found valid Y-shape, use this candidate
                        bestScore = score;
                        bestRotation = rotation;
                        bestMatchCenter = center;
                        Log($"  [TemplateMatch] Verified Y-shape at ({center.X:F0},{center.Y:F0}), score={score:F3}, solidity={solidity:F3}");
                        break;
                    }
                    else if (bestScore == 0 && score > 0.7)
                    {
                        // Store high-score candidate as fallback (even without Y-shape verification)
                        bestScore = score;
                        bestRotation = rotation;
                        bestMatchCenter = center;
                        Log($"  [TemplateMatch] High-score fallback at ({center.X:F0},{center.Y:F0}), score={score:F3} (not Y-shape)");
                    }
                }

                // Save debug image if DebugOutputDir is set
                if (!string.IsNullOrEmpty(DebugOutputDir))
                {
                    try
                    {
                        Directory.CreateDirectory(DebugOutputDir);
                        var debugImg = foreground.Convert<Bgr, byte>();
                        int cx = (int)region.Center.X;
                        int cy = (int)region.Center.Y;

                        // Draw ring boundaries using region values
                        CvInvoke.Circle(debugImg, new Point(cx, cy), (int)region.OuterRadius, new MCvScalar(0, 255, 0), 2);
                        CvInvoke.Circle(debugImg, new Point(cx, cy), (int)region.InnerRadius, new MCvScalar(0, 255, 0), 2);

                        // DIAGNOSTIC: Draw circle using LEAST SQUARES fitted center (ORANGE)
                        // This should pass through all fit points correctly
                        if (_lastFitCenter != PointF.Empty && _lastFitRadius > 0)
                        {
                            int fitCx = (int)_lastFitCenter.X;
                            int fitCy = (int)_lastFitCenter.Y;
                            // Draw the least-squares fitted circle (ORANGE) - this should match the fit points
                            CvInvoke.Circle(debugImg, new Point(fitCx, fitCy), (int)(_lastFitRadius + 2), new MCvScalar(0, 165, 255), 3);
                            // Draw the fitted center (ORANGE)
                            CvInvoke.Circle(debugImg, new Point(fitCx, fitCy), 10, new MCvScalar(0, 165, 255), -1);
                            Log($"  [DEBUG] Green circle (original): center=({cx},{cy}), R={region.OuterRadius:F1}");
                            Log($"  [DEBUG] Orange circle (fitted):  center=({fitCx},{fitCy}), R={_lastFitRadius + 2:F1}");
                            double centerShift = Math.Sqrt(Math.Pow(fitCx - cx, 2) + Math.Pow(fitCy - cy, 2));
                            Log($"  [DEBUG] Center shift = {centerShift:F1} pixels");
                        }

                        // Draw fit points used for outer radius refinement (MAGENTA/ORANGE)
                        // Verify center consistency between fit point calculation and debug drawing
                        bool centerMismatch = false;
                        if (_lastFitCenter != PointF.Empty)
                        {
                            double centerDist = Math.Sqrt(Math.Pow(cx - _lastFitCenter.X, 2) + Math.Pow(cy - _lastFitCenter.Y, 2));
                            if (centerDist > 1)
                            {
                                centerMismatch = true;
                                Log($"  [DEBUG] WARNING: Center mismatch! FitCenter=({_lastFitCenter.X:F0},{_lastFitCenter.Y:F0}), DrawCenter=({cx},{cy}), dist={centerDist:F1}");
                            }
                        }

                        // Calculate actual radius of fit points for debugging
                        double fitPointMaxR = 0;
                        foreach (var pt in _lastFitPoints)
                        {
                            double r = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                            if (r > fitPointMaxR) fitPointMaxR = r;
                        }

                        double radiusDiff = Math.Abs(fitPointMaxR - (region.OuterRadius - 2));
                        Log($"  [DEBUG] Drawing {_lastFitPoints.Count} fit points, maxR={fitPointMaxR:F1}, expectedR={region.OuterRadius - 2:F1}, circleR={region.OuterRadius:F1}, diff={radiusDiff:F1}");

                        // DIAGNOSTIC: Draw a CYAN circle at exactly fitPointMaxR radius
                        // This circle SHOULD pass through the outermost orange fit points
                        CvInvoke.Circle(debugImg, new Point(cx, cy), (int)fitPointMaxR, new MCvScalar(255, 255, 0), 1);
                        Log($"  [DEBUG] Cyan circle at maxR={fitPointMaxR:F1} should pass through orange points");

                        // DIAGNOSTIC: Find and mark the furthest fit point
                        Point furthestPt = Point.Empty;
                        double furthestR = 0;
                        foreach (var pt in _lastFitPoints)
                        {
                            double r = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                            if (r > furthestR)
                            {
                                furthestR = r;
                                furthestPt = pt;
                            }
                        }
                        if (furthestPt != Point.Empty)
                        {
                            // Draw line from center to furthest point (MAGENTA)
                            CvInvoke.Line(debugImg, new Point(cx, cy), furthestPt, new MCvScalar(255, 0, 255), 2);
                            // Mark the furthest point with a big circle
                            CvInvoke.Circle(debugImg, furthestPt, 15, new MCvScalar(255, 0, 255), 3);
                            Log($"  [DEBUG] Furthest fit point: ({furthestPt.X},{furthestPt.Y}), distance from center ({cx},{cy}) = {furthestR:F1}");
                            Log($"  [DEBUG] Expected on green circle at R={region.OuterRadius:F1}, actual R={furthestR:F1}, DIFF={furthestR - region.OuterRadius:F1}");
                        }

                        if (radiusDiff > 10)
                        {
                            Log($"  [DEBUG] WARNING: Large radius discrepancy ({radiusDiff:F1}px)! Possible center or coordinate mismatch.");
                        }

                        // Only draw fit points that are near the circle (within 15px of max)
                        // These are the points that "define" the outer circle
                        int nearCircleCount = 0;
                        int farFromCircleCount = 0;
                        foreach (var pt in _lastFitPoints)
                        {
                            double r = Math.Sqrt(Math.Pow(pt.X - cx, 2) + Math.Pow(pt.Y - cy, 2));
                            bool nearCircle = Math.Abs(r - fitPointMaxR) <= 15;

                            if (nearCircle)
                            {
                                // Draw points near circle in ORANGE (these define the outer boundary)
                                CvInvoke.Circle(debugImg, pt, 6, new MCvScalar(0, 165, 255), -1);  // Orange filled
                                CvInvoke.Circle(debugImg, pt, 6, new MCvScalar(0, 0, 0), 1);  // Black outline
                                nearCircleCount++;
                            }
                            else
                            {
                                // Draw points far from circle in smaller YELLOW (for reference)
                                CvInvoke.Circle(debugImg, pt, 3, new MCvScalar(0, 255, 255), -1);  // Yellow filled
                                farFromCircleCount++;
                            }
                        }
                        Log($"  [DEBUG] FitPoints: {nearCircleCount} near circle (orange), {farFromCircleCount} far (yellow)");

                        // If center mismatch, draw the original fit center in MAGENTA
                        if (centerMismatch && _lastFitCenter != PointF.Empty)
                        {
                            CvInvoke.Circle(debugImg, Point.Round(_lastFitCenter), 12, new MCvScalar(255, 0, 255), 3);  // Magenta ring
                            CvInvoke.PutText(debugImg, "FitCenter", new Point((int)_lastFitCenter.X + 15, (int)_lastFitCenter.Y),
                                FontFace.HersheySimplex, 0.5, new MCvScalar(255, 0, 255), 1);
                        }

                        // Draw center point
                        CvInvoke.Circle(debugImg, new Point(cx, cy), 8, new MCvScalar(0, 0, 255), -1);

                        // Draw best match position if found
                        if (bestScore > 0 && bestMatchCenter != PointF.Empty)
                        {
                            int matchX = (int)bestMatchCenter.X;
                            int matchY = (int)bestMatchCenter.Y;

                            // Draw match center (CYAN)
                            CvInvoke.Circle(debugImg, new Point(matchX, matchY), 12, new MCvScalar(255, 255, 0), -1);
                            CvInvoke.Circle(debugImg, new Point(matchX, matchY), 12, new MCvScalar(0, 0, 0), 2);

                            // Draw line from ring center to match center (thin YELLOW line - template center)
                            CvInvoke.Line(debugImg, new Point(cx, cy), new Point(matchX, matchY), new MCvScalar(0, 255, 255), 2);

                            // NOTE: The cyan arrow will be drawn AFTER arrowTip is calculated
                            // Store debug image reference for later use
                            _debugImgForArrow = debugImg;
                            _debugImgCx = cx;
                            _debugImgCy = cy;
                            _debugImgOuterR = region.OuterRadius;

                            // Add text (score and match position - angle will be added later)
                            CvInvoke.PutText(debugImg, $"Score: {bestScore:F3}", new Point(10, 30),
                                FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);
                            CvInvoke.PutText(debugImg, $"Match: ({matchX},{matchY})", new Point(10, 90),
                                FontFace.HersheySimplex, 0.6, new MCvScalar(255, 255, 0), 2);
                        }
                        else
                        {
                            CvInvoke.PutText(debugImg, $"NO MATCH (best={bestScore:F3})", new Point(10, 30),
                                FontFace.HersheySimplex, 0.8, new MCvScalar(0, 0, 255), 2);
                        }

                        string timestamp = DateTime.Now.ToString("HHmmss_fff");
                        string debugPath = Path.Combine(DebugOutputDir, $"arrow_debug_{timestamp}.png");
                        debugImg.Save(debugPath);
                        Log($"  [DEBUG] Saved arrow detection image: {debugPath}");
                        debugImg.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Log($"  [DEBUG] Failed to save debug image: {ex.Message}");
                    }
                }

                if (bestScore >= 0.4)
                {
                    // Find the actual arrow TIP (closest white pixel to ring center)
                    // The template match center is the CENTER of the Y-shape, not the tip
                    PointF arrowTip = bestMatchCenter;  // Default to match center

                    // Search within a region around the match center for the closest white pixel
                    int searchRadius = 60;  // Half of template size
                    float cx = region.Center.X;
                    float cy = region.Center.Y;
                    double minDistToCenter = double.MaxValue;

                    int startX = Math.Max(0, (int)(bestMatchCenter.X - searchRadius));
                    int endX = Math.Min(foreground.Width - 1, (int)(bestMatchCenter.X + searchRadius));
                    int startY = Math.Max(0, (int)(bestMatchCenter.Y - searchRadius));
                    int endY = Math.Min(foreground.Height - 1, (int)(bestMatchCenter.Y + searchRadius));

                    for (int py = startY; py <= endY; py++)
                    {
                        for (int px = startX; px <= endX; px++)
                        {
                            if (foreground.Data[py, px, 0] > 128)  // White pixel (part of arrow)
                            {
                                double distToCenter = Math.Sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
                                if (distToCenter < minDistToCenter)
                                {
                                    minDistToCenter = distToCenter;
                                    arrowTip = new PointF(px, py);
                                }
                            }
                        }
                    }

                    // Calculate angle from ring center to arrow tip
                    double dx = arrowTip.X - region.Center.X;
                    double dy = arrowTip.Y - region.Center.Y;
                    double finalAngle = Math.Atan2(dy, dx) * 180.0 / Math.PI;
                    if (finalAngle < 0) finalAngle += 360;

                    Log($"  [TemplateMatch] Best match: score={bestScore:F3}, rotation={bestRotation:F1}°");
                    Log($"  [TemplateMatch] Match center: ({bestMatchCenter.X:F0},{bestMatchCenter.Y:F0}), Arrow tip: ({arrowTip.X:F0},{arrowTip.Y:F0})");
                    Log($"  [TemplateMatch] Arrow direction: atan2({dy:F0},{dx:F0}) = {finalAngle:F1}°");

                    // Store arrow match center for CenterRadius calculation (HALCON method)
                    _lastArrowMatchCenter = bestMatchCenter;

                    // NOW draw the cyan arrow using the calculated arrowTip (not bestMatchCenter)
                    if (_debugImgForArrow != null)
                    {
                        try
                        {
                            // Draw arrow tip marker (MAGENTA circle)
                            CvInvoke.Circle(_debugImgForArrow, Point.Round(arrowTip), 15, new MCvScalar(255, 0, 255), 3);

                            // Draw extended arrow line from center to arrow tip direction (CYAN)
                            double rad = finalAngle * Math.PI / 180;
                            int arrowEndX = (int)(_debugImgCx + _debugImgOuterR * 1.2 * Math.Cos(rad));
                            int arrowEndY = (int)(_debugImgCy + _debugImgOuterR * 1.2 * Math.Sin(rad));
                            CvInvoke.ArrowedLine(_debugImgForArrow, new Point(_debugImgCx, _debugImgCy),
                                new Point(arrowEndX, arrowEndY), new MCvScalar(255, 255, 0), 4, LineType.AntiAlias);

                            // Add angle text
                            CvInvoke.PutText(_debugImgForArrow, $"Angle: {finalAngle:F1} deg", new Point(10, 60),
                                FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 255), 2);
                            CvInvoke.PutText(_debugImgForArrow, $"Tip: ({arrowTip.X:F0},{arrowTip.Y:F0})", new Point(10, 120),
                                FontFace.HersheySimplex, 0.6, new MCvScalar(255, 0, 255), 2);
                        }
                        catch { /* Ignore debug drawing errors */ }
                        _debugImgForArrow = null;  // Clear reference
                    }

                    var matchResult = new ArrowMatchResult
                    {
                        IsFound = true,
                        Score = bestScore,
                        Angle = finalAngle,
                        Center = arrowTip  // Use arrow TIP instead of match center
                    };

                    LastTemplateMatchResult = matchResult;
                    return (finalAngle, null, matchResult, templateType);
                }

                Log($"  [TemplateMatch] No match found (best score={bestScore:F3} < 0.4)");
                LastTemplateMatchResult = null;
                return (null, $"Template match score too low: {bestScore:F3}", null, templateType);
            }
            finally
            {
                activeTemplate?.Dispose();
            }
        }

        // Store last detected arrow contour for visualization
        private VectorOfPoint _lastArrowContour;
        private VectorOfPoint _lastArrowContour2;  // Second contour for Y-shape
        private bool _lastIsYShapeArrow;  // Flag to indicate Y-shape detection
        private PointF _lastArrowTip;

        // Store all HuMoments candidates for conflict resolution with template matching
        // (angle, score, contour, tip)
        private List<(double angle, double score, VectorOfPoint contour, PointF tip)> _huMomentsCandidates = new();

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
            _lastArrowContour2 = null;
            _lastIsYShapeArrow = false;
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
            float outerR = region.OuterRadius;  // Now correctly set to data ring outer by RingImageSegmentation
            float innerR = region.InnerRadius;  // Now correctly set by radial scan

            Log($"    Ring radii: inner={innerR:F0}, outer={outerR:F0}, ratio={innerR/outerR:F2}");
            Log($"    Expected: inner ~40% of outer, actual={100*innerR/outerR:F0}%");

            // Sanity check: inner should be ~40-42% of outer for ring codes
            if (innerR / outerR < 0.35 || innerR / outerR > 0.50)
            {
                Log($"    WARNING: Inner/outer ratio {innerR/outerR:F2} outside expected range [0.35-0.50]!");
            }

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

            // Determine ring type: light ring has bright background (mean > 150), dark ring has gray background
            bool isLightRingForArrow = meanValue > 150;

            // Calculate threshold based on ring type
            double calculatedThresh;
            ThresholdType threshType;
            if (isLightRingForArrow)
            {
                // Light ring: find dark marks (below mean - k*std)
                calculatedThresh = meanValue - 1.5 * stdDev;
                threshType = ThresholdType.BinaryInv;  // Pixels below threshold → WHITE
            }
            else
            {
                // Dark ring: find bright marks (above mean + k*std)
                calculatedThresh = meanValue + 1.0 * stdDev;
                threshType = ThresholdType.Binary;  // Pixels above threshold → WHITE
            }
            byte optimalThresh = (byte)Math.Max(80, Math.Min(220, calculatedThresh));

            Log($"    Ring-based threshold: {optimalThresh} (mean={meanValue:F0}, std={stdDev:F0}, type={(_lastRingIsLight ? "LIGHT" : "DARK")})");

            // Apply binary threshold based on ring type
            var binaryResult = new Image<Gray, byte>(enhancedImg.Size);
            CvInvoke.Threshold(enhancedImg, binaryResult, optimalThresh, 255, threshType);

            // Apply mask
            var maskedBinary = new Image<Gray, byte>(original.Size);
            CvInvoke.BitwiseAnd(binaryResult, dataMask, maskedBinary);

            // Morphological cleanup - OPEN only (no CLOSE to preserve Y-arrow shape and low solidity)
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new System.Drawing.Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(maskedBinary, maskedBinary, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
            // NO CLOSE operation - preserve arrow Y-shape low solidity (matches Test Program)

            // Collect candidates from preprocessed binary
            // contour2 is for Y-shape (both branches), null for single contour
            var allCandidates = new List<(double score, double solidity, double angle, double area,
                double centroidDistRatio, double tipDistRatio, double elongation, VectorOfPoint contour,
                VectorOfPoint contour2, PointF centroid, PointF basePoint, PointF tipPoint, bool tipPointsOutward, string method)>();

            using var contours = new VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(maskedBinary.Clone(), contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            // Arrow characteristics - reasonable area based on ring size
            double minArea = outerR * outerR * 0.005;  // ~0.5% of ring area
            double maxArea = outerR * outerR * 0.10;   // ~10% of ring area

            // Create arrow template once for shape matching
            var arrowTemplate = CreateArrowTemplate();

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

                // Calculate convexity defects for Y-shape detection
                double maxDefectDepth = 0;
                try
                {
                    using var hullIndices = new Mat();
                    CvInvoke.ConvexHull(contour, hullIndices, false, false);
                    if (hullIndices.Rows >= 4)
                    {
                        using var defects = new Mat();
                        CvInvoke.ConvexityDefects(contour, hullIndices, defects);
                        if (!defects.IsEmpty && defects.Rows > 0)
                        {
                            var defectData = new int[defects.Rows * 4];
                            defects.CopyTo(defectData);
                            for (int d = 0; d < defects.Rows; d++)
                            {
                                double depth = defectData[d * 4 + 3] / 256.0;
                                if (depth > maxDefectDepth) maxDefectDepth = depth;
                            }
                        }
                    }
                }
                catch { /* Ignore defect errors */ }

                double defectRatio = maxDefectDepth / Math.Sqrt(area);

                // Calculate angle using center → basePoint direction
                // basePoint = closest point to ring center = the TIP of the Y-arrow (apex pointing inward)
                // This gives the exact direction to the arrow tip, not the centroid
                // Image coords: 0°=right, 90°=down (clockwise)
                double angle = Math.Atan2(basePoint.Y - cy, basePoint.X - cx) * 180.0 / Math.PI;
                if (angle < 0) angle += 360;

                // === SHAPE MATCHING with Y-arrow template ===
                double shapeMatchScore = CalculateShapeMatchScore(contour, arrowTemplate);

                // Base score from multi-feature analysis
                double baseScore = CalculateArrowScore(solidity, centroidDistRatio, tipDistRatio,
                    area, outerR, tipPointsOutward, elongation);

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
                    double defectBoostScore = Math.Min(0.75, 0.55 + defectRatio * 0.6);
                    score = Math.Max(defectBoostScore, baseScore * 0.6);
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

                // PENALTY for very small contours - likely noise, not real arrow
                double minExpectedArea = outerR * outerR * 0.02;  // ~2% of ring area
                double maxExpectedArea = outerR * outerR * 0.04;  // ~4% of ring area

                if (area < minExpectedArea && shapeMatchScore < 0.7)
                {
                    double areaRatio = area / minExpectedArea;
                    double areaPenalty = 0.25 * (1.0 - areaRatio);
                    if (areaRatio < 0.5)
                        areaPenalty += 0.15;
                    score -= areaPenalty;
                }

                // PENALTY for oversized contours - likely merged data marks, not Y-arrow
                if (area > maxExpectedArea)
                {
                    double oversizeRatio = area / maxExpectedArea;
                    double oversizePenalty = 0.30 * (oversizeRatio - 1.0);
                    if (oversizeRatio > 1.5)
                        oversizePenalty += 0.20;
                    if (hasVeryLowSolidity && oversizeRatio > 1.2)
                        oversizePenalty += 0.15;
                    score -= oversizePenalty;
                }

                Log($"    Contour {i}: area={area:F0}, solidity={solidity:F3}, defect={defectRatio:F2}, centroid={centroidDistRatio:F2}R, tip={tipDistRatio:F2}R, angle={angle:F0}°, shapeMatch={shapeMatchScore:F2}, score={score:F2}");

                if (score > 0.1)
                {
                    var contourCopy = new VectorOfPoint(contour.ToArray());
                    allCandidates.Add((score, solidity, angle, area, centroidDistRatio, tipDistRatio,
                        elongation, contourCopy, null, centroid, basePoint, tipPoint, tipPointsOutward, "Preprocessed"));
                }
            }

            // === Y-SHAPE DETECTION: DISABLED due to high false positive rate ===
            // Adjacent data marks (15° apart) were incorrectly being detected as Y-pairs
            // Rely on single-contour detection with low solidity instead

            // Cleanup
            dataMask.Dispose();
            enhanced.Dispose();
            enhancedImg.Dispose();
            binaryResult.Dispose();
            maskedBinary.Dispose();
            kernel.Dispose();

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
                    _lastArrowContour2 = best.contour2;  // Store second contour for Y-shape
                    _lastIsYShapeArrow = best.contour2 != null;  // Flag if Y-shape
                    _lastArrowTip = best.tipPoint;  // Store TIP point (furthest from center) for visualization

                    Log($"  Arrow found: score={best.score:F2}, angle={best.angle:F1}° -> sector {sector} ({snappedAngle}°), isYShape={_lastIsYShapeArrow}");
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
                // Calculate segment angular boundaries - COUNTER-CLOCKWISE from arrow
                // In image coordinates (Y down), counter-clockwise = decreasing angle
                double angle1 = startRad - (i * segmentRad) + segmentRad;      // startRad - (i-1)*segmentRad
                double angle2 = startRad - ((i + 1) * segmentRad) + segmentRad; // startRad - i*segmentRad
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
                bool parityValid = checkParityNum != "00" && checkParityNum != "11" && CheckParity(data, checkParityNum);
                bool bccValid = ValidateBCC(temp);

                Log($"    Binary: {temp}");
                Log($"    Parity: {checkParityNum} (valid={parityValid}), BCC valid={bccValid}");

                // Accept if EITHER parity OR BCC is valid
                // Note: For more reliable decode, caller should try multiple angles and prefer both valid
                if (!parityValid && !bccValid)
                {
                    return "-1";
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

        /// <summary>
        /// Decode binary string with validation details
        /// Returns (decodedValue, parityValid, bccValid)
        /// </summary>
        private (string decoded, bool parityValid, bool bccValid) DecryptBinaryWithValidation(string temp)
        {
            if (string.IsNullOrEmpty(temp) || temp.Length < 48)
                return ("-1", false, false);

            if (temp == "000000000000000000000000000000000000000000000000")
                return ("0", false, false);

            try
            {
                string data = temp.Substring(4, temp.Length - 6);
                string checkParityNum = temp.Substring(46, 2);

                bool parityValid = checkParityNum != "00" && checkParityNum != "11" && CheckParity(data, checkParityNum);
                bool bccValid = ValidateBCC(temp);

                // Accept if EITHER parity OR BCC is valid
                if (!parityValid && !bccValid)
                    return ("-1", false, false);

                string dueDateBits = temp.Substring(4, 8);
                string machineIdBits = temp.Substring(12, 12);
                string serialNumberBits = temp.Substring(24, 22);

                int dueDateValue = Convert.ToInt32(dueDateBits, 2);
                string dueDate = DecodeDueDate(dueDateValue);
                string machineId = Convert.ToInt64(machineIdBits, 2).ToString().PadLeft(4, '0');
                string serialNumber = Convert.ToInt64(serialNumberBits, 2).ToString().PadLeft(7, '0');

                return ($"{dueDate}{machineId}{serialNumber}", parityValid, bccValid);
            }
            catch
            {
                return ("-1", false, false);
            }
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
        /// OPTIMIZED version - faster than original
        /// </summary>
        public Image<Bgr, byte> CreateVisualization(Image<Bgr, byte> source, RingCodeResult result)
        {
            var visualization = source.Clone();
            int cx = (int)result.Center.X;
            int cy = (int)result.Center.Y;
            float outerR = result.OuterRadius;
            float innerR = result.InnerRadius;
            float middleR = result.MiddleRadius;

            var green = new MCvScalar(0, 255, 0);
            var yellow = new MCvScalar(0, 255, 255);

            // Draw circles (green)
            CvInvoke.Circle(visualization, new Point(cx, cy), (int)outerR, green, 2);
            CvInvoke.Circle(visualization, new Point(cx, cy), (int)middleR, green, 1);
            CvInvoke.Circle(visualization, new Point(cx, cy), (int)innerR, green, 2);

            // Draw center point (green filled)
            CvInvoke.Circle(visualization, new Point(cx, cy), 4, green, -1);

            // Pre-compute angles
            double startAngle = result.RotationAngle * Math.PI / 180;
            double segmentRad = SEGMENT_ANGLE * Math.PI / 180;

            // Draw segment lines (green)
            for (int i = 0; i < SEGMENTS; i++)
            {
                double angle = startAngle - (i * segmentRad) + segmentRad;
                int x1 = (int)(cx + innerR * 0.9 * Math.Cos(angle));
                int y1 = (int)(cy + innerR * 0.9 * Math.Sin(angle));
                int x2 = (int)(cx + outerR * Math.Cos(angle));
                int y2 = (int)(cy + outerR * Math.Sin(angle));
                CvInvoke.Line(visualization, new Point(x1, y1), new Point(x2, y2), green, 1);
            }

            // Draw yellow arrow from center to arrow position
            int arrowX = (int)(cx + outerR * 0.95 * Math.Cos(startAngle));
            int arrowY = (int)(cy + outerR * 0.95 * Math.Sin(startAngle));
            CvInvoke.ArrowedLine(visualization, new Point(cx, cy), new Point(arrowX, arrowY),
                yellow, 3, LineType.AntiAlias, 0, 0.15);

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
        /// Algebraic Least Squares Circle Fitting
        /// Reference: HALCON_TO_EMGUCV_CONVERSION.md Section 6.3.1
        /// </summary>
        private (PointF center, float radius, bool success) FitCircleLeastSquares(Point[] points)
        {
            if (points.Length < 3)
                return (PointF.Empty, 0, false);

            int n = points.Length;

            // Build matrix [x y 1] and vector [x² + y²]
            double sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
            double sumXXX = 0, sumYYY = 0, sumXXY = 0, sumXYY = 0;

            for (int i = 0; i < n; i++)
            {
                double x = points[i].X;
                double y = points[i].Y;
                double xx = x * x;
                double yy = y * y;

                sumX += x;
                sumY += y;
                sumXX += xx;
                sumYY += yy;
                sumXY += x * y;
                sumXXX += xx * x;
                sumYYY += yy * y;
                sumXXY += xx * y;
                sumXYY += x * yy;
            }

            // Solve linear system using Cramer's rule
            // | sumXX  sumXY  sumX | | A |   | sumXXX + sumXYY |
            // | sumXY  sumYY  sumY | | B | = | sumXXY + sumYYY |
            // | sumX   sumY   n    | | C |   | sumXX + sumYY   |

            double[,] matrix = {
                { sumXX, sumXY, sumX },
                { sumXY, sumYY, sumY },
                { sumX,  sumY,  n    }
            };

            double[] rhs = {
                sumXXX + sumXYY,
                sumXXY + sumYYY,
                sumXX + sumYY
            };

            // Gaussian elimination
            double[] solution = SolveLinearSystem3x3(matrix, rhs);
            if (solution == null)
                return (PointF.Empty, 0, false);

            double A = solution[0];
            double B = solution[1];
            double C = solution[2];

            // Convert to circle parameters
            float cx = (float)(A / 2.0);
            float cy = (float)(B / 2.0);
            double rSquared = C + cx * cx + cy * cy;
            if (rSquared <= 0)
                return (PointF.Empty, 0, false);

            float r = (float)Math.Sqrt(rSquared);

            return (new PointF(cx, cy), r, true);
        }

        /// <summary>
        /// Gaussian elimination for 3x3 linear system
        /// </summary>
        private double[] SolveLinearSystem3x3(double[,] matrix, double[] rhs)
        {
            int n = 3;
            double[,] a = (double[,])matrix.Clone();
            double[] b = (double[])rhs.Clone();

            // Forward elimination
            for (int k = 0; k < n - 1; k++)
            {
                // Pivot selection
                int maxRow = k;
                for (int i = k + 1; i < n; i++)
                {
                    if (Math.Abs(a[i, k]) > Math.Abs(a[maxRow, k]))
                        maxRow = i;
                }

                // Row swap
                for (int j = k; j < n; j++)
                {
                    double temp = a[k, j];
                    a[k, j] = a[maxRow, j];
                    a[maxRow, j] = temp;
                }
                double tempB = b[k];
                b[k] = b[maxRow];
                b[maxRow] = tempB;

                // Elimination
                if (Math.Abs(a[k, k]) < 1e-10)
                    return null;  // Singular matrix

                for (int i = k + 1; i < n; i++)
                {
                    double factor = a[i, k] / a[k, k];
                    for (int j = k; j < n; j++)
                        a[i, j] -= factor * a[k, j];
                    b[i] -= factor * b[k];
                }
            }

            // Back substitution
            double[] x = new double[n];
            for (int i = n - 1; i >= 0; i--)
            {
                x[i] = b[i];
                for (int j = i + 1; j < n; j++)
                    x[i] -= a[i, j] * x[j];
                x[i] /= a[i, i];
            }

            return x;
        }

        /// <summary>
        /// Create combined visualization for all decoded rings - OPTIMIZED & HALCON-style
        /// Features: Red overlay on data marks, green grid, yellow arrow, segment numbers
        /// </summary>
        public Image<Bgr, byte> CreateCombinedVisualization(Image<Bgr, byte> source, List<RingCodeResult> results)
        {
            var visualization = source.Clone();

            foreach (var result in results)
            {
                int cx = (int)result.Center.X;
                int cy = (int)result.Center.Y;
                float outerR = result.OuterRadius;
                float innerR = result.InnerRadius;
                float middleR = result.MiddleRadius;

                var green = new MCvScalar(0, 255, 0);
                var yellow = new MCvScalar(0, 255, 255);
                var red = new MCvScalar(0, 0, 255);
                var white = new MCvScalar(255, 255, 255);
                var mainColor = result.IsValid ? green : red;

                // === STEP 1: Draw red semi-transparent overlay on foreground mask (FAST) ===
                if (result.ForegroundMask != null)
                {
                    // Create red overlay image
                    using var redOverlay = new Image<Bgr, byte>(visualization.Size);
                    redOverlay.SetValue(new Bgr(0, 0, 200));  // Dark red

                    // Create 3-channel mask from foreground
                    using var mask3ch = new Image<Bgr, byte>(visualization.Size);
                    CvInvoke.CvtColor(result.ForegroundMask, mask3ch, ColorConversion.Gray2Bgr);

                    // Blend: where mask is white, blend with red
                    using var masked = new Image<Bgr, byte>(visualization.Size);
                    CvInvoke.BitwiseAnd(redOverlay, mask3ch, masked);

                    // Add to visualization (50% blend)
                    CvInvoke.AddWeighted(visualization, 0.7, masked, 0.3, 0, visualization);

                    // Draw contours in red (outline only, no analysis)
                    using var contours = new VectorOfVectorOfPoint();
                    CvInvoke.FindContours(result.ForegroundMask.Clone(), contours, null,
                        RetrType.External, ChainApproxMethod.ChainApproxSimple);
                    CvInvoke.DrawContours(visualization, contours, -1, red, 2);
                }

                // === STEP 2: Draw circles (green) ===
                CvInvoke.Circle(visualization, new Point(cx, cy), (int)outerR, green, 2);
                CvInvoke.Circle(visualization, new Point(cx, cy), (int)middleR, green, 1);
                CvInvoke.Circle(visualization, new Point(cx, cy), (int)innerR, green, 2);

                // Draw center point
                CvInvoke.Circle(visualization, new Point(cx, cy), 5, green, -1);

                // === STEP 3: Draw segment lines and numbers ===
                double startAngle = result.RotationAngle * Math.PI / 180;
                double segmentRad = SEGMENT_ANGLE * Math.PI / 180;

                for (int i = 0; i < SEGMENTS; i++)
                {
                    // Counter-clockwise from arrow
                    double angle = startAngle - (i * segmentRad) + segmentRad;

                    // Segment line
                    int x1 = (int)(cx + innerR * 0.9 * Math.Cos(angle));
                    int y1 = (int)(cy + innerR * 0.9 * Math.Sin(angle));
                    int x2 = (int)(cx + outerR * Math.Cos(angle));
                    int y2 = (int)(cy + outerR * Math.Sin(angle));
                    CvInvoke.Line(visualization, new Point(x1, y1), new Point(x2, y2), green, 1);

                    // Segment number (at outer edge)
                    double midAngle = angle - segmentRad / 2;
                    int numX = (int)(cx + (outerR + 15) * Math.Cos(midAngle));
                    int numY = (int)(cy + (outerR + 15) * Math.Sin(midAngle));
                    CvInvoke.PutText(visualization, i.ToString(), new Point(numX - 5, numY + 4),
                        FontFace.HersheySimplex, 0.35, green, 1);
                }

                // === STEP 4: Draw X marks on filled segments ===
                for (int i = 0; i < SEGMENTS; i++)
                {
                    double midAngle = startAngle - (i * segmentRad) + segmentRad / 2;
                    bool hasInnerBit = result.BinaryString.Length > i * 2 && result.BinaryString[i * 2] == '1';
                    bool hasOuterBit = result.BinaryString.Length > i * 2 + 1 && result.BinaryString[i * 2 + 1] == '1';

                    if (hasInnerBit)
                    {
                        float r = (innerR + middleR) / 2;
                        int xi = (int)(cx + r * Math.Cos(midAngle));
                        int yi = (int)(cy + r * Math.Sin(midAngle));
                        DrawCross(visualization, xi, yi, 4, green);
                    }
                    if (hasOuterBit)
                    {
                        float r = (middleR + outerR) / 2;
                        int xo = (int)(cx + r * Math.Cos(midAngle));
                        int yo = (int)(cy + r * Math.Sin(midAngle));
                        DrawCross(visualization, xo, yo, 4, green);
                    }
                }

                // === STEP 5: Draw yellow arrow from center ===
                {
                    int arrowX = (int)(cx + outerR * 0.9 * Math.Cos(startAngle));
                    int arrowY = (int)(cy + outerR * 0.9 * Math.Sin(startAngle));
                    CvInvoke.ArrowedLine(visualization, new Point(cx, cy), new Point(arrowX, arrowY),
                        yellow, 3, LineType.AntiAlias, 0, 0.15);

                    // Arrow tip circle
                    if (result.ArrowTip != PointF.Empty)
                    {
                        CvInvoke.Circle(visualization, Point.Round(result.ArrowTip), 8, yellow, 2);
                    }

                    // "S0" marker at arrow position
                    CvInvoke.PutText(visualization, "S0", new Point(arrowX + 5, arrowY - 5),
                        FontFace.HersheySimplex, 0.4, yellow, 1);
                }

                // === STEP 6: Draw text information ===
                int textY = (int)(cy - outerR - 25);

                // Decoded result at top (large, green)
                string dataText = result.IsValid ? result.DecodedData : "Invalid";
                CvInvoke.PutText(visualization, dataText, new Point(cx - 80, textY),
                    FontFace.HersheySimplex, 0.7, mainColor, 2);

                // Rotation and arrow angle at bottom left
                int bottomY = (int)(cy + outerR + 20);
                CvInvoke.PutText(visualization, $"Rotation: {result.RotationAngle:F1} deg",
                    new Point(cx - (int)outerR, bottomY),
                    FontFace.HersheySimplex, 0.4, green, 1);
                CvInvoke.PutText(visualization, $"Arrow: {result.RotationAngle:F1} deg",
                    new Point(cx - (int)outerR, bottomY + 18),
                    FontFace.HersheySimplex, 0.4, yellow, 1);

                // Binary strings (inner ring and outer ring separated)
                if (!string.IsNullOrEmpty(result.BinaryString) && result.BinaryString.Length == 48)
                {
                    // Extract inner bits (even indices) and outer bits (odd indices)
                    var innerBits = new System.Text.StringBuilder(24);
                    var outerBits = new System.Text.StringBuilder(24);
                    for (int i = 0; i < 24; i++)
                    {
                        innerBits.Append(result.BinaryString[i * 2]);
                        outerBits.Append(result.BinaryString[i * 2 + 1]);
                    }

                    CvInvoke.PutText(visualization, innerBits.ToString(),
                        new Point(cx - (int)outerR, bottomY + 40),
                        FontFace.HersheySimplex, 0.35, white, 1);
                    CvInvoke.PutText(visualization, outerBits.ToString(),
                        new Point(cx - (int)outerR, bottomY + 55),
                        FontFace.HersheySimplex, 0.35, white, 1);
                }
            }

            return visualization;
        }

        /// <summary>
        /// Create a ROTATED main visualization with arrow pointing right
        /// Similar to HALCON's output - the main image shows the ring rotated so arrow is at 3 o'clock
        /// Includes foreground region overlay for debugging
        /// </summary>
        public Image<Bgr, byte> CreateRotatedMainVisualization(Image<Bgr, byte> source, RingCodeResult result, int outputSize = 400)
        {
            if (result.OuterRadius < 10) return source.Clone();

            try
            {
                // Extract ring region with margin
                int margin = (int)(result.OuterRadius * 1.3);
                int x = Math.Max(0, (int)(result.Center.X - margin));
                int y = Math.Max(0, (int)(result.Center.Y - margin));
                int w = Math.Min(margin * 2, source.Width - x);
                int h = Math.Min(margin * 2, source.Height - y);

                if (w <= 0 || h <= 0) return source.Clone();

                source.ROI = new Rectangle(x, y, w, h);
                var cropped = source.Clone();
                source.ROI = Rectangle.Empty;

                // Use Green Line Angle for rotation (apex → baseMid direction)
                // This is the most accurate representation of arrow direction
                double actualArrowAngle;
                if (result.GreenLineAngle.HasValue)
                {
                    actualArrowAngle = result.GreenLineAngle.Value;
                }
                else if (result.TemplateMatchCenter.HasValue && result.TemplateMatchCenter.Value != PointF.Empty)
                {
                    // Fallback: calculate from TemplateMatchCenter
                    double dx = result.TemplateMatchCenter.Value.X - result.Center.X;
                    double dy = result.TemplateMatchCenter.Value.Y - result.Center.Y;
                    actualArrowAngle = Math.Atan2(dy, dx) * 180.0 / Math.PI;
                    if (actualArrowAngle < 0) actualArrowAngle += 360;
                }
                else
                {
                    actualArrowAngle = result.RotationAngle;
                }

                // Calculate ring center position in CROPPED image coordinates
                float ringCenterInCroppedX = (float)(result.Center.X - x);
                float ringCenterInCroppedY = (float)(result.Center.Y - y);

                // Rotate to make arrow point right (0 degrees)
                // Arrow is at angle A (in image coords where Y is down)
                // GetRotationMatrix2D: positive angle = counter-clockwise in math coords
                // In IMAGE coords (Y-down), positive angle appears CLOCKWISE
                // To rotate arrow from A° to 0°: we need to rotate CW by A° (which is +A in GetRotationMatrix2D)
                // This matches TestArrowDetection logic: rotationAngle = 0 - A, then -rotationAngle = A
                double rotationNeeded = actualArrowAngle;
                using var rotationMat = new Mat();
                CvInvoke.GetRotationMatrix2D(new PointF(ringCenterInCroppedX, ringCenterInCroppedY), rotationNeeded, 1.0, rotationMat);

                using var rotated = new Image<Bgr, byte>(cropped.Size);
                CvInvoke.WarpAffine(cropped, rotated, rotationMat, cropped.Size);

                // Also rotate the foreground mask if available
                Image<Gray, byte> rotatedForeground = null;
                Image<Gray, byte> croppedForeground = null;
                if (result.ForegroundMask != null)
                {
                    // Crop foreground mask to same ROI
                    result.ForegroundMask.ROI = new Rectangle(x, y, w, h);
                    croppedForeground = result.ForegroundMask.Clone();
                    result.ForegroundMask.ROI = Rectangle.Empty;

                    // Rotate foreground mask
                    rotatedForeground = new Image<Gray, byte>(croppedForeground.Size);
                    CvInvoke.WarpAffine(croppedForeground, rotatedForeground, rotationMat, croppedForeground.Size);
                }

                // Resize to output size
                var output = new Image<Bgr, byte>(outputSize, outputSize);
                CvInvoke.Resize(rotated, output, new System.Drawing.Size(outputSize, outputSize));

                // Calculate scale factor for drawing
                float scale = outputSize / (float)(margin * 2);

                // Default center (will be updated by finding inner circle)
                int cx = (int)(ringCenterInCroppedX * scale);
                int cy = (int)(ringCenterInCroppedY * scale);
                float scaledOuterR = result.OuterRadius * scale;
                float scaledInnerR = result.InnerRadius * scale;
                float scaledMiddleR = result.MiddleRadius * scale;

                // Find TRUE center from rotated FOREGROUND MASK using Least Squares Circle Fitting
                // Reference: HALCON_TO_EMGUCV_CONVERSION.md Section 6.3.2
                // Use edge of foreground mask (markerRegion) to find outer circle
                double arrowAngleFromNewCenter = 0;  // Will be updated after finding new center
                Image<Gray, byte> scaledFgForArrow = null;

                if (rotatedForeground != null)
                {
                    // Scale foreground to output size first
                    scaledFgForArrow = new Image<Gray, byte>(outputSize, outputSize);
                    CvInvoke.Resize(rotatedForeground, scaledFgForArrow, new System.Drawing.Size(outputSize, outputSize));

                    // 1. Apply Canny edge detection on foreground mask
                    using var edges = new Image<Gray, byte>(outputSize, outputSize);
                    CvInvoke.Canny(scaledFgForArrow, edges, 50, 150);

                    // 2. Create ring-shaped ROI around expected OUTER radius
                    int expectedOuterR = (int)scaledOuterR;
                    using var ringMask = new Image<Gray, byte>(outputSize, outputSize);
                    ringMask.SetValue(new MCvScalar(0));
                    Point estCenter = new Point(cx, cy);
                    CvInvoke.Circle(ringMask, estCenter, expectedOuterR + 15, new MCvScalar(255), -1);
                    CvInvoke.Circle(ringMask, estCenter, expectedOuterR - 15, new MCvScalar(0), -1);

                    // 3. Get edge points within ring ROI
                    using var maskedEdges = new Image<Gray, byte>(outputSize, outputSize);
                    CvInvoke.BitwiseAnd(edges, ringMask, maskedEdges);

                    // 4. Collect edge points
                    var edgePoints = new List<Point>();
                    byte[,,] edgeData = maskedEdges.Data;
                    for (int py = 0; py < outputSize; py++)
                    {
                        for (int px = 0; px < outputSize; px++)
                        {
                            if (edgeData[py, px, 0] > 0)
                                edgePoints.Add(new Point(px, py));
                        }
                    }

                    // 5. Least Squares Circle Fitting for OUTER circle (if enough points)
                    if (edgePoints.Count >= 20)
                    {
                        var fitResult = FitCircleLeastSquares(edgePoints.ToArray());
                        if (fitResult.success && fitResult.radius > 50 && fitResult.radius < outputSize * 0.6f)
                        {
                            cx = (int)fitResult.center.X;
                            cy = (int)fitResult.center.Y;
                            scaledOuterR = fitResult.radius;
                            Log($"  [Rotated] LeastSquares fit (outer): center=({cx}, {cy}), outerR={scaledOuterR:F1}, points={edgePoints.Count}");
                        }
                        else
                        {
                            Log($"  [Rotated] LeastSquares fit failed or invalid radius={fitResult.radius:F1}, using original center: ({cx}, {cy})");
                        }
                    }
                    else
                    {
                        Log($"  [Rotated] Not enough edge points ({edgePoints.Count}), using original center: ({cx}, {cy})");
                    }

                    // 5b. Fit INNER circle from grayscale image (dark hole boundary)
                    using (var grayScaled = output.Convert<Gray, byte>())
                    {
                        using var innerEdges = new Image<Gray, byte>(outputSize, outputSize);
                        CvInvoke.Canny(grayScaled, innerEdges, 30, 100);

                        // Create ring mask around expected inner radius
                        int expectedInnerR = (int)(scaledOuterR * result.InnerRadius / result.OuterRadius);
                        using var innerRingMask = new Image<Gray, byte>(outputSize, outputSize);
                        innerRingMask.SetValue(new MCvScalar(0));
                        CvInvoke.Circle(innerRingMask, new Point(cx, cy), expectedInnerR + 20, new MCvScalar(255), -1);
                        CvInvoke.Circle(innerRingMask, new Point(cx, cy), Math.Max(10, expectedInnerR - 20), new MCvScalar(0), -1);

                        using var innerMaskedEdges = new Image<Gray, byte>(outputSize, outputSize);
                        CvInvoke.BitwiseAnd(innerEdges, innerRingMask, innerMaskedEdges);

                        var innerEdgePoints = new List<Point>();
                        byte[,,] innerEdgeData = innerMaskedEdges.Data;
                        for (int py = 0; py < outputSize; py++)
                        {
                            for (int px = 0; px < outputSize; px++)
                            {
                                if (innerEdgeData[py, px, 0] > 0)
                                    innerEdgePoints.Add(new Point(px, py));
                            }
                        }

                        if (innerEdgePoints.Count >= 20)
                        {
                            var innerFit = FitCircleLeastSquares(innerEdgePoints.ToArray());
                            if (innerFit.success && innerFit.radius > 20 && innerFit.radius < scaledOuterR * 0.8f)
                            {
                                scaledInnerR = innerFit.radius;
                                // Middle radius = average of inner and outer data track boundaries
                                scaledMiddleR = (scaledInnerR + scaledOuterR) / 2;
                                Log($"  [Rotated] LeastSquares fit (inner): innerR={scaledInnerR:F1}, middleR={scaledMiddleR:F1}, points={innerEdgePoints.Count}");
                            }
                            else
                            {
                                // Fallback to ratio calculation
                                scaledInnerR = scaledOuterR * result.InnerRadius / result.OuterRadius;
                                scaledMiddleR = scaledOuterR * result.MiddleRadius / result.OuterRadius;
                                Log($"  [Rotated] Inner fit failed, using ratio: innerR={scaledInnerR:F1}");
                            }
                        }
                        else
                        {
                            // Fallback to ratio calculation
                            scaledInnerR = scaledOuterR * result.InnerRadius / result.OuterRadius;
                            scaledMiddleR = scaledOuterR * result.MiddleRadius / result.OuterRadius;
                            Log($"  [Rotated] Not enough inner edge points ({innerEdgePoints.Count}), using ratio: innerR={scaledInnerR:F1}");
                        }
                    }

                    // 6. Calculate Y-arrow angle from NEW center
                    // After rotation, the apex is at 0° direction from the ORIGINAL center
                    // Transform this point to find angle from NEW center
                    // Original center in scaled coords: (ringCenterInCroppedX * scale, ringCenterInCroppedY * scale)
                    float origCxScaled = ringCenterInCroppedX * scale;
                    float origCyScaled = ringCenterInCroppedY * scale;

                    // Apex after rotation is at 0° from original center, at innerRadius distance
                    float apexX = origCxScaled + scaledInnerR;  // 0° direction = right
                    float apexY = origCyScaled;

                    // Calculate angle from NEW center to apex
                    arrowAngleFromNewCenter = Math.Atan2(apexY - cy, apexX - cx);
                    Log($"  [Rotated] Y-arrow angle from new center: {arrowAngleFromNewCenter * 180 / Math.PI:F1}° (orig center: {origCxScaled:F0},{origCyScaled:F0}, new: {cx},{cy})");
                }

                // Resize rotated foreground and overlay (OPTIMIZED - no pixel loop)
                if (rotatedForeground != null)
                {
                    var scaledForeground = new Image<Gray, byte>(outputSize, outputSize);
                    CvInvoke.Resize(rotatedForeground, scaledForeground, new System.Drawing.Size(outputSize, outputSize));

                    // Create red overlay and blend using AddWeighted (FAST)
                    using var redOverlay = new Image<Bgr, byte>(outputSize, outputSize);
                    redOverlay.SetValue(new Bgr(0, 0, 200));  // Dark red

                    using var mask3ch = new Image<Bgr, byte>(outputSize, outputSize);
                    CvInvoke.CvtColor(scaledForeground, mask3ch, ColorConversion.Gray2Bgr);

                    using var masked = new Image<Bgr, byte>(outputSize, outputSize);
                    CvInvoke.BitwiseAnd(redOverlay, mask3ch, masked);

                    CvInvoke.AddWeighted(output, 0.7, masked, 0.3, 0, output);

                    scaledForeground.Dispose();
                }

                // Draw circles
                var mainColor = result.IsValid ? new MCvScalar(0, 255, 0) : new MCvScalar(0, 0, 255);
                CvInvoke.Circle(output, new Point(cx, cy), (int)scaledOuterR, mainColor, 2);
                CvInvoke.Circle(output, new Point(cx, cy), (int)scaledMiddleR, mainColor, 1);
                CvInvoke.Circle(output, new Point(cx, cy), (int)scaledInnerR, mainColor, 2);

                // Draw center point (red for visibility)
                CvInvoke.Circle(output, new Point(cx, cy), 5, new MCvScalar(0, 0, 255), -1);

                // Grid starts at Y-arrow position (calculated from NEW center)
                // arrowAngleFromNewCenter is the angle from the fitted center to the Y-arrow apex
                double gridOffsetRad = arrowAngleFromNewCenter;

                // Draw 24 segment lines with numbers (starting from arrow at 0° + offset)
                var lineColor = new MCvScalar(0, 255, 0);
                for (int i = 0; i < SEGMENTS; i++)
                {
                    double angle = (i * SEGMENT_ANGLE - SEGMENT_ANGLE) * Math.PI / 180 + gridOffsetRad;
                    int x1 = (int)(cx + scaledInnerR * 0.9 * Math.Cos(angle));
                    int y1 = (int)(cy + scaledInnerR * 0.9 * Math.Sin(angle));
                    int x2 = (int)(cx + scaledOuterR * Math.Cos(angle));
                    int y2 = (int)(cy + scaledOuterR * Math.Sin(angle));
                    CvInvoke.Line(output, new Point(x1, y1), new Point(x2, y2), lineColor, 1);

                    // Segment number
                    double midAngle = (i * SEGMENT_ANGLE - SEGMENT_ANGLE / 2) * Math.PI / 180 + gridOffsetRad;
                    int numX = (int)(cx + (scaledOuterR + 12) * Math.Cos(midAngle));
                    int numY = (int)(cy + (scaledOuterR + 12) * Math.Sin(midAngle));
                    CvInvoke.PutText(output, i.ToString(), new Point(numX - 5, numY + 4),
                        FontFace.HersheySimplex, 0.3, lineColor, 1);
                }

                // Draw X marks on filled segments (bit=1)
                // Use same angle direction as grid lines (clockwise from arrow position)
                if (!string.IsNullOrEmpty(result.BinaryString) && result.BinaryString.Length == 48)
                {
                    for (int i = 0; i < SEGMENTS; i++)
                    {
                        // Same direction as segment numbers: clockwise from gridOffset
                        double midAngle = (i * SEGMENT_ANGLE - SEGMENT_ANGLE / 2) * Math.PI / 180 + gridOffsetRad;
                        bool hasInnerBit = result.BinaryString[i * 2] == '1';
                        bool hasOuterBit = result.BinaryString[i * 2 + 1] == '1';

                        if (hasInnerBit)
                        {
                            float r = (scaledInnerR + scaledMiddleR) / 2;
                            int xi = (int)(cx + r * Math.Cos(midAngle));
                            int yi = (int)(cy + r * Math.Sin(midAngle));
                            DrawCross(output, xi, yi, 4, lineColor);
                        }
                        if (hasOuterBit)
                        {
                            float r = (scaledMiddleR + scaledOuterR) / 2;
                            int xo = (int)(cx + r * Math.Cos(midAngle));
                            int yo = (int)(cy + r * Math.Sin(midAngle));
                            DrawCross(output, xo, yo, 4, lineColor);
                        }
                    }
                }

                // Draw arrow indicator at Y-arrow position (calculated from NEW center)
                var arrowColor = new MCvScalar(0, 255, 255);  // Yellow
                double arrowPositionAfterRotation = arrowAngleFromNewCenter;
                int arrowX = (int)(cx + scaledOuterR * 0.85 * Math.Cos(arrowPositionAfterRotation));
                int arrowY = (int)(cy + scaledOuterR * 0.85 * Math.Sin(arrowPositionAfterRotation));
                CvInvoke.ArrowedLine(output, new Point(cx, cy), new Point(arrowX, arrowY),
                    arrowColor, 3, LineType.AntiAlias, 0, 0.15);
                CvInvoke.PutText(output, "S0", new Point(arrowX + 5, arrowY - 5),
                    FontFace.HersheySimplex, 0.4, arrowColor, 1);

                // Add decoded data text at top
                string dataText = result.IsValid ? result.DecodedData : "Invalid";
                CvInvoke.PutText(output, dataText, new Point(10, 25),
                    FontFace.HersheySimplex, 0.7, mainColor, 2);

                // Add binary string (inner and outer separated)
                if (result.IsValid && !string.IsNullOrEmpty(result.BinaryString) && result.BinaryString.Length == 48)
                {
                    var innerBits = new System.Text.StringBuilder(24);
                    var outerBits = new System.Text.StringBuilder(24);
                    for (int i = 0; i < 24; i++)
                    {
                        innerBits.Append(result.BinaryString[i * 2]);
                        outerBits.Append(result.BinaryString[i * 2 + 1]);
                    }
                    CvInvoke.PutText(output, innerBits.ToString(),
                        new Point(10, outputSize - 35), FontFace.HersheySimplex, 0.35, new MCvScalar(255, 255, 255), 1);
                    CvInvoke.PutText(output, outerBits.ToString(),
                        new Point(10, outputSize - 15), FontFace.HersheySimplex, 0.35, new MCvScalar(255, 255, 255), 1);
                }

                // Save debug image if DebugOutputDir is set
                if (!string.IsNullOrEmpty(DebugOutputDir))
                {
                    try
                    {
                        Directory.CreateDirectory(DebugOutputDir);
                        string timestamp = DateTime.Now.ToString("HHmmss_fff");

                        // Add rotation info text
                        CvInvoke.PutText(output, $"Arrow: {actualArrowAngle:F1} deg", new Point(10, outputSize - 60),
                            FontFace.HersheySimplex, 0.4, new MCvScalar(0, 255, 255), 1);
                        CvInvoke.PutText(output, $"Rotation: {rotationNeeded:F1} deg", new Point(10, outputSize - 80),
                            FontFace.HersheySimplex, 0.4, new MCvScalar(255, 255, 0), 1);

                        string debugPath = Path.Combine(DebugOutputDir, $"rotated_debug_{timestamp}.png");
                        output.Save(debugPath);
                        Log($"  [DEBUG] Saved rotated image: {debugPath}");
                    }
                    catch (Exception ex)
                    {
                        Log($"  [DEBUG] Failed to save rotated debug image: {ex.Message}");
                    }
                }

                // Cleanup temporary images
                cropped.Dispose();
                croppedForeground?.Dispose();
                rotatedForeground?.Dispose();
                scaledFgForArrow?.Dispose();

                return output;
            }
            catch
            {
                return source.Clone();
            }
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
                using var cropped = source.Clone();
                source.ROI = Rectangle.Empty;

                // Calculate actual arrow angle from center to arrow tip
                // Image coords: angles clockwise from right (0°=right, 90°=down)
                double actualArrowAngle;
                if (result.ArrowTip != PointF.Empty)
                {
                    actualArrowAngle = Math.Atan2(result.ArrowTip.Y - result.Center.Y,
                                                   result.ArrowTip.X - result.Center.X) * 180.0 / Math.PI;
                }
                else
                {
                    // Fallback to rotation angle
                    actualArrowAngle = result.RotationAngle;
                }

                // Calculate rotation angle to make arrow point right (0 degrees)
                // GetRotationMatrix2D rotates counter-clockwise, so use positive angle
                double rotationNeeded = actualArrowAngle;

                // Calculate ring center position in CROPPED image coordinates
                float ringCenterInCroppedX = (float)(result.Center.X - x);
                float ringCenterInCroppedY = (float)(result.Center.Y - y);

                // Rotate around the ACTUAL ring center (not image center)
                using var rotationMat = new Mat();
                CvInvoke.GetRotationMatrix2D(new PointF(ringCenterInCroppedX, ringCenterInCroppedY), rotationNeeded, 1.0, rotationMat);

                using var rotated = new Image<Bgr, byte>(cropped.Size);
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
#endif
