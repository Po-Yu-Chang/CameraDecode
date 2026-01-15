#if ANDROID || WINDOWS
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using SkiaSharp;
using System.Diagnostics;

namespace CameraMaui
{
    /// <summary>
    /// Test class to verify Emgu.CV functionality on Windows
    /// </summary>
    public static class EmguCVTest
    {
        /// <summary>
        /// Run all Emgu.CV tests and return results
        /// </summary>
        public static async Task<string> RunAllTestsAsync()
        {
            var results = new List<string>();

            results.Add("=== Emgu.CV Test Results ===\n");

            // Test 1: Basic Image Creation
            results.Add(TestBasicImageCreation());

            // Test 2: Color Conversion
            results.Add(TestColorConversion());

            // Test 3: Gaussian Blur
            results.Add(TestGaussianBlur());

            // Test 4: Canny Edge Detection
            results.Add(TestCannyEdgeDetection());

            // Test 5: Threshold
            results.Add(TestThreshold());

            // Test 6: Morphological Operations
            results.Add(TestMorphologicalOperations());

            // Test 7: Contour Detection
            results.Add(TestContourDetection());

            results.Add("\n=== Tests Complete ===");

            return string.Join("\n", results);
        }

        private static string TestBasicImageCreation()
        {
            try
            {
                using var image = new Image<Bgr, byte>(200, 200, new Bgr(255, 0, 0)); // Blue image
                return $"[PASS] Basic Image Creation: {image.Width}x{image.Height}";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Basic Image Creation: {ex.Message}";
            }
        }

        private static string TestColorConversion()
        {
            try
            {
                using var colorImage = new Image<Bgr, byte>(100, 100, new Bgr(100, 150, 200));
                using var grayImage = colorImage.Convert<Gray, byte>();
                return $"[PASS] Color Conversion (BGR to Gray): {grayImage.Width}x{grayImage.Height}";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Color Conversion: {ex.Message}";
            }
        }

        private static string TestGaussianBlur()
        {
            try
            {
                using var image = new Image<Gray, byte>(100, 100, new Gray(128));
                using var blurred = image.SmoothGaussian(5);
                return "[PASS] Gaussian Blur (kernel size 5)";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Gaussian Blur: {ex.Message}";
            }
        }

        private static string TestCannyEdgeDetection()
        {
            try
            {
                using var image = new Image<Gray, byte>(100, 100, new Gray(128));
                // Draw a rectangle to have edges
                CvInvoke.Rectangle(image, new System.Drawing.Rectangle(20, 20, 60, 60), new MCvScalar(255), -1);
                using var edges = image.Canny(100, 200);
                return "[PASS] Canny Edge Detection (threshold 100-200)";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Canny Edge Detection: {ex.Message}";
            }
        }

        private static string TestThreshold()
        {
            try
            {
                using var image = new Image<Gray, byte>(100, 100, new Gray(128));
                using var thresholded = image.ThresholdBinary(new Gray(100), new Gray(255));
                return "[PASS] Binary Threshold (value 100)";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Threshold: {ex.Message}";
            }
        }

        private static string TestMorphologicalOperations()
        {
            try
            {
                using var image = new Image<Gray, byte>(100, 100, new Gray(255));
                using var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(3, 3), new System.Drawing.Point(-1, -1));

                using var eroded = image.Erode(1);
                using var dilated = image.Dilate(1);

                return "[PASS] Morphological Operations (Erode & Dilate)";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Morphological Operations: {ex.Message}";
            }
        }

        private static string TestContourDetection()
        {
            try
            {
                using var image = new Image<Gray, byte>(100, 100, new Gray(0));
                // Draw a filled rectangle
                CvInvoke.Rectangle(image, new System.Drawing.Rectangle(25, 25, 50, 50), new MCvScalar(255), -1);

                using var contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
                using var hierarchy = new Mat();

                CvInvoke.FindContours(image, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                return $"[PASS] Contour Detection: Found {contours.Size} contour(s)";
            }
            catch (Exception ex)
            {
                return $"[FAIL] Contour Detection: {ex.Message}";
            }
        }

        /// <summary>
        /// Creates a test SKBitmap for testing without camera
        /// </summary>
        public static SKBitmap CreateTestBitmap(int width = 200, int height = 200)
        {
            var bitmap = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Premul);
            using var canvas = new SKCanvas(bitmap);

            // Draw gradient background
            canvas.Clear(SKColors.White);

            // Draw some shapes for testing
            using var paint = new SKPaint
            {
                Color = SKColors.Blue,
                Style = SKPaintStyle.Fill
            };
            canvas.DrawRect(new SKRect(20, 20, 80, 80), paint);

            paint.Color = SKColors.Red;
            canvas.DrawCircle(150, 50, 30, paint);

            paint.Color = SKColors.Green;
            canvas.DrawRect(new SKRect(50, 120, 150, 180), paint);

            return bitmap;
        }
    }
}
#endif
