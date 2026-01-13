using System;
using SkiaSharp;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

[assembly: Dependency(typeof(CameraMaui.Platforms.DeviceOrientationService))]
namespace CameraMaui.Platforms
{
    public class DeviceOrientationService : IDeviceOrentationService
    {
        public DeviceOrientation GetOrentation()
        {
            // Windows desktop is typically landscape
            return DeviceOrientation.Landscape;
        }

        public object GetImage(SKBitmap skBitmap)
        {
            // Convert SKBitmap to Emgu.CV Image
            var image = new Image<Bgr, byte>(skBitmap.Width, skBitmap.Height);
            var pixelData = new byte[skBitmap.Width * skBitmap.Height * 4];

            IntPtr ptr = skBitmap.GetPixels();
            System.Runtime.InteropServices.Marshal.Copy(ptr, pixelData, 0, pixelData.Length);

            for (int y = 0; y < skBitmap.Height; y++)
            {
                for (int x = 0; x < skBitmap.Width; x++)
                {
                    int i = (y * skBitmap.Width + x) * 4;
                    byte blue = pixelData[i];
                    byte green = pixelData[i + 1];
                    byte red = pixelData[i + 2];

                    image.Data[y, x, 0] = blue;
                    image.Data[y, x, 1] = green;
                    image.Data[y, x, 2] = red;
                }
            }

            // Example: Apply Gaussian blur using OpenCV
            var processedImage = image.SmoothGaussian(5);

            return ConvertToSKBitmap(processedImage);
        }

        public SKBitmap ConvertToSKBitmap(Image<Bgr, byte> image)
        {
            SKBitmap skBitmap = new SKBitmap(image.Width, image.Height, SKColorType.Bgra8888, SKAlphaType.Premul);

            unsafe
            {
                byte* data = (byte*)image.Mat.DataPointer;
                int step = image.Mat.Step;

                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        byte b = data[y * step + x * 3];
                        byte g = data[y * step + x * 3 + 1];
                        byte r = data[y * step + x * 3 + 2];

                        skBitmap.SetPixel(x, y, new SKColor(r, g, b, 255));
                    }
                }
            }

            return skBitmap;
        }

        /// <summary>
        /// Test method to verify Emgu.CV is working
        /// </summary>
        public static bool TestEmguCV()
        {
            try
            {
                // Create a simple test image
                using var testImage = new Image<Bgr, byte>(100, 100, new Bgr(255, 0, 0));

                // Apply some OpenCV operations
                using var grayImage = testImage.Convert<Gray, byte>();
                using var blurred = grayImage.SmoothGaussian(3);

                // Try edge detection
                using var edges = blurred.Canny(100, 200);

                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Emgu.CV Test Failed: {ex.Message}");
                return false;
            }
        }
    }
}
