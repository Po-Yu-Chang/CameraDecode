using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;

namespace CameraMaui.ShapeMatcher
{
    /// <summary>
    /// Extension methods for Emgu.CV integration
    /// </summary>
    public static class EmguCvExtensions
    {
        /// <summary>
        /// Add a template from an Emgu.CV image with rotation variants
        /// </summary>
        public static int AddTemplateWithRotations(
            this IShapeBasedMatcher matcher,
            Image<Gray, byte> templateImage,
            string classId,
            float angleStart = 0f,
            float angleEnd = 360f,
            float angleStep = 15f)
        {
            byte[] data = ExtractImageData(templateImage);
            return matcher.AddTemplateWithRotations(
                data,
                templateImage.Width,
                templateImage.Height,
                classId,
                angleStart,
                angleEnd,
                angleStep);
        }

        /// <summary>
        /// Add a single template from an Emgu.CV image
        /// </summary>
        public static int AddTemplate(
            this IShapeBasedMatcher matcher,
            Image<Gray, byte> templateImage,
            string classId,
            float angle = 0f)
        {
            byte[] data = ExtractImageData(templateImage);
            return matcher.AddTemplate(data, templateImage.Width, templateImage.Height, classId, angle);
        }

        /// <summary>
        /// Find matches in an Emgu.CV image
        /// </summary>
        public static ShapeMatcherResult[] Match(
            this IShapeBasedMatcher matcher,
            Image<Gray, byte> searchImage,
            float threshold = 0.5f,
            string? classId = null,
            int maxResults = 10)
        {
            byte[] data = ExtractImageData(searchImage);
            return matcher.Match(data, searchImage.Width, searchImage.Height, threshold, classId, maxResults);
        }

        /// <summary>
        /// Find arrow in ring region from an Emgu.CV image
        /// </summary>
        public static ShapeMatcherResult FindArrowInRing(
            this IShapeBasedMatcher matcher,
            Image<Gray, byte> searchImage,
            PointF ringCenter,
            float innerRadius,
            float outerRadius,
            float threshold = 0.5f,
            string classId = "y_arrow")
        {
            byte[] data = ExtractImageData(searchImage);
            return matcher.FindArrowInRing(
                data,
                searchImage.Width,
                searchImage.Height,
                ringCenter.X,
                ringCenter.Y,
                innerRadius,
                outerRadius,
                threshold,
                classId);
        }

        /// <summary>
        /// Extract raw grayscale data from Emgu.CV image
        /// </summary>
        private static byte[] ExtractImageData(Image<Gray, byte> image)
        {
            int width = image.Width;
            int height = image.Height;
            byte[] data = new byte[width * height];

            // Use MIplImage to access data directly
            unsafe
            {
                var miplImage = image.MIplImage;
                int stride = miplImage.WidthStep;
                byte* srcPtr = (byte*)miplImage.ImageData.ToPointer();

                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = srcPtr + y * stride;
                    for (int x = 0; x < width; x++)
                    {
                        data[y * width + x] = rowPtr[x];
                    }
                }
            }

            return data;
        }

        /// <summary>
        /// Create a Y-arrow template image
        /// </summary>
        /// <param name="size">Template size (width and height)</param>
        /// <returns>Grayscale template image</returns>
        public static Image<Gray, byte> CreateYArrowTemplate(int size = 80)
        {
            var template = new Image<Gray, byte>(size, size);
            template.SetZero();

            int cx = size / 2;
            int cy = size / 2;

            // Draw Y-arrow shape using Emgu.CV drawing functions
            // Tip pointing down (towards ring center), branches pointing up-left and up-right

            // Main stem (pointing down)
            var stemPoints = new Point[]
            {
                new Point(cx - 5, cy - 5),
                new Point(cx + 5, cy - 5),
                new Point(cx + 4, cy + 25),
                new Point(cx - 4, cy + 25)
            };

            // Left branch
            var leftBranch = new Point[]
            {
                new Point(cx - 3, cy - 5),
                new Point(cx - 7, cy + 3),
                new Point(8, 8),
                new Point(15, 3)
            };

            // Right branch
            var rightBranch = new Point[]
            {
                new Point(cx + 3, cy - 5),
                new Point(cx + 7, cy + 3),
                new Point(size - 15, 3),
                new Point(size - 8, 8)
            };

            // Center junction
            var center = new Point[]
            {
                new Point(cx - 8, cy - 8),
                new Point(cx + 8, cy - 8),
                new Point(cx + 8, cy + 8),
                new Point(cx - 8, cy + 8)
            };

            // Draw all parts
            using var stemContour = new Emgu.CV.Util.VectorOfPoint(stemPoints);
            using var leftContour = new Emgu.CV.Util.VectorOfPoint(leftBranch);
            using var rightContour = new Emgu.CV.Util.VectorOfPoint(rightBranch);
            using var centerContour = new Emgu.CV.Util.VectorOfPoint(center);

            Emgu.CV.CvInvoke.FillPoly(template, centerContour, new MCvScalar(255));
            Emgu.CV.CvInvoke.FillPoly(template, stemContour, new MCvScalar(255));
            Emgu.CV.CvInvoke.FillPoly(template, leftContour, new MCvScalar(255));
            Emgu.CV.CvInvoke.FillPoly(template, rightContour, new MCvScalar(255));

            return template;
        }
    }
}
