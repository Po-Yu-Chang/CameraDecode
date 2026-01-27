using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace CameraMaui.ShapeMatcher
{
    /// <summary>
    /// Pure C# shape-based matcher using Emgu.CV
    /// Implements gradient-based template matching similar to line2Dup
    /// </summary>
    public class ManagedShapeMatcher : IShapeBasedMatcher
    {
        private readonly List<ShapeTemplate> _templates = new();
        private readonly object _lock = new();
        private bool _disposed;

        // Parameters
        private readonly int _numFeatures;
        private readonly float _weakThreshold;
        private readonly float _strongThreshold;

        /// <summary>
        /// Template data structure
        /// </summary>
        private class ShapeTemplate
        {
            public string ClassId { get; set; } = "";
            public float Angle { get; set; }
            public float Scale { get; set; } = 1.0f;
            public int Width { get; set; }
            public int Height { get; set; }
            public List<GradientFeature> Features { get; set; } = new();
            public Image<Gray, byte>? BinaryImage { get; set; }
        }

        /// <summary>
        /// Gradient feature point
        /// </summary>
        private struct GradientFeature
        {
            public int X;      // Relative to template center
            public int Y;
            public int Label;  // Quantized gradient direction (0-7)
        }

        public ManagedShapeMatcher(int numFeatures = 128, float weakThreshold = 30f, float strongThreshold = 60f)
        {
            _numFeatures = numFeatures;
            _weakThreshold = weakThreshold;
            _strongThreshold = strongThreshold;
        }

        public int TemplateCount
        {
            get
            {
                lock (_lock) { return _templates.Count; }
            }
        }

        public bool IsReady => true;  // Always ready - no native DLL needed

        public int AddTemplateWithRotations(byte[] templateImage, int width, int height,
            string classId, float angleStart = 0f, float angleEnd = 360f, float angleStep = 15f)
        {
            var image = CreateImageFromBytes(templateImage, width, height);
            return AddTemplateWithRotations(image, classId, angleStart, angleEnd, angleStep);
        }

        public int AddTemplateWithRotations(Image<Gray, byte> templateImage, string classId,
            float angleStart = 0f, float angleEnd = 360f, float angleStep = 15f)
        {
            // OPTIMIZED: Reduced scales from 5 to 3 for faster matching
            // Original: { 0.85f, 0.925f, 1.0f, 1.075f, 1.15f } = 5 scales
            // Optimized: { 0.9f, 1.0f, 1.1f } = 3 scales (covers ±10% range)
            float[] scales = { 0.9f, 1.0f, 1.1f };

            // OPTIMIZED: Use 15° step (24 angles) instead of 10° (36 angles)
            // This gives 3 × 24 = 72 templates instead of 5 × 36 = 180
            float effectiveAngleStep = Math.Max(angleStep, 15f);
            int count = 0;

            foreach (float scale in scales)
            {
                // Scale the template
                Image<Gray, byte> scaledTemplate;
                if (Math.Abs(scale - 1.0f) < 0.01f)
                {
                    scaledTemplate = templateImage.Clone();
                }
                else
                {
                    int newWidth = (int)(templateImage.Width * scale);
                    int newHeight = (int)(templateImage.Height * scale);
                    scaledTemplate = templateImage.Resize(newWidth, newHeight, Inter.Linear);
                }

                var center = new PointF(scaledTemplate.Width / 2f, scaledTemplate.Height / 2f);

                for (float angle = angleStart; angle < angleEnd; angle += effectiveAngleStep)
                {
                    // Rotate template
                    var rotated = RotateImage(scaledTemplate, angle, center);
                    if (AddSingleTemplate(rotated, classId, angle, scale) >= 0)
                    {
                        count++;
                    }
                    rotated.Dispose();
                }

                scaledTemplate.Dispose();
            }

            System.Diagnostics.Debug.WriteLine($"[ManagedShapeMatcher] Added {count} templates (3 scales x rotations @ {effectiveAngleStep}° step) for class '{classId}'");
            return count;
        }

        public int AddTemplate(byte[] templateImage, int width, int height, string classId, float angle = 0f)
        {
            var image = CreateImageFromBytes(templateImage, width, height);
            int result = AddSingleTemplate(image, classId, angle, 1.0f);
            image.Dispose();
            return result;
        }

        private int AddSingleTemplate(Image<Gray, byte> image, string classId, float angle, float scale = 1.0f)
        {
            try
            {
                var template = new ShapeTemplate
                {
                    ClassId = classId,
                    Angle = angle,
                    Scale = scale,
                    Width = image.Width,
                    Height = image.Height
                };

                // Extract gradient features
                ExtractFeatures(image, template);

                // Store binary image for matching
                template.BinaryImage = new Image<Gray, byte>(image.Size);
                CvInvoke.Threshold(image, template.BinaryImage, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

                lock (_lock)
                {
                    _templates.Add(template);
                    return _templates.Count - 1;
                }
            }
            catch
            {
                return -1;
            }
        }

        private void ExtractFeatures(Image<Gray, byte> image, ShapeTemplate template)
        {
            // Compute gradients
            var dx = new Image<Gray, float>(image.Size);
            var dy = new Image<Gray, float>(image.Size);
            CvInvoke.Sobel(image, dx, DepthType.Cv32F, 1, 0, 3);
            CvInvoke.Sobel(image, dy, DepthType.Cv32F, 0, 1, 3);

            // Collect candidate features
            var candidates = new List<(float mag, GradientFeature feat)>();
            int cx = image.Width / 2;
            int cy = image.Height / 2;

            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    float gx = dx.Data[y, x, 0];
                    float gy = dy.Data[y, x, 0];
                    float mag = MathF.Sqrt(gx * gx + gy * gy);

                    if (mag >= _strongThreshold)
                    {
                        // Quantize angle to 8 bins
                        float theta = MathF.Atan2(gy, gx);
                        if (theta < 0) theta += MathF.PI * 2;
                        int bin = (int)(theta * 4 / MathF.PI) % 8;

                        candidates.Add((mag, new GradientFeature
                        {
                            X = x - cx,
                            Y = y - cy,
                            Label = bin
                        }));
                    }
                }
            }

            // Sort by magnitude and take top features
            candidates.Sort((a, b) => b.mag.CompareTo(a.mag));
            int count = Math.Min(_numFeatures, candidates.Count);
            template.Features = candidates.Take(count).Select(c => c.feat).ToList();

            dx.Dispose();
            dy.Dispose();
        }

        public ShapeMatcherResult[] Match(byte[] searchImage, int width, int height,
            float threshold = 0.5f, string? classId = null, int maxResults = 10)
        {
            var image = CreateImageFromBytes(searchImage, width, height);
            var results = MatchInternal(image, threshold, classId, maxResults);
            image.Dispose();
            return results;
        }

        private ShapeMatcherResult[] MatchInternal(Image<Gray, byte> image, float threshold,
            string? classId, int maxResults)
        {
            var matches = new List<ShapeMatcherResult>();

            // Compute response maps for the search image
            var responseMaps = ComputeResponseMaps(image);

            lock (_lock)
            {
                foreach (var template in _templates)
                {
                    if (classId != null && template.ClassId != classId)
                        continue;

                    // Slide template over image
                    var templateMatches = MatchTemplate(responseMaps, template, threshold, image.Width, image.Height);
                    matches.AddRange(templateMatches);
                }
            }

            // Sort by score and apply NMS
            matches.Sort((a, b) => b.Score.CompareTo(a.Score));
            ApplyNMS(matches, 30f);

            // Dispose response maps
            foreach (var map in responseMaps)
                map.Dispose();

            return matches.Take(maxResults).ToArray();
        }

        private List<Image<Gray, byte>> ComputeResponseMaps(Image<Gray, byte> image)
        {
            var maps = new List<Image<Gray, byte>>(8);

            // Compute gradients
            var dx = new Image<Gray, float>(image.Size);
            var dy = new Image<Gray, float>(image.Size);
            CvInvoke.Sobel(image, dx, DepthType.Cv32F, 1, 0, 3);
            CvInvoke.Sobel(image, dy, DepthType.Cv32F, 0, 1, 3);

            // Create 8 response maps (one per orientation bin)
            for (int i = 0; i < 8; i++)
            {
                maps.Add(new Image<Gray, byte>(image.Size));
            }

            // Fill response maps
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    float gx = dx.Data[y, x, 0];
                    float gy = dy.Data[y, x, 0];
                    float mag = MathF.Sqrt(gx * gx + gy * gy);

                    if (mag >= _weakThreshold)
                    {
                        float theta = MathF.Atan2(gy, gx);
                        if (theta < 0) theta += MathF.PI * 2;
                        int bin = (int)(theta * 4 / MathF.PI) % 8;

                        // Spread to neighboring bins for robustness
                        maps[bin].Data[y, x, 0] = 255;
                        maps[(bin + 1) % 8].Data[y, x, 0] = 255;
                        maps[(bin + 7) % 8].Data[y, x, 0] = 255;
                    }
                }
            }

            dx.Dispose();
            dy.Dispose();

            return maps;
        }

        private List<ShapeMatcherResult> MatchTemplate(List<Image<Gray, byte>> responseMaps,
            ShapeTemplate template, float threshold, int imageWidth, int imageHeight)
        {
            var matches = new List<ShapeMatcherResult>();

            if (template.Features.Count == 0)
                return matches;

            int step = 4;  // Sliding window step
            int numFeatures = template.Features.Count;
            int thresholdCount = (int)(threshold * numFeatures);

            // Find template bounds
            int minX = template.Features.Min(f => f.X);
            int maxX = template.Features.Max(f => f.X);
            int minY = template.Features.Min(f => f.Y);
            int maxY = template.Features.Max(f => f.Y);

            int startX = -minX + 1;
            int startY = -minY + 1;
            int endX = imageWidth - maxX - 1;
            int endY = imageHeight - maxY - 1;

            for (int cy = startY; cy < endY; cy += step)
            {
                for (int cx = startX; cx < endX; cx += step)
                {
                    int matchCount = 0;

                    foreach (var feat in template.Features)
                    {
                        int x = cx + feat.X;
                        int y = cy + feat.Y;

                        if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight)
                        {
                            if (responseMaps[feat.Label].Data[y, x, 0] > 0)
                            {
                                matchCount++;
                            }
                        }
                    }

                    if (matchCount >= thresholdCount)
                    {
                        float score = (float)matchCount / numFeatures;
                        matches.Add(new ShapeMatcherResult
                        {
                            IsFound = true,
                            X = cx,
                            Y = cy,
                            Angle = template.Angle,
                            Score = score,
                            ClassId = template.ClassId
                        });
                    }
                }
            }

            return matches;
        }

        public ShapeMatcherResult FindArrowInRing(byte[] searchImage, int width, int height,
            float centerX, float centerY, float innerRadius, float outerRadius,
            float threshold = 0.5f, string classId = "y_arrow")
        {
            var image = CreateImageFromBytes(searchImage, width, height);
            var result = FindArrowInRingInternal(image, new PointF(centerX, centerY),
                innerRadius, outerRadius, threshold, classId);
            image.Dispose();
            return result;
        }

        private ShapeMatcherResult FindArrowInRingInternal(Image<Gray, byte> image,
            PointF center, float innerRadius, float outerRadius, float threshold, string classId)
        {
            // OPTIMIZED: Two-stage search with parallelization
            // Stage 1: Coarse search on downscaled image to find candidate angles
            // Stage 2: Fine search only on best candidates

            var validTemplates = new List<ShapeTemplate>();
            lock (_lock)
            {
                validTemplates = _templates.Where(t =>
                    (classId == null || t.ClassId == classId) &&
                    t.BinaryImage != null &&
                    t.BinaryImage.Width < image.Width &&
                    t.BinaryImage.Height < image.Height).ToList();
            }

            if (validTemplates.Count == 0)
                return ShapeMatcherResult.NotFound("No valid templates");

            // Stage 1: Coarse search - use only scale=1.0 templates on downscaled image
            float coarseScale = 0.5f;
            using var coarseImage = image.Resize((int)(image.Width * coarseScale), (int)(image.Height * coarseScale), Inter.Linear);

            var coarseTemplates = validTemplates.Where(t => Math.Abs(t.Scale - 1.0f) < 0.05f).ToList();
            float bestCoarseAngle = 0f;
            float bestCoarseScore = 0f;

            // Parallel coarse search
            var coarseResults = new System.Collections.Concurrent.ConcurrentBag<(float angle, float score)>();
            System.Threading.Tasks.Parallel.ForEach(coarseTemplates, template =>
            {
                if (template.BinaryImage == null) return;

                // Scale template for coarse search
                int tw = (int)(template.BinaryImage.Width * coarseScale);
                int th = (int)(template.BinaryImage.Height * coarseScale);
                if (tw < 10 || th < 10) return;

                using var scaledTemplate = template.BinaryImage.Resize(tw, th, Inter.Linear);
                using var result = new Mat();
                CvInvoke.MatchTemplate(coarseImage, scaledTemplate, result, TemplateMatchingType.CcoeffNormed);

                double minVal = 0, maxVal = 0;
                Point minLoc = new Point(), maxLoc = new Point();
                CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                // Check if in ring region (scaled coordinates)
                float matchCenterX = (maxLoc.X + tw / 2f) / coarseScale;
                float matchCenterY = (maxLoc.Y + th / 2f) / coarseScale;
                float dx = matchCenterX - center.X;
                float dy = matchCenterY - center.Y;
                float dist = MathF.Sqrt(dx * dx + dy * dy);

                if (dist >= innerRadius && dist <= outerRadius && maxVal >= threshold * 0.7f)
                {
                    coarseResults.Add((template.Angle, (float)maxVal));
                }
            });

            // Find best coarse angle
            foreach (var (angle, score) in coarseResults)
            {
                if (score > bestCoarseScore)
                {
                    bestCoarseScore = score;
                    bestCoarseAngle = angle;
                }
            }

            // Stage 2: Fine search - only templates within ±30° of best coarse angle
            float angleRange = 30f;
            var fineTemplates = validTemplates.Where(t =>
            {
                float angleDiff = Math.Abs(t.Angle - bestCoarseAngle);
                if (angleDiff > 180) angleDiff = 360 - angleDiff;
                return angleDiff <= angleRange || bestCoarseScore < threshold * 0.7f;
            }).ToList();

            // Parallel fine search
            ShapeMatcherResult? best = null;
            float bestScore = 0;
            var lockObj = new object();

            System.Threading.Tasks.Parallel.ForEach(fineTemplates, template =>
            {
                if (template.BinaryImage == null) return;

                using var result = new Mat();
                CvInvoke.MatchTemplate(image, template.BinaryImage, result, TemplateMatchingType.CcoeffNormed);

                double minVal = 0, maxVal = 0;
                Point minLoc = new Point(), maxLoc = new Point();
                CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

                float matchCenterX = maxLoc.X + template.BinaryImage.Width / 2f;
                float matchCenterY = maxLoc.Y + template.BinaryImage.Height / 2f;
                float dx = matchCenterX - center.X;
                float dy = matchCenterY - center.Y;
                float dist = MathF.Sqrt(dx * dx + dy * dy);

                if (dist >= innerRadius && dist <= outerRadius && maxVal >= threshold)
                {
                    lock (lockObj)
                    {
                        if (maxVal > bestScore)
                        {
                            bestScore = (float)maxVal;

                            // Calculate actual angle from ring center to match position
                            // This is the geometric angle, not the template rotation angle
                            // Using image coordinates: 0°=right, 90°=down, 180°=left, 270°=up
                            double actualAngle = Math.Atan2(dy, dx) * 180.0 / Math.PI;
                            if (actualAngle < 0) actualAngle += 360;

                            best = new ShapeMatcherResult
                            {
                                IsFound = true,
                                X = matchCenterX,
                                Y = matchCenterY,
                                Angle = (float)actualAngle,  // Geometric angle from ring center to match
                                TemplateAngle = template.Angle,  // Template rotation angle for overlay
                                Scale = template.Scale,  // Template scale factor
                                Score = (float)maxVal,
                                ClassId = template.ClassId
                            };
                        }
                    }
                }
            });

            System.Diagnostics.Debug.WriteLine($"[ManagedShapeMatcher] Coarse: {coarseTemplates.Count} templates, best angle={bestCoarseAngle:F0}°, Fine: {fineTemplates.Count} templates");
            return best ?? ShapeMatcherResult.NotFound("No arrow found in ring region");
        }

        public int GetClassTemplateCount(string classId)
        {
            lock (_lock)
            {
                return _templates.Count(t => t.ClassId == classId);
            }
        }

        public bool SaveTemplates(string filePath)
        {
            // Not implemented for managed version - templates are regenerated
            return false;
        }

        public int LoadTemplates(string filePath)
        {
            // Not implemented for managed version
            return 0;
        }

        public void ClearTemplates()
        {
            lock (_lock)
            {
                foreach (var t in _templates)
                    t.BinaryImage?.Dispose();
                _templates.Clear();
            }
        }

        public string? GetLastError() => null;

        private static void ApplyNMS(List<ShapeMatcherResult> matches, float minDist)
        {
            for (int i = 0; i < matches.Count; i++)
            {
                if (matches[i] == null!) continue;

                for (int j = i + 1; j < matches.Count; j++)
                {
                    if (matches[j] == null!) continue;

                    float dx = matches[i].X - matches[j].X;
                    float dy = matches[i].Y - matches[j].Y;
                    float dist = MathF.Sqrt(dx * dx + dy * dy);

                    if (dist < minDist)
                    {
                        matches[j] = null!;
                    }
                }
            }

            matches.RemoveAll(m => m == null!);
        }

        private static Image<Gray, byte> CreateImageFromBytes(byte[] data, int width, int height)
        {
            var image = new Image<Gray, byte>(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    image.Data[y, x, 0] = data[y * width + x];
                }
            }
            return image;
        }

        private static Image<Gray, byte> RotateImage(Image<Gray, byte> src, float angle, PointF center)
        {
            var rotMat = new Mat();
            CvInvoke.GetRotationMatrix2D(center, angle, 1.0, rotMat);

            var dst = new Image<Gray, byte>(src.Size);
            CvInvoke.WarpAffine(src, dst, rotMat, src.Size, Inter.Linear, Warp.Default,
                BorderType.Constant, new MCvScalar(0));

            rotMat.Dispose();
            return dst;
        }

        public void Dispose()
        {
            if (_disposed) return;

            lock (_lock)
            {
                foreach (var t in _templates)
                    t.BinaryImage?.Dispose();
                _templates.Clear();
            }

            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~ManagedShapeMatcher()
        {
            Dispose();
        }
    }
}
