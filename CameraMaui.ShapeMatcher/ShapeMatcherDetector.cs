using System.Runtime.InteropServices;
using System.Text;

namespace CameraMaui.ShapeMatcher
{
    /// <summary>
    /// Shape-based template matcher implementation using native library
    /// </summary>
    public class ShapeMatcherDetector : IShapeBasedMatcher
    {
        private IntPtr _handle;
        private bool _disposed;
        private readonly object _lock = new();

        /// <summary>
        /// Default number of gradient features per template
        /// </summary>
        public const int DefaultNumFeatures = 128;

        /// <summary>
        /// Default weak gradient threshold
        /// </summary>
        public const float DefaultWeakThreshold = 30.0f;

        /// <summary>
        /// Default strong gradient threshold
        /// </summary>
        public const float DefaultStrongThreshold = 60.0f;

        /// <summary>
        /// Create a new shape matcher with default parameters
        /// </summary>
        public ShapeMatcherDetector()
            : this(DefaultNumFeatures, DefaultWeakThreshold, DefaultStrongThreshold)
        {
        }

        /// <summary>
        /// Create a new shape matcher with custom parameters
        /// </summary>
        /// <param name="numFeatures">Number of gradient features per template</param>
        /// <param name="weakThreshold">Weak gradient threshold</param>
        /// <param name="strongThreshold">Strong gradient threshold</param>
        public ShapeMatcherDetector(int numFeatures, float weakThreshold, float strongThreshold)
        {
            try
            {
                _handle = NativeMethods.ShapeMatcher_Create(numFeatures, weakThreshold, strongThreshold);
                if (_handle == IntPtr.Zero)
                {
                    throw new InvalidOperationException("Failed to create native shape matcher");
                }
            }
            catch (DllNotFoundException ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ShapeMatcher] Native DLL not found: {ex.Message}");
                _handle = IntPtr.Zero;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ShapeMatcher] Failed to initialize: {ex.Message}");
                _handle = IntPtr.Zero;
            }
        }

        /// <inheritdoc/>
        public int TemplateCount
        {
            get
            {
                if (_handle == IntPtr.Zero) return 0;
                lock (_lock)
                {
                    return NativeMethods.ShapeMatcher_GetTemplateCount(_handle);
                }
            }
        }

        /// <inheritdoc/>
        public bool IsReady => _handle != IntPtr.Zero;

        /// <inheritdoc/>
        public int AddTemplateWithRotations(byte[] templateImage, int width, int height,
            string classId, float angleStart = 0f, float angleEnd = 360f, float angleStep = 15f)
        {
            if (_handle == IntPtr.Zero)
                return -1;

            if (templateImage == null || templateImage.Length < width * height)
                return -1;

            lock (_lock)
            {
                return NativeMethods.ShapeMatcher_AddTemplateWithRotations(
                    _handle,
                    templateImage,
                    width,
                    height,
                    width, // stride = width for continuous data
                    null,  // no mask
                    classId ?? "default",
                    angleStart,
                    angleEnd,
                    angleStep);
            }
        }

        /// <inheritdoc/>
        public int AddTemplate(byte[] templateImage, int width, int height, string classId, float angle = 0f)
        {
            if (_handle == IntPtr.Zero)
                return -1;

            if (templateImage == null || templateImage.Length < width * height)
                return -1;

            lock (_lock)
            {
                return NativeMethods.ShapeMatcher_AddTemplate(
                    _handle,
                    templateImage,
                    width,
                    height,
                    width,
                    null,
                    classId ?? "default",
                    angle);
            }
        }

        /// <inheritdoc/>
        public ShapeMatcherResult[] Match(byte[] searchImage, int width, int height,
            float threshold = 0.5f, string? classId = null, int maxResults = 10)
        {
            if (_handle == IntPtr.Zero)
                return Array.Empty<ShapeMatcherResult>();

            if (searchImage == null || searchImage.Length < width * height)
                return Array.Empty<ShapeMatcherResult>();

            var nativeResults = new NativeShapeMatchResult[maxResults];

            int count;
            lock (_lock)
            {
                count = NativeMethods.ShapeMatcher_Match(
                    _handle,
                    searchImage,
                    width,
                    height,
                    width,
                    null,
                    threshold,
                    classId,
                    nativeResults,
                    maxResults);
            }

            if (count <= 0)
                return Array.Empty<ShapeMatcherResult>();

            var results = new ShapeMatcherResult[count];
            for (int i = 0; i < count; i++)
            {
                results[i] = ConvertResult(nativeResults[i]);
            }

            return results;
        }

        /// <inheritdoc/>
        public ShapeMatcherResult FindArrowInRing(byte[] searchImage, int width, int height,
            float centerX, float centerY, float innerRadius, float outerRadius,
            float threshold = 0.5f, string classId = "y_arrow")
        {
            if (_handle == IntPtr.Zero)
                return ShapeMatcherResult.NotFound("Matcher not initialized");

            if (searchImage == null || searchImage.Length < width * height)
                return ShapeMatcherResult.NotFound("Invalid search image");

            NativeShapeMatchResult nativeResult;
            int found;

            lock (_lock)
            {
                found = NativeMethods.ShapeMatcher_MatchInRing(
                    _handle,
                    searchImage,
                    width,
                    height,
                    width,
                    centerX,
                    centerY,
                    innerRadius,
                    outerRadius,
                    threshold,
                    classId,
                    out nativeResult);
            }

            if (found <= 0)
            {
                return ShapeMatcherResult.NotFound(found < 0 ? GetLastError() : "No match found");
            }

            return ConvertResult(nativeResult);
        }

        /// <inheritdoc/>
        public int GetClassTemplateCount(string classId)
        {
            if (_handle == IntPtr.Zero) return 0;

            lock (_lock)
            {
                return NativeMethods.ShapeMatcher_GetClassTemplateCount(_handle, classId);
            }
        }

        /// <inheritdoc/>
        public bool SaveTemplates(string filePath)
        {
            if (_handle == IntPtr.Zero) return false;

            lock (_lock)
            {
                return NativeMethods.ShapeMatcher_SaveTemplates(_handle, filePath) == 1;
            }
        }

        /// <inheritdoc/>
        public int LoadTemplates(string filePath)
        {
            if (_handle == IntPtr.Zero) return -1;

            lock (_lock)
            {
                return NativeMethods.ShapeMatcher_LoadTemplates(_handle, filePath);
            }
        }

        /// <inheritdoc/>
        public void ClearTemplates()
        {
            if (_handle == IntPtr.Zero) return;

            lock (_lock)
            {
                NativeMethods.ShapeMatcher_ClearTemplates(_handle);
            }
        }

        /// <inheritdoc/>
        public string? GetLastError()
        {
            if (_handle == IntPtr.Zero) return "Matcher not initialized";

            var buffer = new StringBuilder(256);
            lock (_lock)
            {
                int len = NativeMethods.ShapeMatcher_GetLastError(_handle, buffer, buffer.Capacity);
                return len > 0 ? buffer.ToString() : null;
            }
        }

        /// <summary>
        /// Get library version
        /// </summary>
        public static string? GetVersion()
        {
            try
            {
                IntPtr ptr = NativeMethods.ShapeMatcher_GetVersion();
                return ptr != IntPtr.Zero ? Marshal.PtrToStringAnsi(ptr) : null;
            }
            catch
            {
                return null;
            }
        }

        private static ShapeMatcherResult ConvertResult(NativeShapeMatchResult native)
        {
            return new ShapeMatcherResult
            {
                IsFound = true,
                X = native.X,
                Y = native.Y,
                Angle = native.Angle,
                Scale = native.Scale,
                Score = native.Score,
                TemplateId = native.TemplateId,
                ClassId = native.ClassId ?? ""
            };
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (_handle != IntPtr.Zero)
            {
                lock (_lock)
                {
                    if (_handle != IntPtr.Zero)
                    {
                        NativeMethods.ShapeMatcher_Destroy(_handle);
                        _handle = IntPtr.Zero;
                    }
                }
            }

            _disposed = true;
        }

        ~ShapeMatcherDetector()
        {
            Dispose(false);
        }
    }
}
