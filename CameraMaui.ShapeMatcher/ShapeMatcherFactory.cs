namespace CameraMaui.ShapeMatcher
{
    /// <summary>
    /// Factory for creating shape matcher instances.
    /// Automatically selects native or managed implementation based on availability.
    /// </summary>
    public static class ShapeMatcherFactory
    {
        private static bool? _nativeAvailable;

        /// <summary>
        /// Check if native DLL is available
        /// </summary>
        public static bool IsNativeAvailable
        {
            get
            {
                if (_nativeAvailable.HasValue)
                    return _nativeAvailable.Value;

                try
                {
                    var version = ShapeMatcherDetector.GetVersion();
                    _nativeAvailable = version != null;
                }
                catch
                {
                    _nativeAvailable = false;
                }

                return _nativeAvailable.Value;
            }
        }

        /// <summary>
        /// Create a shape matcher instance.
        /// Returns native implementation if available, otherwise managed implementation.
        /// </summary>
        public static IShapeBasedMatcher Create(
            int numFeatures = 128,
            float weakThreshold = 30f,
            float strongThreshold = 60f)
        {
            if (IsNativeAvailable)
            {
                System.Diagnostics.Debug.WriteLine("[ShapeMatcherFactory] Using native implementation");
                return new ShapeMatcherDetector(numFeatures, weakThreshold, strongThreshold);
            }
            else
            {
                System.Diagnostics.Debug.WriteLine("[ShapeMatcherFactory] Using managed implementation");
                return new ManagedShapeMatcher(numFeatures, weakThreshold, strongThreshold);
            }
        }

        /// <summary>
        /// Create a shape matcher instance using default parameters.
        /// </summary>
        public static IShapeBasedMatcher Create() => Create(128, 30f, 60f);

        /// <summary>
        /// Force using managed implementation (for testing)
        /// </summary>
        public static IShapeBasedMatcher CreateManaged(
            int numFeatures = 128,
            float weakThreshold = 30f,
            float strongThreshold = 60f)
        {
            return new ManagedShapeMatcher(numFeatures, weakThreshold, strongThreshold);
        }
    }
}
