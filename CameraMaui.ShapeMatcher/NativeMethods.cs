using System.Runtime.InteropServices;

namespace CameraMaui.ShapeMatcher
{
    /// <summary>
    /// Native result structure (must match C struct exactly)
    /// </summary>
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal struct NativeShapeMatchResult
    {
        public float X;
        public float Y;
        public float Angle;
        public float Scale;
        public float Score;
        public int TemplateId;
        public int ClassIdLen;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string ClassId;
    }

    /// <summary>
    /// P/Invoke declarations for ShapeBasedMatcher native library
    /// </summary>
    internal static class NativeMethods
    {
        private const string DllName = "ShapeBasedMatcher";

        /// <summary>
        /// Create a new shape matcher instance
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ShapeMatcher_Create(
            int num_features,
            float weak_thresh,
            float strong_thresh);

        /// <summary>
        /// Destroy a shape matcher instance
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ShapeMatcher_Destroy(IntPtr handle);

        /// <summary>
        /// Add template with rotation variants
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_AddTemplateWithRotations(
            IntPtr handle,
            byte[] image_data,
            int width,
            int height,
            int stride,
            byte[]? mask_data,
            [MarshalAs(UnmanagedType.LPStr)] string class_id,
            float angle_start,
            float angle_end,
            float angle_step);

        /// <summary>
        /// Add single template
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_AddTemplate(
            IntPtr handle,
            byte[] image_data,
            int width,
            int height,
            int stride,
            byte[]? mask_data,
            [MarshalAs(UnmanagedType.LPStr)] string class_id,
            float angle);

        /// <summary>
        /// Match templates in image
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_Match(
            IntPtr handle,
            byte[] image_data,
            int width,
            int height,
            int stride,
            byte[]? mask_data,
            float threshold,
            [MarshalAs(UnmanagedType.LPStr)] string? class_id,
            [In, Out] NativeShapeMatchResult[] results,
            int max_results);

        /// <summary>
        /// Match in ring region
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_MatchInRing(
            IntPtr handle,
            byte[] image_data,
            int width,
            int height,
            int stride,
            float center_x,
            float center_y,
            float inner_radius,
            float outer_radius,
            float threshold,
            [MarshalAs(UnmanagedType.LPStr)] string? class_id,
            out NativeShapeMatchResult result);

        /// <summary>
        /// Get template count
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_GetTemplateCount(IntPtr handle);

        /// <summary>
        /// Get class template count
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_GetClassTemplateCount(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string class_id);

        /// <summary>
        /// Save templates
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_SaveTemplates(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string file_path);

        /// <summary>
        /// Load templates
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_LoadTemplates(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string file_path);

        /// <summary>
        /// Clear templates
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ShapeMatcher_ClearTemplates(IntPtr handle);

        /// <summary>
        /// Get last error
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ShapeMatcher_GetLastError(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] System.Text.StringBuilder buffer,
            int buffer_size);

        /// <summary>
        /// Get version
        /// </summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ShapeMatcher_GetVersion();
    }
}
