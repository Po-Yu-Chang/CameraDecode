namespace CameraMaui
{
    /// <summary>
    /// Service helper to access DI services from anywhere in the app
    /// </summary>
    public static class ServiceHelper
    {
        public static IServiceProvider Services { get; private set; }

        public static void Initialize(IServiceProvider serviceProvider)
        {
            Services = serviceProvider;
        }

        public static T GetService<T>() where T : class
        {
            return Services?.GetService<T>();
        }
    }
}
