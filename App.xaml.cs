using Camera.MAUI;

namespace CameraMaui
{
    public partial class App : Application
    {
        public App(IServiceProvider serviceProvider)
        {
            InitializeComponent();

            // Initialize ServiceHelper for global service access
            ServiceHelper.Initialize(serviceProvider);

            // Initialize Emgu.CV for mobile platforms
            // Windows doesn't require explicit initialization
#if ANDROID
            Emgu.CV.CvInvokeAndroid.Init();
#elif IOS
            Emgu.CV.CvInvokeIOS.Init();
#endif

            MainPage = new AppShell();
        }
       

    }

    public static class Global
    {
        public static CameraView cameraView= new CameraView();
        public static CancellationTokenSource cts;
    }
}
