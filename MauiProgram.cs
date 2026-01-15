using Microsoft.Extensions.Logging;
using CameraMaui;
using CameraMaui.Pages;
using CameraMaui.Services;
using Camera.MAUI;
using Microsoft.Extensions.DependencyInjection.Extensions;
using SkiaSharp.Views.Maui.Controls.Hosting;

namespace CameraMaui
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCameraView()
                .UseSkiaSharp()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

            // Register platform-specific services using MAUI DI
#if ANDROID
            builder.Services.AddSingleton<IDeviceOrentationService, CameraMaui.Platforms.DeviceOrientationService>();
#elif WINDOWS
            builder.Services.AddSingleton<IDeviceOrentationService, CameraMaui.Platforms.DeviceOrientationService>();
#else
            // Fallback for other platforms - can add iOS/Mac implementations later
#endif

            // Register MainPage for DI
            builder.Services.AddTransient<MainPage>();

#if DEBUG
    		builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}
