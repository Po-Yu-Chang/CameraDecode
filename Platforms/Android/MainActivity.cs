using Android.App;
using Android.Content.PM;
using Android.OS;
using Emgu.CV;
using Emgu.CV.Structure;

namespace CameraMaui
{
    [Activity(Theme = "@style/Maui.SplashTheme", MainLauncher = true, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.UiMode | ConfigChanges.ScreenLayout | ConfigChanges.SmallestScreenSize | ConfigChanges.Density)]
    public class MainActivity : MauiAppCompatActivity
    {
      public MainActivity()
      {
            Image<Gray, Byte> image = new Image<Gray, byte>(100, 100, new Gray(0));
        }
    }
}
