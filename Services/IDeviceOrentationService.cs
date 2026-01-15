using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CameraMaui.Services
{
    public enum DeviceOrientation
    {
        Undefined,
        Landscape,
        Portrait
    }
    public interface IDeviceOrentationService
    {
        DeviceOrientation GetOrentation();
        object GetImage(SKBitmap skBitmap);
    }
}
