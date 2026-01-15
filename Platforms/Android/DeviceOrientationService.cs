using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Android.Graphics;
using Android.Runtime;
using Android.Views;
using System.Drawing;
using CameraMaui;
using CameraMaui.Services;
using Emgu.CV;
using Emgu.CV.Structure;
using SkiaSharp;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
[assembly: Dependency(typeof(CameraMaui.Platforms.DeviceOrientationService))]
namespace CameraMaui.Platforms
{
    public class DeviceOrientationService : IDeviceOrentationService
    {
        public DeviceOrientation GetOrentation()
        {
           IWindowManager windowManager = Android.App.Application.Context.GetSystemService(Android.Content.Context.WindowService).JavaCast<IWindowManager>();
            var rotation = windowManager.DefaultDisplay.Rotation;
            bool isLandscape = rotation == SurfaceOrientation.Rotation90 || rotation == SurfaceOrientation.Rotation270;
            return isLandscape ? DeviceOrientation.Landscape : DeviceOrientation.Portrait;
        }

        //public object GetImage()
        //{
        //    Image<Gray, Byte> image = new Image<Gray, byte>(100, 100, new Gray(0));
        //    SKBitmap skBitmap = new SKBitmap(image.Width, image.Height, SKColorType.Gray8, SKAlphaType.Premul);
        //    for (int y = 0; y < image.Height; y++)
        //    {
        //        for (int x = 0; x < image.Width; x++)
        //        {
        //            Gray color = image[y, x];
        //            byte intensity = (byte)(color.Intensity * 255); // 将强度转换为 0 到 255 范围内的值
        //            skBitmap.SetPixel(x, y, new SKColor(intensity, intensity, intensity));
        //        }
        //    }

        //    return skBitmap;
        //}
        public object GetImage(SKBitmap skBitmap)
        {
            var image = new Image<Bgr, byte>(skBitmap.Width, skBitmap.Height);
            var pixelData = new byte[skBitmap.Width * skBitmap.Height * 4];

            // 读取 SKBitmap 的像素数据
            IntPtr ptr = skBitmap.GetPixels();
            System.Runtime.InteropServices.Marshal.Copy(ptr, pixelData, 0, pixelData.Length);

            for (int y = 0; y < skBitmap.Height; y++)
            {
                for (int x = 0; x < skBitmap.Width; x++)
                {
                    // 由于 SkiaSharp 中的 SKColor 是预乘 Alpha 的，需要转换回标准的 RGBA
                    int i = (y * skBitmap.Width + x) * 4;
                    byte blue = pixelData[i];
                    byte green = pixelData[i + 1];
                    byte red = pixelData[i + 2];
                    //byte alpha = pixelData[i + 3]; // Alpha 通道，如果需要

                    image.Data[y, x, 0] = blue;
                    image.Data[y, x, 1] = green;
                    image.Data[y, x, 2] = red;
                }
            }

            // 可以应用阈值或其他处理
            // 对彩色图像应用阈值需要其他方法，因为这会涉及到颜色空间转换等问题

            return ConvertToSKBitmap(image);
        }

        public SKBitmap ConvertToSKBitmap(Image<Bgr, Byte> image)
        {
            SKBitmap skBitmap = new SKBitmap(image.Width, image.Height, SKColorType.Bgra8888, SKAlphaType.Premul);

            unsafe
            {
                // 获取 Image 的 Mat 对象的指针
                byte* data = (byte*)image.Mat.DataPointer;
                int step = image.Mat.Step;

                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        // 直接访问图像数据
                        byte b = data[y * step + x * 3];
                        byte g = data[y * step + x * 3 + 1];
                        byte r = data[y * step + x * 3 + 2];

                        // 更新 SKBitmap 数据
                        skBitmap.SetPixel(x, y, new SKColor(r, g, b, 255));
                    }
                }
            }

            return skBitmap;
        }


    }
}
