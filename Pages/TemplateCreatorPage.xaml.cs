using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;
using SkiaSharp;
using SkiaSharp.Views.Maui;
using SkiaSharp.Views.Maui.Controls;

#if ANDROID || WINDOWS
using CameraMaui.RingCode;
using Emgu.CV;
using Emgu.CV.Structure;
#endif

namespace CameraMaui.Pages
{
    public enum TemplateType
    {
        Dark,   // 深色箭頭
        Light   // 淺色箭頭
    }

    public partial class TemplateCreatorPage : ContentPage
    {
        private SKBitmap _sourceBitmap;
#if ANDROID || WINDOWS
        private Image<Gray, byte> _sourceImage;
        private ArrowTemplateMatcher _darkTemplateMatcher;
        private ArrowTemplateMatcher _lightTemplateMatcher;
#endif
        private int _selectionSize = 50;
        private SKPoint _selectionCenter;
        private bool _hasSelection;

        // Cached transform values for coordinate conversion
        private float _scale = 0f;
        private float _offsetX = 0f;
        private float _offsetY = 0f;

        // Drag state
        private bool _isDragging = false;

        // Current template type being edited
        private TemplateType _currentTemplateType = TemplateType.Dark;

        public TemplateCreatorPage()
        {
            InitializeComponent();
#if ANDROID || WINDOWS
            _darkTemplateMatcher = new ArrowTemplateMatcher();
            _lightTemplateMatcher = new ArrowTemplateMatcher();
            LoadExistingTemplates();
#endif
            UpdateTemplateTypeSelection();
        }

#if ANDROID || WINDOWS
        private void LoadExistingTemplates()
        {
            // Load dark template
            var darkPath = GetTemplatePath(TemplateType.Dark);
            if (File.Exists(darkPath))
            {
                if (_darkTemplateMatcher.LoadTemplate(darkPath))
                {
                    UpdateTemplatePreview(TemplateType.Dark);
                    lblDarkTemplateStatus.Text = "已載入";
                }
            }

            // Load light template
            var lightPath = GetTemplatePath(TemplateType.Light);
            if (File.Exists(lightPath))
            {
                if (_lightTemplateMatcher.LoadTemplate(lightPath))
                {
                    UpdateTemplatePreview(TemplateType.Light);
                    lblLightTemplateStatus.Text = "已載入";
                }
            }
        }
#endif

        private string GetTemplatePath(TemplateType type)
        {
            var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var filename = type == TemplateType.Dark ? "arrow_template_dark.png" : "arrow_template_light.png";
            return Path.Combine(appData, filename);
        }

        private void UpdateTemplateTypeSelection()
        {
            // Update visual selection state
            if (_currentTemplateType == TemplateType.Dark)
            {
                frameDarkTemplate.BorderColor = Color.FromArgb("#F9C846");
                frameLightTemplate.BorderColor = Colors.Transparent;
                lblTemplateInfo.Text = "正在編輯: 深色箭頭範本";
            }
            else
            {
                frameDarkTemplate.BorderColor = Colors.Transparent;
                frameLightTemplate.BorderColor = Color.FromArgb("#F9C846");
                lblTemplateInfo.Text = "正在編輯: 淺色箭頭範本";
            }
        }

        private void OnDarkTemplateSelected(object sender, EventArgs e)
        {
            _currentTemplateType = TemplateType.Dark;
            UpdateTemplateTypeSelection();
        }

        private void OnLightTemplateSelected(object sender, EventArgs e)
        {
            _currentTemplateType = TemplateType.Light;
            UpdateTemplateTypeSelection();
        }

        /// <summary>
        /// SKCanvasView PaintSurface - draws image and selection rectangle
        /// </summary>
        private void OnPaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;
            var info = e.Info;

            // Clear with white background
            canvas.Clear(SKColors.White);

            if (_sourceBitmap == null) return;

            // Calculate AspectFit transform
            float viewWidth = info.Width;
            float viewHeight = info.Height;
            float imageWidth = _sourceBitmap.Width;
            float imageHeight = _sourceBitmap.Height;

            float scaleX = viewWidth / imageWidth;
            float scaleY = viewHeight / imageHeight;
            _scale = Math.Min(scaleX, scaleY);

            float scaledWidth = imageWidth * _scale;
            float scaledHeight = imageHeight * _scale;

            _offsetX = (viewWidth - scaledWidth) / 2;
            _offsetY = (viewHeight - scaledHeight) / 2;

            // Draw the image
            var destRect = new SKRect(_offsetX, _offsetY, _offsetX + scaledWidth, _offsetY + scaledHeight);
            canvas.DrawBitmap(_sourceBitmap, destRect);

            // Draw selection rectangle if we have one
            if (_hasSelection)
            {
                // Convert image coordinates to view coordinates
                float viewX = _selectionCenter.X * _scale + _offsetX;
                float viewY = _selectionCenter.Y * _scale + _offsetY;
                float viewSize = _selectionSize * _scale;

                // Calculate rectangle bounds
                var selRect = new SKRect(
                    viewX - viewSize / 2,
                    viewY - viewSize / 2,
                    viewX + viewSize / 2,
                    viewY + viewSize / 2);

                // Draw semi-transparent fill
                using var fillPaint = new SKPaint
                {
                    Color = new SKColor(255, 0, 0, 77), // Red with 30% opacity
                    Style = SKPaintStyle.Fill
                };
                canvas.DrawRect(selRect, fillPaint);

                // Draw border
                using var strokePaint = new SKPaint
                {
                    Color = SKColors.Red,
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 3
                };
                canvas.DrawRect(selRect, strokePaint);

                // Draw crosshair at center
                using var crossPaint = new SKPaint
                {
                    Color = SKColors.Yellow,
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 2
                };
                float crossSize = 15;
                canvas.DrawLine(viewX - crossSize, viewY, viewX + crossSize, viewY, crossPaint);
                canvas.DrawLine(viewX, viewY - crossSize, viewX, viewY + crossSize, crossPaint);
            }
        }

        /// <summary>
        /// Handle touch events on SKCanvasView - supports tap and drag
        /// </summary>
        private void OnCanvasTouch(object sender, SKTouchEventArgs e)
        {
#if ANDROID || WINDOWS
            if (_sourceBitmap == null) return;

            // Guard against touch before first paint (scale not yet calculated)
            if (_scale <= 0)
            {
                e.Handled = true;
                return;
            }

            // Get touch location and convert to image coordinates
            float touchX = e.Location.X;
            float touchY = e.Location.Y;
            float imageX = (touchX - _offsetX) / _scale;
            float imageY = (touchY - _offsetY) / _scale;

            switch (e.ActionType)
            {
                case SKTouchAction.Pressed:
                    // Start dragging
                    _isDragging = true;
                    UpdateSelectionPosition(imageX, imageY);
                    break;

                case SKTouchAction.Moved:
                    // Continue dragging - update position in real-time
                    if (_isDragging)
                    {
                        UpdateSelectionPosition(imageX, imageY);
                    }
                    break;

                case SKTouchAction.Released:
                case SKTouchAction.Cancelled:
                    // End dragging - finalize and extract template
                    if (_isDragging && _hasSelection)
                    {
                        ExtractAndPreviewTemplate();
                    }
                    _isDragging = false;
                    break;
            }

            e.Handled = true;
#endif
        }

#if ANDROID || WINDOWS
        /// <summary>
        /// Update selection position with bounds checking
        /// </summary>
        private void UpdateSelectionPosition(float imageX, float imageY)
        {
            // Clamp to image bounds
            imageX = Math.Max(0, Math.Min(imageX, _sourceBitmap.Width - 1));
            imageY = Math.Max(0, Math.Min(imageY, _sourceBitmap.Height - 1));

            _selectionCenter = new SKPoint(imageX, imageY);
            _hasSelection = true;

            // Trigger redraw
            skCanvasView.InvalidateSurface();

            btnSave.IsEnabled = true;
            lblStatus.Text = $"Selected at ({imageX:F0}, {imageY:F0})";
        }
#endif

        private async void OnLoadImageClicked(object sender, EventArgs e)
        {
#if ANDROID || WINDOWS
            try
            {
                var result = await FilePicker.Default.PickAsync(new PickOptions
                {
                    PickerTitle = "Select an image with ring codes",
                    FileTypes = FilePickerFileType.Images
                });

                if (result != null)
                {
                    await LoadImage(result.FullPath);
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", $"Failed to load image: {ex.Message}", "OK");
            }
#else
            await DisplayAlert("Not Supported", "This feature is only available on Android and Windows", "OK");
#endif
        }

#if ANDROID || WINDOWS
        private async Task LoadImage(string path)
        {
            try
            {
                using var stream = File.OpenRead(path);
                _sourceBitmap = SKBitmap.Decode(stream);

                if (_sourceBitmap != null)
                {
                    // Convert to Emgu.CV image
                    _sourceImage = ConvertSKBitmapToEmguGray(_sourceBitmap);

                    // Clear selection and trigger redraw
                    _hasSelection = false;
                    skCanvasView.InvalidateSurface();

                    instructionOverlay.IsVisible = false;
                    lblStatus.Text = $"{_sourceBitmap.Width}x{_sourceBitmap.Height} - Tap to select arrow";
                    btnSave.IsEnabled = false;
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", $"Failed to load image: {ex.Message}", "OK");
            }
        }

        private void ExtractAndPreviewTemplate()
        {
            if (_sourceImage == null || !_hasSelection) return;

            try
            {
                int halfSize = _selectionSize / 2;
                int x = (int)Math.Max(0, _selectionCenter.X - halfSize);
                int y = (int)Math.Max(0, _selectionCenter.Y - halfSize);
                int w = Math.Min(_selectionSize, _sourceImage.Width - x);
                int h = Math.Min(_selectionSize, _sourceImage.Height - y);

                if (w <= 0 || h <= 0) return;

                _sourceImage.ROI = new System.Drawing.Rectangle(x, y, w, h);
                var cropped = _sourceImage.Clone();
                _sourceImage.ROI = System.Drawing.Rectangle.Empty;

                // Resize to standard size
                var resized = new Image<Gray, byte>(100, 100);
                CvInvoke.Resize(cropped, resized, new System.Drawing.Size(100, 100));

                // *** 預覽時也二值化，讓用戶看到實際儲存的效果 ***
                var binarized = new Image<Gray, byte>(100, 100);
                CvInvoke.Threshold(resized, binarized, 0, 255,
                    Emgu.CV.CvEnum.ThresholdType.Binary | Emgu.CV.CvEnum.ThresholdType.Otsu);

                // Load binarized template into matcher for preview
                var matcher = _currentTemplateType == TemplateType.Dark ? _darkTemplateMatcher : _lightTemplateMatcher;
                matcher.LoadTemplateFromImage(binarized);
                UpdateTemplatePreview(_currentTemplateType);

                var typeName = _currentTemplateType == TemplateType.Dark ? "深色" : "淺色";
                lblTemplateInfo.Text = $"{typeName}範本: {_selectionSize}x{_selectionSize}px (二值化預覽)";
            }
            catch (Exception ex)
            {
                lblTemplateInfo.Text = $"錯誤: {ex.Message}";
            }
        }

        private void UpdateTemplatePreview(TemplateType type)
        {
            var matcher = type == TemplateType.Dark ? _darkTemplateMatcher : _lightTemplateMatcher;
            var preview = type == TemplateType.Dark ? darkTemplatePreview : lightTemplatePreview;

            if (matcher.IsLoaded && matcher.TemplatePath != null)
            {
                try
                {
                    var path = matcher.TemplatePath;
                    if (File.Exists(path))
                    {
                        preview.Source = ImageSource.FromFile(path);
                    }
                }
                catch { }
            }
        }
#endif

        private void OnSizeDecreaseClicked(object sender, EventArgs e)
        {
            if (_selectionSize > 20)
            {
                _selectionSize -= 10;
                lblSelectionSize.Text = _selectionSize.ToString();
                if (_hasSelection)
                {
                    skCanvasView.InvalidateSurface();
#if ANDROID || WINDOWS
                    ExtractAndPreviewTemplate();
#endif
                }
            }
        }

        private void OnSizeIncreaseClicked(object sender, EventArgs e)
        {
            if (_selectionSize < 200)
            {
                _selectionSize += 10;
                lblSelectionSize.Text = _selectionSize.ToString();
                if (_hasSelection)
                {
                    skCanvasView.InvalidateSurface();
#if ANDROID || WINDOWS
                    ExtractAndPreviewTemplate();
#endif
                }
            }
        }

        private async void OnLoadTemplateClicked(object sender, EventArgs e)
        {
#if ANDROID || WINDOWS
            try
            {
                var typeName = _currentTemplateType == TemplateType.Dark ? "深色" : "淺色";
                var result = await FilePicker.Default.PickAsync(new PickOptions
                {
                    PickerTitle = $"選擇{typeName}箭頭範本圖片",
                    FileTypes = FilePickerFileType.Images
                });

                if (result != null)
                {
                    var matcher = _currentTemplateType == TemplateType.Dark ? _darkTemplateMatcher : _lightTemplateMatcher;
                    var statusLabel = _currentTemplateType == TemplateType.Dark ? lblDarkTemplateStatus : lblLightTemplateStatus;

                    if (matcher.LoadTemplate(result.FullPath))
                    {
                        // Copy to default location
                        var defaultPath = GetTemplatePath(_currentTemplateType);
                        File.Copy(result.FullPath, defaultPath, true);
                        matcher.LoadTemplate(defaultPath);

                        UpdateTemplatePreview(_currentTemplateType);
                        statusLabel.Text = "已載入";
                        lblTemplateInfo.Text = $"{typeName}範本已載入";
                        await DisplayAlert("成功", $"{typeName}箭頭範本載入成功", "確定");
                    }
                    else
                    {
                        await DisplayAlert("錯誤", "無法載入範本", "確定");
                    }
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("錯誤", $"載入範本失敗: {ex.Message}", "確定");
            }
#else
            await DisplayAlert("不支援", "此功能僅適用於 Android 和 Windows", "確定");
#endif
        }

        private async void OnSaveTemplateClicked(object sender, EventArgs e)
        {
#if ANDROID || WINDOWS
            if (!_hasSelection || _sourceImage == null)
            {
                await DisplayAlert("錯誤", "請先選取範本區域", "確定");
                return;
            }

            try
            {
                var typeName = _currentTemplateType == TemplateType.Dark ? "深色" : "淺色";
                var templatePath = GetTemplatePath(_currentTemplateType);
                var matcher = _currentTemplateType == TemplateType.Dark ? _darkTemplateMatcher : _lightTemplateMatcher;
                var statusLabel = _currentTemplateType == TemplateType.Dark ? lblDarkTemplateStatus : lblLightTemplateStatus;

                // Extract template
                int halfSize = _selectionSize / 2;
                int x = (int)Math.Max(0, _selectionCenter.X - halfSize);
                int y = (int)Math.Max(0, _selectionCenter.Y - halfSize);
                int w = Math.Min(_selectionSize, _sourceImage.Width - x);
                int h = Math.Min(_selectionSize, _sourceImage.Height - y);

                _sourceImage.ROI = new System.Drawing.Rectangle(x, y, w, h);
                var cropped = _sourceImage.Clone();
                _sourceImage.ROI = System.Drawing.Rectangle.Empty;

                // Resize to standard size
                var resized = new Image<Gray, byte>(100, 100);
                CvInvoke.Resize(cropped, resized, new System.Drawing.Size(100, 100));

                // *** 關鍵：二值化範本 ***
                // 使用 Otsu 自動閾值二值化
                var binarized = new Image<Gray, byte>(100, 100);
                CvInvoke.Threshold(resized, binarized, 0, 255,
                    Emgu.CV.CvEnum.ThresholdType.Binary | Emgu.CV.CvEnum.ThresholdType.Otsu);

                // 儲存二值化後的範本
                binarized.Save(templatePath);

                matcher.LoadTemplate(templatePath);
                statusLabel.Text = "已儲存(二值化)";
                lblTemplateInfo.Text = $"{typeName}範本已儲存(二值化)";

                await DisplayAlert("成功", $"{typeName}箭頭範本已二值化並儲存至:\n{templatePath}", "確定");
            }
            catch (Exception ex)
            {
                await DisplayAlert("錯誤", $"儲存範本失敗: {ex.Message}", "確定");
            }
#else
            await DisplayAlert("不支援", "此功能僅適用於 Android 和 Windows", "確定");
#endif
        }

        private async void OnBackClicked(object sender, EventArgs e)
        {
            await Shell.Current.GoToAsync("..");
        }

#if ANDROID || WINDOWS
        private Image<Gray, byte> ConvertSKBitmapToEmguGray(SKBitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            var image = new Image<Gray, byte>(width, height);

            // Use direct byte access for faster conversion
            IntPtr pixelsPtr = bitmap.GetPixels();
            int bytesPerPixel = bitmap.BytesPerPixel;
            int rowBytes = bitmap.RowBytes;

            unsafe
            {
                byte* srcPtr = (byte*)pixelsPtr.ToPointer();

                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = srcPtr + y * rowBytes;
                    for (int x = 0; x < width; x++)
                    {
                        int offset = x * bytesPerPixel;
                        byte b = rowPtr[offset];
                        byte g = rowPtr[offset + 1];
                        byte r = rowPtr[offset + 2];
                        byte gray = (byte)(r * 0.299 + g * 0.587 + b * 0.114);
                        image.Data[y, x, 0] = gray;
                    }
                }
            }

            return image;
        }
#endif
    }
}
