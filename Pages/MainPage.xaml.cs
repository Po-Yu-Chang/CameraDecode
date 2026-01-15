using CameraMaui.RingCode;
using CameraMaui.Services;
using Emgu.CV;
using Emgu.CV.Structure;
using Newtonsoft.Json;
using SkiaSharp;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;

namespace CameraMaui.Pages
{
    public partial class MainPage : ContentPage
    {
        private readonly IDeviceOrentationService _deviceOrientationService;
        private readonly RingCodeDecoder _ringCodeDecoder;
        private readonly RingImageSegmentation _imageSegmentation;

        private SKBitmap _currentImage;
        private Image<Bgr, byte> _currentEmguImage;
        private double _currentZoom = 1.0;
        private bool _isSourceTabActive = true;

        // Log file path
        private static readonly string LogFilePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "CameraMaui_Log.txt");

        private static void Log(string message)
        {
            try
            {
                string logLine = $"[{DateTime.Now:HH:mm:ss.fff}] {message}";
                File.AppendAllText(LogFilePath, logLine + Environment.NewLine);
                System.Diagnostics.Debug.WriteLine(logLine);
            }
            catch { }
        }

        // Parameterless constructor for Shell DataTemplate
        public MainPage() : this(ServiceHelper.GetService<IDeviceOrentationService>())
        {
        }

        public MainPage(IDeviceOrentationService deviceOrientationService)
        {
            InitializeComponent();
            _deviceOrientationService = deviceOrientationService;
            _ringCodeDecoder = new RingCodeDecoder();
            _imageSegmentation = new RingImageSegmentation();

            // Connect decoder and segmentation logs to our log
            RingCodeDecoder.Log = Log;
            RingCodeDecoder.EnableDetailedLog = false;  // Set to true for detailed per-segment logs
            RingImageSegmentation.Log = Log;

            // Initialize arrow templates for decoder
            _ringCodeDecoder.InitializeTemplates();
            Log($"Templates loaded - Dark: {_ringCodeDecoder.HasDarkTemplate}, Light: {_ringCodeDecoder.HasLightTemplate}");

            // Test orientation service
            var orientation = _deviceOrientationService?.GetOrentation() ?? DeviceOrientation.Undefined;
            System.Diagnostics.Debug.WriteLine($"Device Orientation: {orientation}");
        }

        private void cameraView_CamerasLoaded(object sender, EventArgs e)
        {
            var cameras = cameraView.Cameras;

            // Check if cameras are available
            if (cameras == null || !cameras.Any())
            {
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    MyLabel.Text = "No camera detected. Use 'Select Image' or 'Test Sample' to test.";
                    lblStatus.Text = "Status: No camera - use image selection";
                });
                return;
            }

            Global.cameraView.Camera = cameraView.Camera = cameras.First();

            MainThread.BeginInvokeOnMainThread(async () =>
            {
                await cameraView.StopCameraAsync();
                cameraView.ForceAutoFocus();
                var result = await cameraView.StartCameraAsync();
                lblStatus.Text = "Status: Camera ready";
            });
        }

        /// <summary>
        /// Capture image from camera
        /// </summary>
        private async void OnCaptureClicked(object sender, EventArgs e)
        {
            try
            {
                lblStatus.Text = "Status: Capturing...";
                MyLabel.Text = "Capturing from camera...";

                var imageSource = cameraView.GetSnapShot(Camera.MAUI.ImageFormat.JPEG);

                if (imageSource == null)
                {
                    await DisplayAlert("Error", "Failed to capture image. Camera may not be available.", "OK");
                    lblStatus.Text = "Status: Capture failed";
                    MyLabel.Text = "Camera capture returned null";
                    return;
                }

                _currentImage = await ConvertImageSourceToSKBitmap(imageSource);
                _currentEmguImage = ConvertSKBitmapToEmguImage(_currentImage);

                await SetSourceImage(_currentImage);
                lblStatus.Text = "Status: Image captured - Ready to analyze";
            }
            catch (Exception ex)
            {
                lblStatus.Text = $"Status: Error - {ex.Message}";
                MyLabel.Text = ex.Message;
            }
        }

        /// <summary>
        /// Select image from gallery/file
        /// </summary>
        private async void OnSelectImageClicked(object sender, EventArgs e)
        {
            try
            {
                lblStatus.Text = "Selecting...";
                btnSelectImage.IsEnabled = false;

                var result = await FilePicker.PickAsync(new PickOptions
                {
                    PickerTitle = "Select Ring Code Image",
                    FileTypes = FilePickerFileType.Images
                });

                if (result == null)
                {
                    lblStatus.Text = "Cancelled";
                    return;
                }

                string fullPath = result.FullPath;

                if (!string.IsNullOrEmpty(fullPath) && File.Exists(fullPath))
                {
                    await LoadImageFromPath(fullPath);
                }
                else
                {
                    // Fallback to stream-based loading
                    lblStatus.Text = "Loading...";
                    await Task.Run(async () =>
                    {
                        using var stream = await result.OpenReadAsync();
                        _currentImage = SKBitmap.Decode(stream);

                        if (_currentImage == null)
                        {
                            throw new Exception("Failed to decode image");
                        }

                        _currentEmguImage = ConvertSKBitmapToEmguImage(_currentImage);
                    });

                    await SetSourceImageFast(_currentImage);
                    lblStatus.Text = $"Loaded ({_currentImage.Width}x{_currentImage.Height})";
                }
            }
            catch (Exception ex)
            {
                lblStatus.Text = $"Error: {ex.Message}";
                Log($"OnSelectImageClicked Error: {ex}");
                await DisplayAlert("Error", ex.Message, "OK");
            }
            finally
            {
                btnSelectImage.IsEnabled = true;
            }
        }

        /// <summary>
        /// Load image directly from file path (handles special characters like [R])
        /// Optimized: Uses SKCodec subsampling for ultra-fast loading of large images
        /// </summary>
        private async Task LoadImageFromPath(string filePath)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                lblStatus.Text = "Loading...";
                string fileName = Path.GetFileName(filePath);
                Log($"LoadImageFromPath: {filePath}");

                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"File not found: {filePath}");
                }

                string tempPath = Path.Combine(FileSystem.CacheDirectory, $"source_{DateTime.Now.Ticks}.jpg");

                await Task.Run(async () =>
                {
                    // Read file bytes
                    byte[] imageBytes = await File.ReadAllBytesAsync(filePath);
                    Log($"Read {imageBytes.Length / 1024} KB in {sw.ElapsedMilliseconds}ms");

                    // Decode full image first
                    using var memoryStream = new MemoryStream(imageBytes);
                    var fullBitmap = SKBitmap.Decode(memoryStream);
                    if (fullBitmap == null)
                    {
                        throw new Exception("Failed to decode image");
                    }

                    Log($"Decoded: {fullBitmap.Width}x{fullBitmap.Height} in {sw.ElapsedMilliseconds}ms");

                    // Resize for analysis if too large (target ~2000px)
                    const int analysisSize = 2000;
                    if (fullBitmap.Width > analysisSize || fullBitmap.Height > analysisSize)
                    {
                        float scale = Math.Min((float)analysisSize / fullBitmap.Width,
                                               (float)analysisSize / fullBitmap.Height);
                        var resizeInfo = new SKImageInfo(
                            (int)(fullBitmap.Width * scale),
                            (int)(fullBitmap.Height * scale),
                            SKColorType.Bgra8888, SKAlphaType.Premul);
                        _currentImage = fullBitmap.Resize(resizeInfo, SKFilterQuality.Medium);
                        fullBitmap.Dispose();
                    }
                    else
                    {
                        _currentImage = fullBitmap;
                    }

                    Log($"Analysis size: {_currentImage.Width}x{_currentImage.Height} in {sw.ElapsedMilliseconds}ms");

                    // Convert to Emgu for analysis
                    _currentEmguImage = ConvertSKBitmapToEmguImage(_currentImage);
                    Log($"Emgu conversion done in {sw.ElapsedMilliseconds}ms");

                    // Create display thumbnail
                    const int displaySize = 1200;
                    SKBitmap displayBitmap = _currentImage;
                    if (_currentImage.Width > displaySize || _currentImage.Height > displaySize)
                    {
                        float scale = Math.Min((float)displaySize / _currentImage.Width,
                                               (float)displaySize / _currentImage.Height);
                        var resizeInfo = new SKImageInfo(
                            (int)(_currentImage.Width * scale),
                            (int)(_currentImage.Height * scale),
                            SKColorType.Bgra8888, SKAlphaType.Premul);
                        displayBitmap = _currentImage.Resize(resizeInfo, SKFilterQuality.Medium);
                    }

                    // Save display image
                    using var fs = File.OpenWrite(tempPath);
                    displayBitmap.Encode(fs, SKEncodedImageFormat.Jpeg, 85);
                    if (displayBitmap != _currentImage) displayBitmap.Dispose();

                    Log($"Total: {sw.ElapsedMilliseconds}ms");
                });

                // Update UI
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    sourceImage.Source = ImageSource.FromFile(tempPath);
                    sourceImage.WidthRequest = -1;
                    sourceImage.HeightRequest = -1;
                    _currentZoom = 1.0;
                    lblZoom.Text = "Fit";
                    lblStatus.Text = $"{_currentImage.Width}x{_currentImage.Height} ({sw.ElapsedMilliseconds}ms)";
                    MyLabel.Text = fileName;
                });
            }
            catch (Exception ex)
            {
                lblStatus.Text = $"Error: {ex.Message}";
                MyLabel.Text = ex.Message;
                Log($"LoadImageFromPath Error: {ex}");
                throw;
            }
        }

        /// <summary>
        /// Fast image display - creates thumbnail for display, keeps original for analysis
        /// Uses scaled decoding for better performance
        /// </summary>
        private async Task SetSourceImageFast(SKBitmap originalBitmap)
        {
            Log("=== SetSourceImageFast START ===");

            if (originalBitmap == null)
            {
                Log("ERROR: bitmap is null");
                return;
            }

            try
            {
                const int maxDisplaySize = 1200;
                string tempPath = Path.Combine(FileSystem.CacheDirectory, $"source_{DateTime.Now.Ticks}.jpg");

                await Task.Run(() =>
                {
                    SKBitmap displayBitmap;
                    bool needsResize = originalBitmap.Width > maxDisplaySize || originalBitmap.Height > maxDisplaySize;

                    if (needsResize)
                    {
                        // Calculate scaled size
                        float scale = Math.Min((float)maxDisplaySize / originalBitmap.Width,
                                               (float)maxDisplaySize / originalBitmap.Height);
                        int newWidth = (int)(originalBitmap.Width * scale);
                        int newHeight = (int)(originalBitmap.Height * scale);
                        Log($"Resizing: {originalBitmap.Width}x{originalBitmap.Height} -> {newWidth}x{newHeight}");

                        // Fast resize using SKBitmap.Resize with new API
                        var resizeInfo = new SKImageInfo(newWidth, newHeight, SKColorType.Bgra8888, SKAlphaType.Premul);
                        displayBitmap = originalBitmap.Resize(resizeInfo, SKFilterQuality.Medium);
                    }
                    else if (originalBitmap.ColorType != SKColorType.Bgra8888)
                    {
                        // Convert color type
                        displayBitmap = new SKBitmap(originalBitmap.Width, originalBitmap.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
                        using var canvas = new SKCanvas(displayBitmap);
                        canvas.DrawBitmap(originalBitmap, 0, 0);
                    }
                    else
                    {
                        displayBitmap = originalBitmap;
                    }

                    // Encode to JPEG (fast)
                    using var fileStream = File.OpenWrite(tempPath);
                    displayBitmap.Encode(fileStream, SKEncodedImageFormat.Jpeg, 80);

                    if (displayBitmap != originalBitmap)
                    {
                        displayBitmap.Dispose();
                    }

                    Log($"Saved thumbnail: {new FileInfo(tempPath).Length / 1024} KB");
                });

                // Update UI
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    sourceImage.Source = ImageSource.FromFile(tempPath);
                    sourceImage.WidthRequest = -1;
                    sourceImage.HeightRequest = -1;
                    _currentZoom = 1.0;
                    lblZoom.Text = "Fit";
                });

                Log("=== SetSourceImageFast END ===");
            }
            catch (Exception ex)
            {
                Log($"ERROR in SetSourceImageFast: {ex}");
                throw;
            }
        }

        /// <summary>
        /// Set source image safely on UI thread - using temp file approach
        /// Optimized: Resize large images for display, keep original for analysis
        /// </summary>
        private async Task SetSourceImage(SKBitmap bitmap)
        {
            Log("=== SetSourceImage START ===");

            if (bitmap == null)
            {
                Log("ERROR: bitmap is null");
                MyLabel.Text = "Error: bitmap is null";
                return;
            }

            Log($"Input bitmap: {bitmap.Width}x{bitmap.Height}, ColorType: {bitmap.ColorType}, ByteCount: {bitmap.ByteCount}");

            try
            {
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    lblStatus.Text = "Status: Loading image...";
                });

                // For display: resize large images to max 1200px for faster loading
                const int maxDisplaySize = 1200;
                SKBitmap displayBitmap = bitmap;
                bool needsResize = bitmap.Width > maxDisplaySize || bitmap.Height > maxDisplaySize;

                if (needsResize)
                {
                    float scale = Math.Min((float)maxDisplaySize / bitmap.Width, (float)maxDisplaySize / bitmap.Height);
                    int newWidth = (int)(bitmap.Width * scale);
                    int newHeight = (int)(bitmap.Height * scale);
                    Log($"Resizing for display: {bitmap.Width}x{bitmap.Height} -> {newWidth}x{newHeight}");

                    displayBitmap = new SKBitmap(newWidth, newHeight, SKColorType.Bgra8888, SKAlphaType.Premul);
                    using (var canvas = new SKCanvas(displayBitmap))
                    {
                        canvas.DrawBitmap(bitmap, new SKRect(0, 0, newWidth, newHeight));
                    }
                }
                else if (bitmap.ColorType != SKColorType.Bgra8888)
                {
                    // Convert to Bgra8888 if needed
                    Log($"Converting from {bitmap.ColorType} to Bgra8888...");
                    displayBitmap = new SKBitmap(bitmap.Width, bitmap.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
                    using (var canvas = new SKCanvas(displayBitmap))
                    {
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }
                }

                Log($"Display bitmap: {displayBitmap.Width}x{displayBitmap.Height}, ColorType: {displayBitmap.ColorType}");

                // Save to temp file using JPEG (faster, smaller)
                string cacheDir = FileSystem.CacheDirectory;
                string tempPath = Path.Combine(cacheDir, $"source_{DateTime.Now.Ticks}.jpg");
                Log($"TempPath: {tempPath}");

                using (var fileStream = File.OpenWrite(tempPath))
                {
                    Log("Encoding to JPEG...");
                    bool encoded = displayBitmap.Encode(fileStream, SKEncodedImageFormat.Jpeg, 85);
                    Log($"Encode result: {encoded}");

                    if (!encoded)
                    {
                        Log("ERROR: Failed to encode bitmap");
                        MyLabel.Text = "Error: Failed to encode bitmap";
                        return;
                    }
                }

                // Dispose display bitmap if we created one
                if (displayBitmap != bitmap)
                {
                    displayBitmap.Dispose();
                }

                // Verify file was created
                if (File.Exists(tempPath))
                {
                    var fileInfo = new FileInfo(tempPath);
                    Log($"File created: {tempPath}, Size: {fileInfo.Length / 1024} KB");
                }

                // Set on UI thread using FileImageSource
                Log("Setting ImageSource on UI thread...");
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    var imgSource = ImageSource.FromFile(tempPath);
                    sourceImage.Source = imgSource;
                    lblStatus.Text = "Status: Image loaded - Ready to analyze";
                });

                // Auto-fit: set image to fit within display area
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    // Reset to auto-size (AspectFit will handle fitting)
                    sourceImage.WidthRequest = -1;
                    sourceImage.HeightRequest = -1;
                    _currentZoom = 1.0;
                    lblZoom.Text = "Fit";
                });

                MyLabel.Text = $"Loaded: {bitmap.Width}x{bitmap.Height}" + (needsResize ? " (縮圖顯示)" : "");
                Log("=== SetSourceImage END ===");
            }
            catch (Exception ex)
            {
                Log($"ERROR in SetSourceImage: {ex}");
                MyLabel.Text = $"SetSourceImage error: {ex.Message}";
                throw;
            }
        }

        /// <summary>
        /// Manually enter a file path to load (for paths with special characters)
        /// </summary>
        private async void OnLoadPathClicked(object sender, EventArgs e)
        {
            try
            {
                string inputPath = await DisplayPromptAsync(
                    "Load Image",
                    "Enter the full path to the image file:",
                    initialValue: @"C:\Users\qoose\Desktop\文件資料\客戶分類\R-RCM\01_Software\RCM\03_Document\NG\20250707\20250700020033288.png",
                    maxLength: 500,
                    keyboard: Keyboard.Default);

                if (string.IsNullOrWhiteSpace(inputPath))
                {
                    lblStatus.Text = "Status: No path entered";
                    return;
                }

                await LoadImageFromPath(inputPath.Trim());
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", ex.Message, "OK");
            }
        }

        /// <summary>
        /// Analyze current image for ring codes (supports multi-ring detection)
        /// </summary>
        private async void OnAnalyzeClicked(object sender, EventArgs e)
        {
            if (_currentEmguImage == null)
            {
                await DisplayAlert("No Image", "Please capture or select an image first.", "OK");
                return;
            }

            try
            {
                lblStatus.Text = "Status: Analyzing...";
                btnAnalyze.IsEnabled = false;
                Log("=== OnAnalyzeClicked START ===");

                List<RingCodeDecoder.RingCodeResult> allResults = null;
                RingImageSegmentation.SegmentationResult segmentResult = null;

                var totalSw = System.Diagnostics.Stopwatch.StartNew();
                await Task.Run(() =>
                {
                    // Step 1: Segment the image to find all rings
                    var segSw = System.Diagnostics.Stopwatch.StartNew();
                    Log("Step 1: Segmenting image...");
                    segmentResult = _imageSegmentation.SegmentImage(_currentEmguImage);
                    Log($"Segmentation: {segSw.ElapsedMilliseconds}ms, Rings found: {segmentResult.DetectedRings.Count}");

                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        MyLabel.Text = segmentResult.Message;
                    });

                    if (!segmentResult.Success || segmentResult.DetectedRings.Count == 0)
                    {
                        MainThread.BeginInvokeOnMainThread(() =>
                        {
                            lblStatus.Text = "Status: No ring codes detected";
                            lblBinary.Text = "-";
                            lblDecoded.Text = "Not found";
                            lblDetails.Text = "Try adjusting image or use a clearer photo";
                        });
                        return;
                    }

                    // Step 2: Decode all detected rings
                    var decodeSw = System.Diagnostics.Stopwatch.StartNew();
                    Log("Step 2: Decoding all rings...");
                    allResults = _ringCodeDecoder.DecodeAllRings(_currentEmguImage);
                    Log($"Decoding: {decodeSw.ElapsedMilliseconds}ms for {allResults.Count} ring(s)");
                });
                Log($"Total analysis time: {totalSw.ElapsedMilliseconds}ms");

                // Step 3: Display results on UI thread
                if (allResults != null && allResults.Count > 0)
                {
                    await MainThread.InvokeOnMainThreadAsync(() =>
                    {
                        // Count valid and invalid results
                        int validCount = allResults.Count(r => r.IsValid);
                        int invalidCount = allResults.Count - validCount;

                        // Update ring count badge
                        lblRingCount.Text = allResults.Count.ToString();
                        lblStatus.Text = $"Found {allResults.Count} ring(s), {validCount} valid";

                        // Display first ring's binary (or combined if multiple)
                        if (allResults.Count == 1)
                        {
                            var result = allResults[0];
                            lblBinary.Text = result.BinaryString;

                            // Build error message including template match errors
                            if (result.IsValid)
                            {
                                lblDecoded.Text = result.DecodedData;
                            }
                            else
                            {
                                var errorMsg = !string.IsNullOrEmpty(result.ErrorMessage)
                                    ? result.ErrorMessage
                                    : "解碼失敗";

                                // Add template match error if present
                                if (!string.IsNullOrEmpty(result.TemplateMatchError))
                                {
                                    errorMsg += $"\n範本比對: {result.TemplateMatchError}";
                                }
                                lblDecoded.Text = $"錯誤: {errorMsg}";
                            }

                            lblDetails.Text = $"Center: ({result.Center.X:F0}, {result.Center.Y:F0}), " +
                                            $"R: {result.OuterRadius:F0}, Angle: {result.RotationAngle:F1}°";
                        }
                        else
                        {
                            // Multiple rings - show summary
                            var decodedValues = allResults
                                .Where(r => r.IsValid)
                                .Select(r => $"#{r.RingIndex}: {r.DecodedData}")
                                .ToList();

                            // Check for any template match errors in invalid results
                            var templateErrors = allResults
                                .Where(r => !r.IsValid && !string.IsNullOrEmpty(r.TemplateMatchError))
                                .Select(r => $"#{r.RingIndex}: {r.TemplateMatchError}")
                                .ToList();

                            lblBinary.Text = $"({allResults.Count} rings detected)";

                            if (decodedValues.Count > 0)
                            {
                                lblDecoded.Text = string.Join(" | ", decodedValues);
                            }
                            else
                            {
                                var errorText = "無有效解碼結果";
                                if (templateErrors.Count > 0)
                                {
                                    errorText += $"\n範本比對錯誤: {string.Join("; ", templateErrors)}";
                                }
                                lblDecoded.Text = errorText;
                            }

                            lblDetails.Text = $"Total: {allResults.Count} rings, Valid: {validCount}, Invalid: {invalidCount}";
                        }

                        // Create and display combined visualization
                        var visualization = _ringCodeDecoder.CreateCombinedVisualization(_currentEmguImage.Clone(), allResults);
                        var processedSKBitmap = ConvertEmguImageToSKBitmap(visualization);
                        processedImage.Source = ConvertSkBitmapToImageSource(processedSKBitmap);

                        // Switch to Result tab to show the visualization
                        SwitchToTab(false);

                        // Log detailed results
                        foreach (var result in allResults)
                        {
                            Log($"Ring #{result.RingIndex}: Valid={result.IsValid}, Data={result.DecodedData}, Binary={result.BinaryString}");
                        }
                    });
                }
                else
                {
                    // No results - update UI
                    await MainThread.InvokeOnMainThreadAsync(() =>
                    {
                        lblRingCount.Text = "0";
                    });
                }

                Log("=== OnAnalyzeClicked END ===");
            }
            catch (Exception ex)
            {
                Log($"ERROR in OnAnalyzeClicked: {ex}");
                lblStatus.Text = $"Status: Error - {ex.Message}";
                await DisplayAlert("Analysis Error", ex.ToString(), "OK");
            }
            finally
            {
                btnAnalyze.IsEnabled = true;
            }
        }

        /// <summary>
        /// Clear current results
        /// </summary>
        private void OnClearClicked(object sender, EventArgs e)
        {
            _currentImage?.Dispose();
            _currentImage = null;
            _currentEmguImage?.Dispose();
            _currentEmguImage = null;

            sourceImage.Source = null;
            processedImage.Source = null;

            lblStatus.Text = "Ready";
            lblBinary.Text = "-";
            lblDecoded.Text = "-";
            lblDetails.Text = "";
            MyLabel.Text = "";
            lblRingCount.Text = "0";

            // Switch back to Source tab
            SwitchToTab(true);

            // Reset zoom
            _currentZoom = 1.0;
            UpdateZoom();
        }

        #region Tab Controls

        private void OnTabSourceClicked(object sender, EventArgs e)
        {
            SwitchToTab(true);
        }

        private void OnTabResultClicked(object sender, EventArgs e)
        {
            SwitchToTab(false);
        }

        private void SwitchToTab(bool isSourceTab)
        {
            _isSourceTabActive = isSourceTab;

            // Update tab button appearance (use new color scheme)
            tabSource.BackgroundColor = isSourceTab ? Color.FromArgb("#F9C846") : Color.FromArgb("#2B8B8B");
            tabSource.TextColor = isSourceTab ? Color.FromArgb("#1E4B5C") : Color.FromArgb("#F5EFE0");

            tabResult.BackgroundColor = !isSourceTab ? Color.FromArgb("#F9C846") : Color.FromArgb("#2B8B8B");
            tabResult.TextColor = !isSourceTab ? Color.FromArgb("#1E4B5C") : Color.FromArgb("#F5EFE0");

            // Toggle visibility - need to toggle both Frame and ScrollView
            sourceScrollView.IsVisible = isSourceTab;
            resultFrame.IsVisible = !isSourceTab;
            resultScrollView.IsVisible = !isSourceTab;

            // Reset zoom for the active tab
            _currentZoom = 1.0;
            UpdateZoom();
        }

        #endregion

        #region Zoom Controls

        private void OnZoomInClicked(object sender, EventArgs e)
        {
            _currentZoom = Math.Min(_currentZoom * 1.25, 5.0); // Max 500%
            UpdateZoom();
        }

        private void OnZoomOutClicked(object sender, EventArgs e)
        {
            _currentZoom = Math.Max(_currentZoom / 1.25, 0.25); // Min 25%
            UpdateZoom();
        }

        private void OnZoomResetClicked(object sender, EventArgs e)
        {
            _currentZoom = 1.0;
            UpdateZoom();
        }

        private void UpdateZoom()
        {
            // Get the active image control
            var activeImage = _isSourceTabActive ? sourceImage : processedImage;

            if (_currentImage != null && _currentZoom != 1.0)
            {
                // Use display size (max 1200) for zoom calculation
                int displayWidth = Math.Min(_currentImage.Width, 1200);
                int displayHeight = Math.Min(_currentImage.Height, 1200);

                activeImage.WidthRequest = displayWidth * _currentZoom;
                activeImage.HeightRequest = displayHeight * _currentZoom;
                lblZoom.Text = $"{(int)(_currentZoom * 100)}%";
            }
            else
            {
                // Fit mode - let AspectFit handle sizing
                activeImage.WidthRequest = -1;
                activeImage.HeightRequest = -1;
                lblZoom.Text = "Fit";
            }
        }

        #endregion

        /// <summary>
        /// Test with a generated sample ring code image
        /// </summary>
        private async void OnTestWithSampleClicked(object sender, EventArgs e)
        {
            Log("=== OnTestWithSampleClicked START ===");
            try
            {
                lblStatus.Text = "Status: Generating sample...";
                MyLabel.Text = "Creating sample ring code...";
                Log("Creating sample ring code image...");

                // Create a sample ring code image for testing
                _currentImage = CreateSampleRingCodeImage();
                Log($"Sample created: {_currentImage?.Width}x{_currentImage?.Height}, ColorType: {_currentImage?.ColorType}");

                _currentEmguImage = ConvertSKBitmapToEmguImage(_currentImage);
                Log($"Converted to Emgu: {_currentEmguImage?.Width}x{_currentEmguImage?.Height}");

                await SetSourceImage(_currentImage);
                lblStatus.Text = "Status: Sample generated - Ready to analyze";

                MyLabel.Text = "Sample ring code image generated. Click 'Analyze' to decode.";
                Log("=== OnTestWithSampleClicked END ===");
            }
            catch (Exception ex)
            {
                Log($"ERROR in OnTestWithSampleClicked: {ex}");
                lblStatus.Text = $"Status: Error - {ex.Message}";
                MyLabel.Text = $"Error: {ex.Message}";
                await DisplayAlert("Error", ex.Message, "OK");
            }
        }

        /// <summary>
        /// Open the Arrow Template Creator page
        /// </summary>
        private async void OnTemplateCreatorClicked(object sender, EventArgs e)
        {
            Log("Opening Template Creator page...");
            try
            {
                await Shell.Current.GoToAsync(nameof(TemplateCreatorPage));
            }
            catch (Exception ex)
            {
                Log($"ERROR opening Template Creator: {ex}");
                await DisplayAlert("Error", $"Failed to open Template Creator: {ex.Message}", "OK");
            }
        }

        /// <summary>
        /// Create a sample ring code image for testing
        /// </summary>
        private SKBitmap CreateSampleRingCodeImage()
        {
            int size = 400;
            var bitmap = new SKBitmap(size, size, SKColorType.Bgra8888, SKAlphaType.Premul);
            using var canvas = new SKCanvas(bitmap);

            // White background
            canvas.Clear(SKColors.White);

            float centerX = size / 2f;
            float centerY = size / 2f;
            float outerRadius = 150;
            float innerRadius = 50;
            float middleRadius = (outerRadius + innerRadius) / 2;

            // Draw outer circle (black outline)
            using var blackPaint = new SKPaint { Color = SKColors.Black, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
            using var fillPaint = new SKPaint { Color = SKColors.Black, Style = SKPaintStyle.Fill };
            using var whitePaint = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };

            canvas.DrawCircle(centerX, centerY, outerRadius, blackPaint);
            canvas.DrawCircle(centerX, centerY, innerRadius, blackPaint);
            canvas.DrawCircle(centerX, centerY, middleRadius, blackPaint);

            // Draw sample data segments (24 segments, inner and outer)
            // Sample binary: alternating pattern for testing
            string sampleBinary = "101010101010101010101010101010101010101010101010";
            float segmentAngle = 360f / 24;

            for (int i = 0; i < 24; i++)
            {
                float startAngle = i * segmentAngle - 90; // Start from top

                // Inner segment
                if (sampleBinary[i * 2] == '1')
                {
                    DrawSegment(canvas, centerX, centerY, innerRadius + 5, middleRadius - 5, startAngle, segmentAngle - 2, fillPaint);
                }

                // Outer segment
                if (sampleBinary[i * 2 + 1] == '1')
                {
                    DrawSegment(canvas, centerX, centerY, middleRadius + 5, outerRadius - 5, startAngle, segmentAngle - 2, fillPaint);
                }
            }

            // Draw locator triangle (at position 0)
            float triangleAngle = -90 * (float)Math.PI / 180; // Top
            float triangleRadius = (middleRadius + outerRadius) / 2;
            float triangleSize = 15;

            var trianglePath = new SKPath();
            float tx = centerX + triangleRadius * (float)Math.Cos(triangleAngle);
            float ty = centerY + triangleRadius * (float)Math.Sin(triangleAngle);

            trianglePath.MoveTo(tx, ty - triangleSize);
            trianglePath.LineTo(tx - triangleSize, ty + triangleSize / 2);
            trianglePath.LineTo(tx + triangleSize, ty + triangleSize / 2);
            trianglePath.Close();
            canvas.DrawPath(trianglePath, fillPaint);

            return bitmap;
        }

        private void DrawSegment(SKCanvas canvas, float cx, float cy, float innerR, float outerR, float startAngle, float sweepAngle, SKPaint paint)
        {
            var path = new SKPath();

            // Outer arc
            var outerRect = new SKRect(cx - outerR, cy - outerR, cx + outerR, cy + outerR);
            path.ArcTo(outerRect, startAngle, sweepAngle, false);

            // Inner arc (reverse direction)
            float endAngle = startAngle + sweepAngle;
            float innerEndX = cx + innerR * (float)Math.Cos(endAngle * Math.PI / 180);
            float innerEndY = cy + innerR * (float)Math.Sin(endAngle * Math.PI / 180);
            path.LineTo(innerEndX, innerEndY);

            var innerRect = new SKRect(cx - innerR, cy - innerR, cx + innerR, cy + innerR);
            path.ArcTo(innerRect, endAngle, -sweepAngle, false);

            path.Close();
            canvas.DrawPath(path, paint);
        }

        #region Image Conversion Utilities

        private Image<Bgr, byte> ConvertSKBitmapToEmguImage(SKBitmap skBitmap)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // Ensure bitmap is in BGRA8888 format
            SKBitmap convertedBitmap = skBitmap;
            if (skBitmap.ColorType != SKColorType.Bgra8888)
            {
                convertedBitmap = new SKBitmap(skBitmap.Width, skBitmap.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
                using var canvas = new SKCanvas(convertedBitmap);
                canvas.DrawBitmap(skBitmap, 0, 0);
            }

            int width = convertedBitmap.Width;
            int height = convertedBitmap.Height;
            var image = new Image<Bgr, byte>(width, height);

            // Use unsafe pointer access for much faster conversion
            IntPtr pixelsPtr = convertedBitmap.GetPixels();
            int rowBytes = convertedBitmap.RowBytes;

            unsafe
            {
                byte* srcPtr = (byte*)pixelsPtr.ToPointer();

                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = srcPtr + y * rowBytes;
                    for (int x = 0; x < width; x++)
                    {
                        int offset = x * 4; // BGRA format
                        image.Data[y, x, 0] = rowPtr[offset];     // B
                        image.Data[y, x, 1] = rowPtr[offset + 1]; // G
                        image.Data[y, x, 2] = rowPtr[offset + 2]; // R
                    }
                }
            }

            // Dispose converted bitmap if we created a new one
            if (convertedBitmap != skBitmap)
            {
                convertedBitmap.Dispose();
            }

            Log($"ConvertSKBitmapToEmguImage: {width}x{height} in {sw.ElapsedMilliseconds}ms");
            return image;
        }

        private SKBitmap ConvertEmguImageToSKBitmap(Image<Bgr, byte> image)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            int width = image.Width;
            int height = image.Height;

            // Create bitmap with proper color space
            var info = new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Premul);
            var skBitmap = new SKBitmap(info);

            // Use unsafe pointer access for much faster conversion
            IntPtr pixelsPtr = skBitmap.GetPixels();
            int rowBytes = skBitmap.RowBytes;

            unsafe
            {
                byte* dstPtr = (byte*)pixelsPtr.ToPointer();

                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = dstPtr + y * rowBytes;
                    for (int x = 0; x < width; x++)
                    {
                        int offset = x * 4; // BGRA format
                        rowPtr[offset] = image.Data[y, x, 0];     // B
                        rowPtr[offset + 1] = image.Data[y, x, 1]; // G
                        rowPtr[offset + 2] = image.Data[y, x, 2]; // R
                        rowPtr[offset + 3] = 255;                  // A
                    }
                }
            }

            // Notify bitmap that pixels have changed
            skBitmap.NotifyPixelsChanged();

            Log($"ConvertEmguImageToSKBitmap: {width}x{height} in {sw.ElapsedMilliseconds}ms");
            return skBitmap;
        }

        public async Task<SKBitmap> ConvertImageSourceToSKBitmap(ImageSource imageSource)
        {
            if (imageSource is StreamImageSource streamImageSource)
            {
                using var stream = await streamImageSource.Stream(CancellationToken.None);
                return SKBitmap.Decode(stream);
            }
            throw new NotSupportedException("Only StreamImageSource is supported.");
        }

        public ImageSource ConvertSkBitmapToImageSource(SKBitmap skBitmap)
        {
            if (skBitmap == null)
                throw new ArgumentNullException(nameof(skBitmap));

            try
            {
                // Save to temp file (more reliable than stream)
                string tempPath = Path.Combine(FileSystem.CacheDirectory, $"result_{DateTime.Now.Ticks}.png");
                using (var fs = File.OpenWrite(tempPath))
                {
                    bool encoded = skBitmap.Encode(fs, SKEncodedImageFormat.Png, 100);
                    if (!encoded)
                    {
                        Log($"ERROR: Failed to encode bitmap {skBitmap.Width}x{skBitmap.Height}, ColorType={skBitmap.ColorType}");
                        // Try JPEG as fallback
                        fs.SetLength(0);
                        encoded = skBitmap.Encode(fs, SKEncodedImageFormat.Jpeg, 90);
                    }
                    if (!encoded)
                    {
                        throw new Exception($"Failed to encode bitmap: {skBitmap.Width}x{skBitmap.Height}, ColorType={skBitmap.ColorType}");
                    }
                }
                return ImageSource.FromFile(tempPath);
            }
            catch (Exception ex)
            {
                Log($"ERROR in ConvertSkBitmapToImageSource: {ex.Message}");
                throw;
            }
        }

        #endregion

        #region Legacy Methods (kept for compatibility)

        private async void Button_Clicked(object sender, EventArgs e)
        {
            // Redirect to new capture method
            OnCaptureClicked(sender, e);
        }

        private async Task<byte[]> ConvertImageSourceToByteArray(ImageSource imageSource)
        {
            if (imageSource is StreamImageSource streamImageSource)
            {
                var cancellationToken = default(CancellationToken);
                var stream = await streamImageSource.Stream(cancellationToken);
                if (stream != null)
                {
                    using var memoryStream = new MemoryStream();
                    await stream.CopyToAsync(memoryStream, 81920, cancellationToken);
                    return memoryStream.ToArray();
                }
            }
            return null;
        }

        public ImageSource ConvertBase64ToImageSource(string base64Image)
        {
            if (string.IsNullOrEmpty(base64Image))
                return null;

            byte[] imageBytes = Convert.FromBase64String(base64Image);
            var stream = new MemoryStream(imageBytes);
            return ImageSource.FromStream(() => stream);
        }

        public async Task SendImageToWebService(ImageSource imageSource)
        {
            var imageBytes = await ConvertImageSourceToByteArray(imageSource);
            if (imageBytes != null)
            {
                var base64Image = Convert.ToBase64String(imageBytes);
                var jsonContent = new StringContent(base64Image, Encoding.UTF8, "application/json");

                using var httpClient = new HttpClient();
                var response = await httpClient.PostAsync("http://192.168.0.179:5000/MyService/UploadImage", jsonContent);
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    sourceImage.Source = ConvertBase64ToImageSource(responseContent);
                }
            }
        }

        public string GenerateRandomNumber()
        {
            Random random = new Random();
            string number = "4";
            for (int i = 0; i < 11; i++)
            {
                number += random.Next(0, 10).ToString();
            }
            return number;
        }

        /// <summary>
        /// Test Emgu.CV functionality without camera
        /// </summary>
        private async void TestEmguCV_Clicked(object sender, EventArgs e)
        {
            try
            {
                MyLabel.Text = "Running Emgu.CV tests...";
                lblStatus.Text = "Status: Testing Emgu.CV...";

                var results = await EmguCVTest.RunAllTestsAsync();

                var testBitmap = EmguCVTest.CreateTestBitmap();

                if (_deviceOrientationService != null)
                {
                    var processedBitmap = (SKBitmap)_deviceOrientationService.GetImage(testBitmap);
                    processedImage.Source = ConvertSkBitmapToImageSource(processedBitmap);
                    results += "\n[PASS] Image processing via DI Service";
                }
                else
                {
                    results += "\n[WARN] DeviceOrientationService not available";
                }

                MyLabel.Text = results;
                lblStatus.Text = "Status: Emgu.CV test complete";
                await DisplayAlert("Emgu.CV Test", results, "OK");
            }
            catch (Exception ex)
            {
                MyLabel.Text = $"Error: {ex.Message}";
                lblStatus.Text = $"Status: Test failed - {ex.Message}";
                await DisplayAlert("Emgu.CV Test Failed", ex.ToString(), "OK");
            }
        }

        #endregion
    }
}
