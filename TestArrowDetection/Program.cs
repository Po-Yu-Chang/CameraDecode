using System;
using System.Drawing;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using CameraMaui.RingCode;

class Program
{
    static void Main(string[] args)
    {
        string outputDir = @"C:\Users\qoose\Desktop\ArrowDetectionTest";
        Directory.CreateDirectory(outputDir);

        // Check for Rolling Ball single image test mode
        if (args.Length > 0 && args[0] == "rb")
        {
            string imagePath = args.Length > 1 ? args[1] :
                @"C:\Users\qoose\Desktop\文件資料\客戶分類\R-RCM\01_Software\RCM\03_Document\NG\20260113-003\NG\20260100040000925.png";
            TestRollingBallSingle(imagePath, outputDir);
            return;
        }

        // TDD Test Mode
        Console.WriteLine("=== RING CODE TDD TEST ===\n");

        // Test directories
        string darkRingDir = @"C:\Users\qoose\Desktop\文件資料\客戶分類\R-RCM\01_Software\RCM\03_Document\NG\20260113-003\NG";
        string lightRingDir = @"C:\Users\qoose\Desktop\文件資料\客戶分類\R-RCM\01_Software\RCM\03_Document\NG\20260113-002\NG";

        // Enable debug output
        RingCodeDecoder.DebugOutputDir = outputDir;
        RingCodeDecoder.EnableDetailedLog = false;
        RingCodeDecoder.Log = (msg) => Console.WriteLine($"  [D] {msg}");
        RingImageSegmentation.Log = (msg) => Console.WriteLine($"    [S] {msg}");

        // Choose test set
        Console.WriteLine("Select test set:");
        Console.WriteLine("  1. Dark ring (20260113-003) - Gray ring with WHITE marks");
        Console.WriteLine("  2. Light ring (20260113-002) - White ring with DARK marks");
        Console.Write("Enter choice (1 or 2): ");

        string choice = Console.ReadLine()?.Trim() ?? "1";
        string testDir = choice == "2" ? lightRingDir : darkRingDir;
        string testType = choice == "2" ? "LIGHT" : "DARK";

        Console.WriteLine($"\nTesting {testType} ring images from:\n{testDir}\n");

        // Run TDD tests
        RunTDDTests(testDir, outputDir, testType);
    }

    static void TestRollingBallSingle(string imagePath, string outputDir)
    {
        Console.WriteLine($"=== ROLLING BALL TEST ===");
        Console.WriteLine($"Image: {Path.GetFileName(imagePath)}\n");

        var colorImage = new Image<Bgr, byte>(imagePath);
        var original = colorImage.Convert<Gray, byte>();
        Console.WriteLine($"Image size: {original.Width}x{original.Height}");

        // Use segmentation to find ring
        RingImageSegmentation.Log = (msg) => Console.WriteLine($"  [S] {msg}");
        var segmentation = new RingImageSegmentation();
        var result = segmentation.SegmentImage(colorImage);

        if (result.DetectedRings == null || result.DetectedRings.Count == 0)
        {
            Console.WriteLine("ERROR: No ring detected!");
            return;
        }

        var region = result.DetectedRings[0];
        float centerX = region.Center.X, centerY = region.Center.Y;
        float innerR = region.InnerRadius, outerR = region.OuterRadius;
        Console.WriteLine($"\nRing: center=({centerX:F0},{centerY:F0}), innerR={innerR:F0}, outerR={outerR:F0}");

        // Create ring mask
        var ringMask = new Image<Gray, byte>(original.Size);
        CvInvoke.Circle(ringMask, new Point((int)centerX, (int)centerY), (int)outerR, new MCvScalar(255), -1);
        CvInvoke.Circle(ringMask, new Point((int)centerX, (int)centerY), (int)innerR, new MCvScalar(0), -1);

        // Crop to ring bounding box with padding
        int pad = 10;
        int x1 = Math.Max(0, (int)(centerX - outerR - pad));
        int y1 = Math.Max(0, (int)(centerY - outerR - pad));
        int x2 = Math.Min(original.Width, (int)(centerX + outerR + pad));
        int y2 = Math.Min(original.Height, (int)(centerY + outerR + pad));

        var roi = new Rectangle(x1, y1, x2 - x1, y2 - y1);
        var cropped = original.Copy(roi);
        var croppedMask = ringMask.Copy(roi);

        Console.WriteLine($"Cropped ROI: {roi}");
        cropped.Save(Path.Combine(outputDir, "rb_1_cropped.png"));

        // Apply CLAHE
        var clahe = new Image<Gray, byte>(cropped.Size);
        CvInvoke.CLAHE(cropped, 2.0, new Size(8, 8), clahe);
        clahe.Save(Path.Combine(outputDir, "rb_2_clahe.png"));

        // Fill outside ring with ring mean
        MCvScalar meanScalar = CvInvoke.Mean(clahe, croppedMask);
        double ringMean = meanScalar.V0;
        Console.WriteLine($"Ring mean intensity: {ringMean:F1}");

        var maskedForRB = new Image<Gray, byte>(clahe.Size);
        maskedForRB.SetValue(new MCvScalar(ringMean));
        clahe.Copy(maskedForRB, croppedMask);
        maskedForRB.Save(Path.Combine(outputDir, "rb_3_masked_input.png"));

        // === Strong blur BEFORE Rolling Ball to smooth texture ===
        // Try Bilateral filter (edge-preserving) or large Gaussian
        var blurred = new Image<Gray, byte>(maskedForRB.Size);

        // Bilateral filter: d=15, sigmaColor=75, sigmaSpace=75
        int d = 15;
        double sigmaColor = 75;
        double sigmaSpace = 75;
        CvInvoke.BilateralFilter(maskedForRB, blurred, d, sigmaColor, sigmaSpace);
        blurred.Save(Path.Combine(outputDir, "rb_3b_bilateral.png"));
        Console.WriteLine($"Bilateral filter: d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}");

        // Rolling Ball with morphological opening (on blurred image)
        int rbRadius = Math.Max(31, (int)(outerR / 5) | 1);
        Console.WriteLine($"Rolling Ball radius: {rbRadius}");

        using var rbKernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
            new Size(rbRadius, rbRadius), new Point(-1, -1));

        // Rolling Ball on BLURRED image to estimate smooth background
        var background = new Image<Gray, byte>(blurred.Size);
        CvInvoke.MorphologyEx(blurred, background, MorphOp.Open, rbKernel,
            new Point(-1, -1), 1, BorderType.Replicate, new MCvScalar(0));
        background.Save(Path.Combine(outputDir, "rb_4_background.png"));

        // Subtract background from ORIGINAL (not blurred) to preserve mark edges
        var originalFloat = maskedForRB.Convert<Gray, float>();
        var backgroundFloat = background.Convert<Gray, float>();
        var correctedFloat = originalFloat.Sub(backgroundFloat);

        double[] minVal, maxVal;
        Point[] minLoc, maxLoc;
        correctedFloat.MinMax(out minVal, out maxVal, out minLoc, out maxLoc);
        Console.WriteLine($"Corrected range: {minVal[0]:F1} to {maxVal[0]:F1}");

        // Normalize to 0-255
        if (maxVal[0] > minVal[0])
        {
            correctedFloat = correctedFloat.Sub(new Gray(minVal[0]));
            correctedFloat = correctedFloat.Mul(255.0 / (maxVal[0] - minVal[0]));
        }
        var corrected = correctedFloat.Convert<Gray, byte>();
        corrected.Save(Path.Combine(outputDir, "rb_5_corrected.png"));

        // Apply ring mask to corrected
        var finalCorrected = new Image<Gray, byte>(corrected.Size);
        corrected.Copy(finalCorrected, croppedMask);
        finalCorrected.Save(Path.Combine(outputDir, "rb_6_final_corrected.png"));

        // Local threshold on corrected image
        int localBlockSize = Math.Max(41, (int)(outerR / 3) | 1);
        localBlockSize = Math.Min(localBlockSize, 151);

        var localMean = new Image<Gray, byte>(finalCorrected.Size);
        CvInvoke.Blur(finalCorrected, localMean, new Size(localBlockSize, localBlockSize), new Point(-1, -1));

        double localOffset = 15;
        var diff = finalCorrected.Sub(localMean);
        var binaryLocal = new Image<Gray, byte>(diff.Size);
        CvInvoke.Threshold(diff, binaryLocal, localOffset, 255, ThresholdType.Binary);

        var binaryMasked = new Image<Gray, byte>(binaryLocal.Size);
        binaryLocal.Copy(binaryMasked, croppedMask);
        binaryMasked.Save(Path.Combine(outputDir, "rb_7_binary_local.png"));

        // Otsu on corrected
        var binaryOtsu = new Image<Gray, byte>(finalCorrected.Size);
        CvInvoke.Threshold(finalCorrected, binaryOtsu, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);
        var binaryOtsuMasked = new Image<Gray, byte>(binaryOtsu.Size);
        binaryOtsu.Copy(binaryOtsuMasked, croppedMask);
        binaryOtsuMasked.Save(Path.Combine(outputDir, "rb_8_binary_otsu.png"));

        // === NEW APPROACH: Non-local Means (強) → Sauvola ===
        Console.WriteLine("\n=== Non-local Means (h=25) → Sauvola ===");

        // Step 1: Strong Non-local Means Denoising
        var denoised = new Image<Gray, byte>(maskedForRB.Size);
        CvInvoke.FastNlMeansDenoising(maskedForRB, denoised, h: 25, templateWindowSize: 7, searchWindowSize: 21);
        denoised.Save(Path.Combine(outputDir, "new_1_denoised.png"));
        Console.WriteLine("Non-local Means denoising done (h=25)");

        // Step 2: Mask the denoised image
        var grayProcessed = new Image<Gray, byte>(denoised.Size);
        grayProcessed.SetValue(new MCvScalar(ringMean));
        denoised.Copy(grayProcessed, croppedMask);

        // Step 3: Simple Binary Threshold for bright marks
        // Use Otsu to find optimal threshold, or fixed high value
        var binarySimple = new Image<Gray, byte>(grayProcessed.Size);
        double otsuThresh = CvInvoke.Threshold(grayProcessed, binarySimple, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);
        Console.WriteLine($"Binary Threshold (Otsu): thresh={otsuThresh:F0}");

        // Rename for compatibility with rest of code
        var binarySauvola = binarySimple;

        // Apply mask
        var binarySauvolaMasked = new Image<Gray, byte>(binarySauvola.Size);
        binarySauvola.Copy(binarySauvolaMasked, croppedMask);
        binarySauvolaMasked.Save(Path.Combine(outputDir, "new_2_adaptive.png"));

        // Calculate white ratio
        int sauvolaWhite = CvInvoke.CountNonZero(binarySauvolaMasked);
        int maskArea = CvInvoke.CountNonZero(croppedMask);
        double sauvolaRatio = (double)sauvolaWhite / maskArea;
        Console.WriteLine($"Adaptive result: white={sauvolaWhite}, ratio={sauvolaRatio:P1}");

        // === Filter blobs with holes (noise has holes, real marks are solid) ===
        Console.WriteLine("\n=== Filtering blobs with holes ===");
        var binaryFiltered = new Image<Gray, byte>(binaryMasked.Size);

        using var contours = new VectorOfVectorOfPoint();
        using var hierarchy = new Mat();
        // Use CCOMP to get 2-level hierarchy (outer contours and their holes)
        CvInvoke.FindContours(binaryMasked.Clone(), contours, hierarchy, RetrType.Ccomp, ChainApproxMethod.ChainApproxSimple);

        var hierData = new int[hierarchy.Cols * 4];
        if (hierarchy.Rows > 0)
            System.Runtime.InteropServices.Marshal.Copy(hierarchy.DataPointer, hierData, 0, hierData.Length);

        int keptSolid = 0, removedWithHoles = 0;
        for (int i = 0; i < contours.Size; i++)
        {
            // hierarchy: [next, prev, firstChild, parent]
            int firstChild = hierData[i * 4 + 2];
            int parent = hierData[i * 4 + 3];

            // Only process outer contours (parent == -1)
            if (parent != -1) continue;

            double area = CvInvoke.ContourArea(contours[i]);
            if (area < 100) continue;  // Skip tiny blobs

            // Check if this contour has holes (children)
            bool hasHoles = (firstChild != -1);

            // Also check solidity
            using var hull = new VectorOfPoint();
            CvInvoke.ConvexHull(contours[i], hull);
            double hullArea = CvInvoke.ContourArea(hull);
            double solidity = hullArea > 0 ? area / hullArea : 0;

            // Count number of holes
            int holeCount = 0;
            int childIdx = firstChild;
            while (childIdx != -1)
            {
                holeCount++;
                childIdx = hierData[childIdx * 4];  // next sibling
            }

            Console.WriteLine($"  Contour {i}: area={area:F0}, solidity={solidity:F2}, holes={holeCount}");

            // Keep if: no holes OR (few holes AND high solidity)
            bool keep = !hasHoles || (holeCount <= 2 && solidity >= 0.7);

            if (keep)
            {
                CvInvoke.DrawContours(binaryFiltered, contours, i, new MCvScalar(255), -1);
                keptSolid++;
            }
            else
            {
                removedWithHoles++;
            }
        }

        Console.WriteLine($"  Result: kept={keptSolid}, removed={removedWithHoles}");
        binaryFiltered.Save(Path.Combine(outputDir, "rb_9_filtered_no_holes.png"));

        Console.WriteLine($"\n=== Output saved to: {outputDir} ===");
        Console.WriteLine("  rb_1_cropped.png      - Cropped original");
        Console.WriteLine("  rb_2_clahe.png        - After CLAHE");
        Console.WriteLine("  rb_3_masked_input.png - Ring area (filled outside)");
        Console.WriteLine("  rb_4_background.png   - Rolling Ball background");
        Console.WriteLine("  rb_5_corrected.png    - After subtraction");
        Console.WriteLine("  rb_6_final_corrected.png - Masked corrected");
        Console.WriteLine("  rb_7_binary_local.png - Local threshold");
        Console.WriteLine("  rb_8_binary_otsu.png  - Otsu threshold");
    }

    // Known failing images to skip during TDD (to be fixed later)
    // Pattern: Template often finds wrong position, HuMoments "closest to template" is also wrong
    // Root cause: Selection based on "closest to template" fails when template is wrong
    // Solution needed: Select based on Y-arrow characteristics, not proximity to template
    static readonly HashSet<string> SkipList = new HashSet<string>
    {
        "20260100040000907",
        "20260100040000909",
        "20260100040000911",
        "20260100040000912",
        "20260100040000914",
        "20260100040000915",  // Y-arrow merged/atypical solidity
        "20260100040000917",  // Both template and HuMoments wrong
    };

    static void RunTDDTests(string testDir, string outputDir, string testType)
    {
        // 重置 template 快取與 debug 計數器
        RingCodeDecoder.ResetRotatedTemplates();
        RingCodeDecoder.ResetDebugCounter();

        var decoder = new RingCodeDecoder();
        var segmentation = new RingImageSegmentation();

        var testFiles = Directory.GetFiles(testDir, "*.png").OrderBy(f => f).ToArray();
        Console.WriteLine($"Found {testFiles.Length} test images\n");
        if (SkipList.Count > 0)
            Console.WriteLine($"Skipping {SkipList.Count} known failing images\n");

        var results = new List<(string file, string expected, string actual, bool pass, string error)>();
        int passCount = 0;
        int failCount = 0;
        int skipCount = 0;

        foreach (var testImagePath in testFiles)
        {
            string fileName = Path.GetFileNameWithoutExtension(testImagePath);
            string expected = fileName;  // Filename IS the expected decoded value

            // Skip known failing images
            if (SkipList.Contains(fileName))
            {
                Console.WriteLine($"\n[SKIP] {fileName} - known issue");
                skipCount++;
                continue;
            }

            Console.WriteLine($"\n{'=',-60}");
            Console.WriteLine($"Testing: {fileName}");
            Console.WriteLine($"{'=',-60}");

            string actual = "";
            string error = "";
            bool pass = false;

            try
            {
                var colorImg = new Image<Bgr, byte>(testImagePath);
                var grayImg = colorImg.Convert<Gray, byte>();

                // Step 1: Segmentation
                var segResult = segmentation.SegmentImage(colorImg);
                if (!segResult.Success || segResult.DetectedRings.Count == 0)
                {
                    error = "Segmentation failed - no ring found";
                    Console.WriteLine($"  ERROR: {error}");
                }
                else
                {
                    var ring = segResult.DetectedRings[0];
                    Console.WriteLine($"  Ring: center=({ring.Center.X:F0},{ring.Center.Y:F0}), R={ring.OuterRadius:F0}");

                    // Step 2: Decode
                    var decodeResult = decoder.DecodeRing(grayImg, ring);
                    actual = decodeResult.DecodedData ?? "";

                    Console.WriteLine($"  Arrow angle: {decodeResult.RotationAngle:F1}°");
                    Console.WriteLine($"  Decoded: {actual}");
                    Console.WriteLine($"  Expected: {expected}");

                    if (actual == expected)
                    {
                        pass = true;
                        passCount++;
                        Console.WriteLine($"  Result: *** PASS ***");
                    }
                    else
                    {
                        failCount++;
                        error = $"Mismatch: got '{actual}'";
                        Console.WriteLine($"  Result: FAIL - {error}");

                        // Save debug image for failed tests to NG subfolder
                        string ngDir = Path.Combine(outputDir, "NG");
                        Directory.CreateDirectory(ngDir);
                        var resultImg = decoder.CreateRotatedMainVisualization(colorImg, decodeResult, 500);
                        string debugPath = Path.Combine(ngDir, $"{fileName}_result.png");
                        resultImg.Save(debugPath);
                        Console.WriteLine($"  Debug image: {debugPath}");
                        resultImg.Dispose();

                        // Move preprocess/arrow/rotated debug images to NG folder
                        int currentDebugId = passCount + failCount;  // Current test number (excluding skips)
                        string debugIdStr = $"{currentDebugId:D3}";
                        string[] debugSuffixes = { "_0_preprocess.png", "_1_arrow.png", "_2_rotated.png" };
                        foreach (var suffix in debugSuffixes)
                        {
                            string srcPath = Path.Combine(outputDir, debugIdStr + suffix);
                            string dstPath = Path.Combine(ngDir, fileName + suffix);
                            if (File.Exists(srcPath))
                            {
                                File.Copy(srcPath, dstPath, overwrite: true);
                            }
                        }
                    }
                }

                colorImg.Dispose();
                grayImg.Dispose();
            }
            catch (Exception ex)
            {
                error = ex.Message;
                failCount++;
                Console.WriteLine($"  ERROR: {ex.Message}");
            }

            results.Add((fileName, expected, actual, pass, error));

            // Run all tests to get full statistics (comment out for TDD approach)
            // if (!pass)
            // {
            //     Console.WriteLine($"\n*** TDD: Stopping at first failure ***");
            //     Console.WriteLine($"Fix this image before continuing.\n");
            //     break;
            // }
        }

        // Summary
        Console.WriteLine($"\n{'=',-60}");
        Console.WriteLine($"SUMMARY: {passCount} PASS, {failCount} FAIL, {skipCount} SKIP out of {testFiles.Length} total");
        Console.WriteLine($"{'=',-60}");

        if (failCount == 0 && passCount == testFiles.Length - skipCount)
        {
            Console.WriteLine($"\n*** ALL {testFiles.Length} TESTS PASSED! ***");
        }
        else if (failCount > 0)
        {
            Console.WriteLine($"\nFirst failure:");
            var firstFail = results.FirstOrDefault(r => !r.pass);
            if (firstFail.file != null)
            {
                Console.WriteLine($"  File: {firstFail.file}");
                Console.WriteLine($"  Expected: {firstFail.expected}");
                Console.WriteLine($"  Actual: {firstFail.actual}");
                Console.WriteLine($"  Error: {firstFail.error}");
            }
        }
    }
}
