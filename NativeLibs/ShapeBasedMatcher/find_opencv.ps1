# Find OpenCV in NuGet packages
$nugetPath = "$env:USERPROFILE\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494"
if (Test-Path $nugetPath) {
    Write-Host "Found Emgu.CV runtime at: $nugetPath"
    Get-ChildItem -Path $nugetPath -Recurse -Filter "*.dll" |
        Where-Object { $_.Name -like "opencv*" } |
        Select-Object -First 5 -ExpandProperty FullName
} else {
    Write-Host "Emgu.CV runtime not found at: $nugetPath"
}

# Also check for OpenCV headers
$opencvInclude = Get-ChildItem -Path "$env:USERPROFILE\.nuget\packages" -Directory -Filter "emgu.cv*" -ErrorAction SilentlyContinue |
    ForEach-Object { Get-ChildItem -Path $_.FullName -Recurse -Filter "opencv2" -Directory -ErrorAction SilentlyContinue } |
    Select-Object -First 1 -ExpandProperty FullName

if ($opencvInclude) {
    Write-Host "OpenCV headers found at: $opencvInclude"
} else {
    Write-Host "OpenCV headers not found in NuGet packages"
}

# Check common OpenCV install locations
$commonPaths = @(
    "C:\opencv",
    "C:\opencv\build",
    "D:\opencv",
    "$env:USERPROFILE\opencv"
)

foreach ($path in $commonPaths) {
    if (Test-Path $path) {
        Write-Host "OpenCV found at: $path"
        break
    }
}
