@echo off
REM Build script for ShapeBasedMatcher on Windows
REM Requires: Visual Studio 2022, CMake 3.16+, OpenCV 4.x

setlocal

REM Set OpenCV path (adjust as needed)
set OPENCV_DIR=C:\opencv\build

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DOpenCV_DIR=%OPENCV_DIR% ^
    -DCMAKE_BUILD_TYPE=Release

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

REM Build
echo Building...
cmake --build . --config Release

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo Build successful!
echo Output: build\Release\ShapeBasedMatcher.dll

endlocal
