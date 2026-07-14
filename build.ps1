# Build helper for yolov5_cpp (MSVC + LibTorch, Release).
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$build = Join-Path $root "build"

if (-not (Test-Path $build)) { New-Item -ItemType Directory -Path $build | Out-Null }

# LibTorch prebuilt binaries are Release/MD, so we must build Release.
# third_party lives one level up (C:\prog\vc\third_party), outside the project.
cmake -S $root -B $build -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_PREFIX_PATH="$root/../third_party/libtorch"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build $build --config Release -- /m
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "`nBuilt: $build\Release\yolov5.exe"
