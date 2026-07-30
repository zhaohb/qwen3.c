# Build qwen_vk.exe (Windows + MinGW + Vulkan)
#   .\scripts\build_win.ps1
#   $env:VKINC=...; $env:VKLIB=...; $env:GV=...; .\scripts\build_win.ps1

$ErrorActionPreference = "Stop"
$cdir = Join-Path $PSScriptRoot "..\c" | Resolve-Path
Set-Location $cdir

if (-not $env:VKINC) { $env:VKINC = "$env:USERPROFILE\.vkbuild\Vulkan-Headers-main\include" }
if (-not $env:VKLIB) { $env:VKLIB = "$env:USERPROFILE\.vkbuild" }
if (-not $env:GV)    { $env:GV    = "$env:USERPROFILE\.vkbuild\glslang\bin\glslangValidator.exe" }

if (-not (Test-Path $env:GV)) { throw "glslangValidator not found: $($env:GV)" }

Write-Host "[1/3] Compile shaders"
Push-Location shaders
Get-ChildItem *.comp | ForEach-Object {
  & $env:GV --target-env vulkan1.2 -V $_.FullName -o ($_.BaseName + ".spv")
  if ($LASTEXITCODE -ne 0) { throw "shader compile failed: $($_.Name)" }
}
Pop-Location

Write-Host "[2/3] Compile qwen_opts.c + backend_vulkan.c"
& gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN `
  "-I$($env:VKINC)" -I. -c qwen_opts.c -o qwen_opts.o
if ($LASTEXITCODE -ne 0) { throw "qwen_opts compile failed" }
& gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN `
  "-I$($env:VKINC)" -I. -c backend_vulkan.c -o backend_vulkan.o
if ($LASTEXITCODE -ne 0) { throw "backend_vulkan compile failed" }

Write-Host "[3/3] Link qwen_vk.exe"
& gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN `
  "-I$($env:VKINC)" -I. qwen.c qwen_opts.o backend_vulkan.o `
  -o qwen_vk.exe "-L$($env:VKLIB)" -lvulkan-1 -lm -lpsapi
if ($LASTEXITCODE -ne 0) { throw "link failed" }

Write-Host "OK: $(Join-Path $cdir 'qwen_vk.exe')"
