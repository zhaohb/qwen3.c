@echo off
REM Build qwen_vk.exe for Windows + MinGW + Intel Arc Vulkan
REM Usage:
REM   scripts\build_win.bat
REM   set VKINC=... & set VKLIB=... & set GV=... & scripts\build_win.bat

setlocal EnableExtensions
cd /d "%~dp0..\c"

if "%VKINC%"=="" set "VKINC=%USERPROFILE%\.vkbuild\Vulkan-Headers-main\include"
if "%VKLIB%"=="" set "VKLIB=%USERPROFILE%\.vkbuild"
if "%GV%"=="" set "GV=%USERPROFILE%\.vkbuild\glslang\bin\glslangValidator.exe"

echo [1/3] Compile shaders -^> *.spv
if not exist "%GV%" (
  echo ERROR: glslangValidator not found: %GV%
  echo Set GV= to your glslangValidator.exe
  exit /b 1
)
pushd shaders
for %%f in (*.comp) do (
  "%GV%" --target-env vulkan1.2 -V "%%f" -o "%%~nf.spv"
  if errorlevel 1 exit /b 1
)
popd

echo [2/3] Compile qwen_opts.c + backend_vulkan.c
gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN ^
  -I"%VKINC%" -I. -c qwen_opts.c -o qwen_opts.o
if errorlevel 1 exit /b 1
gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN ^
  -I"%VKINC%" -I. -c backend_vulkan.c -o backend_vulkan.o
if errorlevel 1 exit /b 1

echo [3/3] Link qwen_vk.exe
gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN ^
  -I"%VKINC%" -I. qwen.c qwen_opts.o backend_vulkan.o ^
  -o qwen_vk.exe -L"%VKLIB%" -lvulkan-1 -lm -lpsapi
if errorlevel 1 (
  echo retry with omp_stub if present...
  if exist "%VKLIB%\omp_stub.c" (
    gcc -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=native -DCOLI_VULKAN ^
      -I"%VKINC%" -I. qwen.c qwen_opts.o backend_vulkan.o "%VKLIB%\omp_stub.c" ^
      -o qwen_vk.exe -L"%VKLIB%" -lvulkan-1 -lm -lpsapi
  ) else (
    exit /b 1
  )
)

echo OK: %CD%\qwen_vk.exe
endlocal
