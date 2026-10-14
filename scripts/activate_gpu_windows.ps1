# Ejecutar después de 'conda activate subdub-titler'.
# PowerShell; las rutas deben existir antes de importar ctranslate2.
$gpuPaths = python -c "import importlib.util,pathlib; mods=['nvidia.cublas','nvidia.cudnn']; paths=[pathlib.Path(next(iter(s.submodule_search_locations)))/'bin' for m in mods if (s:=importlib.util.find_spec(m)) and s.submodule_search_locations]; print(';'.join(map(str,filter(lambda p:p.is_dir(),paths))))"
if (-not $gpuPaths) { throw 'No se encontró runtime CUDA instalado por pip.' }
$env:PATH = "$gpuPaths;$env:PATH"
Write-Host "Runtime CUDA en PATH: $gpuPaths"
