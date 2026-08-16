$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repo
try {
    python -m pip install -e ".[sources]"
    python (Join-Path $PSScriptRoot "build_example.py") @args
}
finally {
    Pop-Location
}
