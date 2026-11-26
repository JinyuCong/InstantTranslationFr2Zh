# 一键安装：Python 虚拟环境 + 依赖 + Ollama + 模型，然后自检
# 用法：双击 setup.bat，或在 PowerShell 中运行  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$LLM = "qwen2.5:7b"

Write-Host "[1/5] 创建 Python 虚拟环境 .venv" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败，请先安装 Python 3.10+（https://www.python.org/）" }
}

Write-Host "[2/5] 安装 Python 依赖（含 CUDA 12 / cuDNN 9 运行库，约 1 GB）" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败（国内网络可加 -i https://pypi.tuna.tsinghua.edu.cn/simple 重试）" }

Write-Host "[3/5] 检查 Ollama" -ForegroundColor Cyan
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (-not (Test-Path $candidate)) {
        Write-Host "未找到 Ollama，正在通过 winget 安装…"
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    }
    $ollama = $candidate
}
if (-not (Test-Path $ollama) -and -not (Get-Command $ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama 安装失败，请手动从 https://ollama.com/download 安装"
}

Write-Host "[4/5] 下载翻译模型 $LLM（约 4.7 GB）" -ForegroundColor Cyan
& $ollama pull $LLM
if ($LASTEXITCODE -ne 0) { throw "模型下载失败" }

Write-Host "[5/5] 下载语音识别模型（约 1.6 GB）并自检" -ForegroundColor Cyan
.\.venv\Scripts\python.exe main.py --check
if ($LASTEXITCODE -ne 0) { throw "自检失败" }

Write-Host ""
Write-Host "安装完成！双击「现场翻译.bat」或「线上会议翻译.bat」开始使用。" -ForegroundColor Green
