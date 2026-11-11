# [SETUP-WIN] One command on a Windows exam machine (PowerShell):   .\setup.ps1
# Override models:  $env:CODER="qwen2.5-coder:1.5b"; .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$CODER = if ($env:CODER) { $env:CODER } else { "qwen2.5-coder:3b" }
$EMBED = if ($env:EMBED) { $env:EMBED } else { "nomic-embed-text" }
$VLM   = if ($env:VLM)   { $env:VLM }   else { "qwen2.5vl:3b" }
$PY    = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "== 1/6 Python packages"
& $PY -m pip install -r requirements.txt

Write-Host "== 2/6 Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found. Install it from https://ollama.com/download (Windows installer)."
    exit 1
}
try { ollama list | Out-Null } catch { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep 5 }

Write-Host "== 3/6 Models: $CODER $EMBED $VLM"
$have = (ollama list) -join "`n"
foreach ($m in @($CODER, $EMBED, $VLM)) {
    if ($have -notmatch [regex]::Escape($m)) { ollama pull $m }
}

Write-Host "== 4/6 API docs from installed libraries -> docs/api/"
& $PY build_corpus.py docs/api

Write-Host "== 5/6 Environment check"
& $PY check_env.py --coder $CODER --embed $EMBED --vlm $VLM

Write-Host "== 6/6 Build the vector store (chroma_db/)"
& $PY rag_agent.py --docs docs --embed $EMBED --rebuild

Write-Host "`nReady. Try:"
Write-Host "  $PY rag_agent.py --docs docs --task-file tasks/01_edges.txt --model $CODER"
Write-Host "  $PY smoke_test.py --model $CODER"
