# One-time migration for an existing Windows checkout. Close Codex and terminals first.
$ErrorActionPreference = 'Stop'
$workspaceSource = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$workspaceParent = Split-Path -Parent $workspaceSource
$workspaceTarget = [System.IO.Path]::GetFullPath((Join-Path $workspaceParent 'Reinforcement Learning-Based API Rate Control'))
if ((Split-Path -Leaf $workspaceSource) -ne 'Intelligent API Rate Limiting') {
    throw 'This migration only moves the explicitly named original workspace folder.'
}
if (-not $workspaceTarget.StartsWith($workspaceParent + [System.IO.Path]::DirectorySeparatorChar)) {
    throw 'Target must remain directly inside the existing project parent.'
}
if (Test-Path -LiteralPath $workspaceTarget) { throw "Target already exists: $workspaceTarget" }
Set-Location -LiteralPath $workspaceParent
Move-Item -LiteralPath $workspaceSource -Destination $workspaceTarget
Set-Location -LiteralPath $workspaceTarget
# Console entry points contain absolute interpreter paths and must be recreated after moving.
uv sync --frozen --extra dev --extra ml --reinstall
if ($LASTEXITCODE -ne 0) { throw 'Folder moved; dependency reinstall failed. Run uv sync --reinstall again.' }
Write-Output "Workspace moved to $workspaceTarget. Reopen this folder in Codex."
