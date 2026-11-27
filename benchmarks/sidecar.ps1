# Dot-sourced by the PowerShell harnesses: the run's half of a sidecar (#930).
# The same as benchmarks/sidecar.sh; see it for the contract.
#
#   . (Join-Path $PSScriptRoot 'sidecar.ps1')
#   $ok = Complete-Sidecar $path @{ benchmark = 'bench_all'; class = 'blas'; bench_cpus = '0' }
#
# Complete-Sidecar adds each given key FILE does not have yet, then machine
# ($env:BENCH_MACHINE, else the cpu_brand the binary recorded), clock_policy
# ($env:CLOCK_POLICY, else "not applied") and bench_cpus ("unpinned"), and
# checks the file against benchmarks/sidecar_keys.txt. It returns $false, and
# names the keys, when one is still missing. A key that doesn't apply still
# gets a value, never an omission. Files are written with LF, as the Linux
# harnesses write them.

$script:SidecarKeysFile = Join-Path $PSScriptRoot 'sidecar_keys.txt'

function Get-SidecarLines([string]$Path) {
    if (Test-Path $Path) { return @(Get-Content $Path) } else { return @() }
}

function Write-SidecarLines([string]$Path, [string[]]$Lines) {
    [System.IO.File]::WriteAllText($Path, (($Lines -join "`n") + "`n"))
}

function Test-SidecarKey([string]$Path, [string]$Key) {
    return [bool](Get-SidecarLines $Path | Where-Object { $_ -like "$Key=*" } | Select-Object -First 1)
}

# Add KEY=VALUE unless the file has KEY already.
function Add-SidecarKey([string]$Path, [string]$Key, [string]$Value) {
    if (-not (Test-SidecarKey $Path $Key)) {
        Write-SidecarLines $Path (@(Get-SidecarLines $Path) + "$Key=$Value")
    }
}

# Set KEY=VALUE, replacing a value already there (one line per key).
function Set-SidecarKey([string]$Path, [string]$Key, [string]$Value) {
    $kept = @(Get-SidecarLines $Path | Where-Object { $_ -notlike "$Key=*" })
    Write-SidecarLines $Path ($kept + "$Key=$Value")
}

# The required keys the file lacks, or has empty (except those marked empty-ok).
function Get-SidecarMissing([string]$Path) {
    $values = @{}
    foreach ($line in (Get-SidecarLines $Path)) {
        $i = $line.IndexOf('=')
        if ($i -gt 0) {
            $k = $line.Substring(0, $i).Trim()
            if (-not $values.ContainsKey($k)) { $values[$k] = $line.Substring($i + 1).Trim() }
        }
    }
    $missing = @()
    foreach ($row in (Get-Content $script:SidecarKeysFile)) {
        $t = $row.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        $parts = $t -split '\s+'
        $key = $parts[0]
        $emptyOk = $parts.Count -gt 1 -and $parts[1] -eq 'empty-ok'
        if (-not $values.ContainsKey($key) -or ($values[$key] -eq '' -and -not $emptyOk)) { $missing += $key }
    }
    return $missing
}

function Complete-Sidecar([string]$Path, [hashtable]$Keys) {
    if (-not (Test-Path $Path)) {
        Write-Host "sidecar: $Path does not exist (did the binary write it?)"
        return $false
    }
    foreach ($k in $Keys.Keys) { Add-SidecarKey $Path $k ([string]$Keys[$k]) }
    $brand = (Get-SidecarLines $Path | Where-Object { $_ -like 'cpu_brand=*' } | Select-Object -First 1)
    $brand = if ($brand) { $brand.Substring(10) } else { 'unknown' }
    $machine = if ($env:BENCH_MACHINE) { $env:BENCH_MACHINE } else { $brand }
    $policy = if ($env:CLOCK_POLICY) { $env:CLOCK_POLICY } else { 'not applied' }
    Add-SidecarKey $Path 'machine' $machine
    Add-SidecarKey $Path 'clock_policy' $policy
    Add-SidecarKey $Path 'bench_cpus' 'unpinned'
    $missing = @(Get-SidecarMissing $Path)
    if ($missing.Count -gt 0) {
        Write-Host "sidecar: $Path lacks required keys: $($missing -join ' ') (benchmarks/sidecar_keys.txt)"
        return $false
    }
    return $true
}
