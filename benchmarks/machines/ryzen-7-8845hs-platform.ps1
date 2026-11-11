<#
.SYNOPSIS
Platform benchmark (#737) on the Ryzen 7 8845HS (Zen 4 / Hawk Point, 8 cores,
SMT, Windows/MSVC): one thread per physical core, under the session power scheme.

.DESCRIPTION
One CPU per core (0,2,...,14): the SMT siblings are excluded, since two threads
on one core share its FMA units. The clock policy, the OpenBLAS lookup and the
restore are in _platform_profile.ps1. Extra arguments go to platform.ps1.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File benchmarks\machines\ryzen-7-8845hs-platform.ps1
$env:CLASS = 'large'; powershell -ExecutionPolicy Bypass -File benchmarks\machines\ryzen-7-8845hs-platform.ps1
#>
. (Join-Path $PSScriptRoot '_platform_profile.ps1')
Assert-Machine '8845HS'
$cpus = Get-OneCpuPerCore
if (@($cpus -split ',').Count -ne 8) { Write-Host "expected 8 cores, got '$cpus'"; exit 2 }
$rc = Invoke-PlatformProfile 'ryzen-7-8845hs' $cpus $args
exit [int]($rc | Select-Object -Last 1)
