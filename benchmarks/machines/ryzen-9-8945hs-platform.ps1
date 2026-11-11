<#
.SYNOPSIS
Platform benchmark (#737) on the Ryzen 9 8945HS (Zen 4 / Hawk Point, 8 cores,
SMT, Windows/MSVC): one thread per physical core, under the session power scheme.

.DESCRIPTION
One CPU per core (0,2,...,14): the SMT siblings are excluded, since two threads
on one core share its FMA units. The clock policy, the OpenBLAS lookup and the
restore are in _platform_profile.ps1. Extra arguments go to platform.ps1.

This is the platform card's profile; ryzen-9-8945hs.ps1 next to it is the
cache-blocking A/B (#430), a different measurement.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File benchmarks\machines\ryzen-9-8945hs-platform.ps1
$env:CLASS = 'large'; powershell -ExecutionPolicy Bypass -File benchmarks\machines\ryzen-9-8945hs-platform.ps1
#>
. (Join-Path $PSScriptRoot '_platform_profile.ps1')
Assert-Machine '8945HS'
$cpus = Get-OneCpuPerCore
if (@($cpus -split ',').Count -ne 8) { Write-Host "expected 8 cores, got '$cpus'"; exit 2 }
$rc = Invoke-PlatformProfile 'ryzen-9-8945hs' $cpus $args
exit [int]($rc | Select-Object -Last 1)
