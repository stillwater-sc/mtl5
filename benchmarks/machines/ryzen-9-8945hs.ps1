<#
.SYNOPSIS
    Cache-blocking A/B on the Ryzen 9 8945HS (Zen 4 / Hawk Point, Windows/MSVC).

.DESCRIPTION
    This file IS the invocation. Two of its settings are not preferences:

    * -Arch "/arch:AVX512". MTL5_NATIVE_ARCH is a NO-OP under MSVC (it is guarded
      `if(MTL5_NATIVE_ARCH AND NOT MSVC)`) and MSVC x64 defaults to SSE2, so
      without this flag the run silently measures an SSE2 build -- and completes,
      and prints a clean table. That is not hypothetical: the first Zen 4 A/B was
      an AVX2 run when AVX-512 was intended, and nothing in the CSV could say so
      until `build_isa` was added. Check it in the sidecar afterwards.

    * -PCores "0,2,4,...,14". One logical id per physical core; the SMT siblings
      are odd. Eight Zen 4 cores, homogeneous -- unlike the i7 there is no
      P/E split here, which is why the Windows CPUID detection caveat (detection
      may run before the affinity mask takes effect) is harmless on this machine
      and not on a hybrid one.

    All four arms run in ONE session: this machine is the AVX-512 half of the
    kc/mc question (#430). Its L1d is 32 KB, so `kc` does NOT move here -- which
    is exactly what makes it the control for the i7.

.EXAMPLE
    pwsh benchmarks/machines/ryzen-9-8945hs.ps1
    pwsh benchmarks/machines/ryzen-9-8945hs.ps1 -Arms "detected,default" -Rounds 3
#>
[CmdletBinding()]
param(
    [string]$Arms    = "default,detected,kconly,mconly",
    [string]$PCores  = "0,2,4,6,8,10,12,14",
    [string]$Threads = "1,8",
    [int]$Rounds     = 5,
    [int]$Reps       = 5,
    [string]$Arch    = "/arch:AVX512",
    [string]$OutDir  = "benchmarks\data\ryzen-9-8945hs",
    [switch]$Force
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Refuse to write this machine's numbers on a different machine: the CSVs are
# named by arm and the runner clears them before writing, so a profile run on the
# wrong host replaces committed evidence with data from somewhere else.
$expect = "8945HS"
$actual = (Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1).Name
if ($actual -notmatch $expect) {
    Write-Host "This profile is for a Ryzen 9 $expect; this machine reports:"
    Write-Host "  $actual"
    if (-not $Force) {
        Write-Host "Refusing to write $expect data. Pass -Force if this really is one."
        exit 2
    }
}

$env:BENCH_PROFILE = "ryzen-9-8945hs"

Write-Host "profile: $($env:BENCH_PROFILE)"
Write-Host "  cpu:     $actual"
Write-Host "  isa:     $Arch   (MTL5_NATIVE_ARCH is a no-op under MSVC)"
Write-Host "  pinning: $PCores (one logical id per physical core)"
Write-Host "  arms:    $Arms"
Write-Host "  outdir:  $OutDir"
Write-Host ""

& (Join-Path $RepoRoot "benchmarks\run_blocking_ab.ps1") `
    -Arms $Arms -PCores $PCores -Threads $Threads `
    -Rounds $Rounds -Reps $Reps -Arch $Arch -OutDir $OutDir
exit $LASTEXITCODE
