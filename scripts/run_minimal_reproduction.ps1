param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [Parameter(Mandatory=$true)][string]$InternalSubject,
    [Parameter(Mandatory=$true)][string]$ExternalSubject,
    [string]$ImagingPython = "",
    [string]$ToolPython = "",
    [string]$RadiomicsPython = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ImagingPython)) {
    $ImagingPython = Join-Path $ProjectRoot ".venv-imaging\Scripts\python.exe"
}
if ([string]::IsNullOrWhiteSpace($ToolPython)) {
    $ToolPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if ([string]::IsNullOrWhiteSpace($RadiomicsPython)) {
    $RadiomicsPython = Join-Path $ProjectRoot ".venv-radiomics\Scripts\python.exe"
}
$SrcRoot = Join-Path $ProjectRoot "src"

if (!(Test-Path -LiteralPath $ImagingPython)) { throw "Missing imaging Python environment: $ImagingPython" }
if (!(Test-Path -LiteralPath $ToolPython)) { throw "Missing tools Python environment: $ToolPython" }
if (!(Test-Path -LiteralPath $RadiomicsPython)) { throw "Missing PyRadiomics Python environment: $RadiomicsPython" }

$env:PYTHONPATH = $SrcRoot

function Invoke-CheckedPython {
    param(
        [string]$Python,
        [string]$ScriptPath,
        [object[]]$Arguments = @()
    )
    & $Python $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: $ScriptPath"
    }
}

Write-Host "== CAP/COP minimal reproduction =="
Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "ImagingPython: $ImagingPython"
Write-Host "ToolPython: $ToolPython"
Write-Host "RadiomicsPython: $RadiomicsPython"

Write-Host "`n[1/7] DICOM -> U-Net predicted masks, single internal/external smoke cases"
Invoke-CheckedPython -Python $ImagingPython -ScriptPath (Join-Path $ProjectRoot "scripts\run_segmentation_smoke.py") -Arguments @(
    "--internal-subject-id", $InternalSubject, "--external-subject-id", $ExternalSubject
)

Write-Host "`n[2/7] Internal single-case DICOM + predicted mask -> PyRadiomics 13 features -> RF probability"
Invoke-CheckedPython -Python $RadiomicsPython -ScriptPath (Join-Path $ProjectRoot "scripts\run_single_case_lesion_radiomics_chain.py") -Arguments @(
    "--subject-id", $InternalSubject,
    "--mask-path", (Join-Path $ProjectRoot "outputs\segmentation_smoke\internal\${InternalSubject}_strict_oof_predicted_mask.npy")
)

Write-Host "`n[3/7] External single-case DICOM -> locked P95 96-cubed tensor"
Invoke-CheckedPython -Python $ImagingPython -ScriptPath (Join-Path $ProjectRoot "scripts\run_p95_tensor_single_case.py") -Arguments @(
    "--project-root", $ProjectRoot,
    "--subject-id", $ExternalSubject,
    "--output-dir", (Join-Path $ProjectRoot "outputs\minimal_reproduction_p95")
)

Write-Host "`n[4/7] Model A locked external PyRadiomics-feature RF replay"
Invoke-CheckedPython -Python $RadiomicsPython -ScriptPath (Join-Path $ProjectRoot "scripts\verify_lesion_radiomics_rf_locked.py")

Write-Host "`n[5/7] Model B locked embedding -> linear probe replay"
Invoke-CheckedPython -Python $ToolPython -ScriptPath (Join-Path $ProjectRoot "scripts\verify_whole_chest_ct_locked_linear_probe.py")

Write-Host "`n[6/7] Model C locked SP6/SP7 table replay and Model D artifact check"
Invoke-CheckedPython -Python $ToolPython -ScriptPath (Join-Path $ProjectRoot "scripts\verify_stable_multirepresentation_locked_tables.py")
Invoke-CheckedPython -Python $ToolPython -ScriptPath (Join-Path $ProjectRoot "scripts\verify_supplementary_exploratory_artifacts.py")

Write-Host "`n[7/7] Model D supplementary exploratory single-case clinical + full-chest DenseNet + fusion replay"
Invoke-CheckedPython -Python $ImagingPython -ScriptPath (Join-Path $ProjectRoot "scripts\run_supplementary_exploratory_subset_replay.py") -Arguments @(
    "--public-subject-id", $ExternalSubject,
    "--output-dir", (Join-Path $ProjectRoot "outputs\supplementary_exploratory_subset_replay\$ExternalSubject")
)

Write-Host "`nMinimal reproduction completed. This is not a full all-case raw-DICOM replay."
