# Exercise copied README control flow. Python and Git calls below are mocks.
param([string]$Readme = (Join-Path (Split-Path $PSScriptRoot -Parent) 'README.md'))
$ErrorActionPreference = 'Stop'
$body = [System.IO.File]::ReadAllText($Readme)
$blocks = @([regex]::Matches($body, '(?ms)^```powershell\r?\n(.*?)^```') | ForEach-Object { $_.Groups[1].Value })
$passed = [System.Collections.Generic.List[string]]::new()
foreach ($block in $blocks) {
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseInput($block, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw ('README PowerShell parse failed: ' + ($parseErrors | Out-String)) }
}
$passed.Add("All $($blocks.Count) PowerShell blocks parsed")

$fresh = @($blocks | Where-Object { $_.Contains('git clone ') })
if ($fresh.Count -ne 1) { throw 'Expected exactly one clone block.' }
$existing = @($blocks | Where-Object { $_.Contains("Read-Host 'Full path to the EXISTING public checkout'") })
if ($existing.Count -ne 1) { throw 'Expected exactly one existing-checkout block.' }
$setup = @($blocks | Where-Object { $_.Contains('Package installation failed. Do not continue to the notebook.') })
$gpu = @($blocks | Where-Object { $_.Contains('$pytorchCudaIndex = Read-Host') })
$training = @($blocks | Where-Object { $_.Contains('$reviewedTrain = Read-Host') })
if ($setup.Count -ne 1 -or $gpu.Count -ne 1 -or $training.Count -ne 1) { throw 'Expected one block per setup stage.' }
function ConvertTo-MockedPythonBlock {
    param([string]$Block)
    return $Block.Replace('.\.venv-train\Scripts\python.exe', 'Mock-Python').Replace('.\.venv\Scripts\python.exe', 'Mock-Python')
}
$originalRoot = Get-Variable SafetyProjectRoot -Scope Global -ErrorAction SilentlyContinue
$hadOriginalRoot = $null -ne $originalRoot
$originalRootValue = if ($hadOriginalRoot) { $originalRoot.Value } else { $null }
$originalLocation = (Get-Location).Path
try {
    & {
        $script:changedLocation = $false
        function Read-Host { return (Join-Path $PSScriptRoot 'never_created_mock_clone') }
        function Test-Path { return $false }
        function git { $global:LASTEXITCODE = 128 }
        function Set-Location { $script:changedLocation = $true; throw 'Unexpected location change.' }
        $global:SafetyProjectRoot = 'older-checkout'
        $caught = $false
        try { & ([scriptblock]::Create($fresh[0])) } catch { $caught = $_.Exception.Message -like 'Clone failed*' }
        if (-not $caught -or $global:SafetyProjectRoot -or $script:changedLocation) { throw 'Failed clone did not prevent old-checkout reuse.' }
    }
    $passed.Add('Failed clone resets checkout selection and stops before changing directory')
    & {
        function Read-Host { throw 'The recommended clone should choose its default without a prompt.' }
        function Test-Path { return $true }
        function git { throw 'An existing destination must not reach Git.' }
        function python { throw 'An existing destination must not reach Python.' }
        $global:SafetyProjectRoot = 'older-checkout'
        $caught = $false
        try { & ([scriptblock]::Create($fresh[0])) } catch { $caught = $_.Exception.Message -like 'That folder already exists*' }
        if (-not $caught -or $global:SafetyProjectRoot) { throw 'Existing destination did not stop and clear the selection.' }
    }
    $passed.Add('Default clone destination needs no typed path and never replaces an existing folder')
    foreach ($scenario in @('old_checkout', 'untracked_checker', 'mixed_checkout')) {
        & {
            $script:pythonCalls = 0
            function Read-Host { return (Get-Location).Path }
            function Set-Location { }
            function Test-Path {
                param([string]$LiteralPath)
                if ($scenario -eq 'old_checkout' -and $LiteralPath -like '*check_public_checkout.py') { return $false }
                return $true
            }
            function git { $global:LASTEXITCODE = if ($scenario -eq 'untracked_checker') { 1 } else { 0 } }
            function python { $script:pythonCalls++; $global:LASTEXITCODE = 2 }
            $global:SafetyProjectRoot = 'older-checkout'
            $message = ''
            try { & ([scriptblock]::Create($existing[0])) } catch { $message = $_.Exception.Message }
            $expectedCalls = if ($scenario -eq 'mixed_checkout') { 1 } else { 0 }
            if (-not $message -or $global:SafetyProjectRoot -or $script:pythonCalls -ne $expectedCalls) {
                throw "Existing-checkout rejection failed for $scenario."
            }
        }
        $passed.Add("Existing $scenario stops before selecting a public checkout")
    }
    & {
        $script:pythonCalls = 0
        function Read-Host { return (Get-Location).Path }
        function Set-Location { }
        function Test-Path { return $true }
        function git { $global:LASTEXITCODE = 0 }
        function python { $script:pythonCalls++; $global:LASTEXITCODE = 0 }
        & ([scriptblock]::Create($existing[0]))
        if ($global:SafetyProjectRoot -ne (Get-Location).Path -or $script:pythonCalls -ne 1) { throw 'Clean public checkout was not selected after its check.' }
    }
    $passed.Add('A checked existing public checkout is selected without cloning or installing')
    & {
        $global:SafetyProjectRoot = $null
        function python { throw 'Unexpected environment creation; setup should have stopped.' }
        function Mock-Python { throw 'Unexpected Python call; setup should have stopped.' }
        $caught = $false
        try { & ([scriptblock]::Create((ConvertTo-MockedPythonBlock $setup[0]))) } catch { $caught = $_.Exception.Message -like 'Select a checkout*' }
        if (-not $caught) { throw 'Notebook setup accepted missing checkout selection.' }
    }
    $passed.Add('Notebook setup refuses an unselected checkout before installation')
    & {
        $global:SafetyProjectRoot = (Get-Location).Path
        $script:guardCalls = 0
        function python {
            if (($args -join ' ') -ne '-B code/check_public_checkout.py') { throw 'Unexpected environment creation after checkout rejection.' }
            $script:guardCalls++
            $global:LASTEXITCODE = 2
        }
        function Mock-Python { throw 'Unexpected install or launch after checkout rejection.' }
        $caught = $false
        try { & ([scriptblock]::Create((ConvertTo-MockedPythonBlock $setup[0]))) } catch { $caught = $_.Exception.Message -like 'Public checkout validation failed*' }
        if (-not $caught -or $script:guardCalls -ne 1) { throw 'Notebook setup continued after a rejected checkout.' }
    }
    $passed.Add('Mixed checkout stops before environment creation, installation or notebook launch')
    foreach ($scenario in @(
        @{ Name = 'preflight'; FailAt = 1; ExpectedCalls = 1; ExpectedError = 'Preflight failed*' },
        @{ Name = 'training'; FailAt = 2; ExpectedCalls = 2; ExpectedError = 'Training failed*' },
        @{ Name = 'evaluation'; FailAt = 3; ExpectedCalls = 3; ExpectedError = 'Evaluation failed*' },
        @{ Name = 'report validation'; FailAt = 4; ExpectedCalls = 4; ExpectedError = 'Report validation failed*' },
        @{ Name = 'success'; FailAt = 0; ExpectedCalls = 4; ExpectedError = '' }
    )) {
        & {
            $global:SafetyProjectRoot = (Get-Location).Path
            $script:mockCalls = [System.Collections.Generic.List[string]]::new()
            $script:promptCount = 0
            function Read-Host {
                $script:promptCount++
                if ($script:promptCount -eq 1) { return 'reviewed_development_train.csv' }
                return 'reviewed_development_holdout.csv'
            }
            function Test-Path {
                param([string]$LiteralPath, [string]$PathType)
                return $LiteralPath.StartsWith('reviewed_development_')
            }
            function Resolve-Path { param([string]$LiteralPath); return [pscustomobject]@{ Path = $LiteralPath } }
            function Mock-Python {
                $script:mockCalls.Add(($args -join ' '))
                $global:LASTEXITCODE = 0
                if ($script:mockCalls.Count -eq $scenario.FailAt) { $global:LASTEXITCODE = 2 }
            }
            $copied = ConvertTo-MockedPythonBlock $training[0]
            $message = ''
            try { & ([scriptblock]::Create($copied)) } catch { $message = $_.Exception.Message }
            if ($script:mockCalls.Count -ne $scenario.ExpectedCalls) { throw "Wrong dependent command count for $($scenario.Name)." }
            if ($scenario.ExpectedError -and $message -notlike $scenario.ExpectedError) { throw "Missing fail-stop for $($scenario.Name): $message" }
            if (-not $scenario.ExpectedError -and $message) { throw "Unexpected success-path error: $message" }
            if ($script:mockCalls[0] -notlike '*--train-file reviewed_development_train.csv --preflight-only') { throw 'The explicit reviewed training file was not used by preflight.' }
            if ($script:mockCalls.Count -ge 3 -and $script:mockCalls[2] -notlike '*--holdout reviewed_development_holdout.csv*') { throw 'Evaluation did not use the separate reviewed holdout.' }
        }
        $passed.Add("Mock $($scenario.Name) path calls only permitted dependent commands")
    }
    foreach ($scenario in @(
        @{ Name = 'wheel installation'; FailAt = 1; ExpectedCalls = 1; ExpectedError = 'CUDA PyTorch installation failed*' },
        @{ Name = 'first CUDA diagnostic'; FailAt = 2; ExpectedCalls = 2; ExpectedError = 'PyTorch cannot use this GPU*' },
        @{ Name = 'training dependencies'; FailAt = 3; ExpectedCalls = 3; ExpectedError = 'Training dependency installation failed*' },
        @{ Name = 'dependency consistency'; FailAt = 4; ExpectedCalls = 4; ExpectedError = 'Resolve training dependency conflicts*' },
        @{ Name = 'second CUDA diagnostic'; FailAt = 5; ExpectedCalls = 5; ExpectedError = 'The GPU check failed after dependency installation*' },
        @{ Name = 'successful GPU checks'; FailAt = 0; ExpectedCalls = 5; ExpectedError = '' }
    )) {
        & {
            $global:SafetyProjectRoot = (Get-Location).Path
            $script:mockCalls = [System.Collections.Generic.List[string]]::new()
            function Read-Host { return 'https://download.pytorch.org/whl/cu999' }
            function Mock-Python {
                $script:mockCalls.Add(($args -join ' '))
                $global:LASTEXITCODE = 0
                if ($script:mockCalls.Count -eq $scenario.FailAt) { $global:LASTEXITCODE = 2 }
            }
            $copied = ConvertTo-MockedPythonBlock $gpu[0]
            $message = ''
            try { & ([scriptblock]::Create($copied)) } catch { $message = $_.Exception.Message }
            if ($script:mockCalls.Count -ne $scenario.ExpectedCalls) { throw "Wrong command count for $($scenario.Name)." }
            if ($scenario.ExpectedError -and $message -notlike $scenario.ExpectedError) { throw "Missing fail-stop for $($scenario.Name): $message" }
            if (-not $scenario.ExpectedError -and $message) { throw "Unexpected GPU success-path error: $message" }
        }
        $passed.Add("Mock $($scenario.Name) path stops or completes as intended")
    }
    $installArgs = $script:mockCalls[0] -split '\s+'
    if (-not ($installArgs -contains '--upgrade') -or -not ($installArgs -contains '--force-reinstall')) {
        throw 'The CUDA installation command must upgrade and force-reinstall an existing CPU build.'
    }
    $passed.Add('Executed CUDA installation arguments include upgrade and force-reinstall')
    & {
        $global:SafetyProjectRoot = (Get-Location).Path
        function Read-Host { return 'https://download.pytorch.org/whl/cpu' }
        function Mock-Python { throw 'Unexpected Python call; the CPU index should have been rejected.' }
        $caught = $false
        try { & ([scriptblock]::Create((ConvertTo-MockedPythonBlock $gpu[0]))) } catch { $caught = $_.Exception.Message -like 'Use the official CUDA wheel index*' }
        if (-not $caught) { throw 'CPU-only wheel index accepted.' }
    }
    $passed.Add('CPU-only wheel index rejected before package installation')
    if ((Get-Location).Path -ne $originalLocation) { throw 'Verification changed its working directory.' }
    [ordered]@{
        schema = 'readme_powershell_mock_checks_v1'
        status = 'passed'
        powershell_version = $PSVersionTable.PSVersion.ToString()
        readme_sha256 = (Get-FileHash -LiteralPath $Readme -Algorithm SHA256).Hash.ToLowerInvariant()
        checks_passed = $passed.Count
        checks = @($passed)
        real_clone_or_install_performed = $false
        model_or_training_performed = $false
        evaluation_data_used = $false
    } | ConvertTo-Json -Depth 4 -Compress
} finally {
    if ($hadOriginalRoot) { $global:SafetyProjectRoot = $originalRootValue }
    else { Remove-Variable SafetyProjectRoot -Scope Global -ErrorAction SilentlyContinue }
}
