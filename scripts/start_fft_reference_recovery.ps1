$ErrorActionPreference = "Stop"
$studyRoot = Join-Path (Split-Path $PSScriptRoot -Parent) ".benchmark-results/2026-10-27-fft-reference-recovery"
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class ReferenceRecoveryPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if ([ReferenceRecoveryPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
    throw "Could not temporarily inhibit automatic sleep"
}
try {
    # Windows PowerShell otherwise turns the first native stderr line into a
    # terminating error and hides the rest of a Python traceback.
    $ErrorActionPreference = "Continue"
    & wsl.exe -e /home/yash/ratcliff-rtx4060-baseline/.venv/bin/python /mnt/c/Users/Yash/repos/ratcliff/scripts/run_fft_reference_recovery.py --source /mnt/c/Users/Yash/repos/ratcliff/.benchmark-results/2026-10-27-fft-reference-recovery/source-v2 2>&1 |
        Tee-Object -FilePath (Join-Path $studyRoot "run.log") -Append
    $studyExitCode = $LASTEXITCODE
} finally {
    [void][ReferenceRecoveryPower]::SetThreadExecutionState([uint32]2147483648)
}
exit $studyExitCode
