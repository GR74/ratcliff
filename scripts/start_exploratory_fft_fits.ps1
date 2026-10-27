$ErrorActionPreference = "Stop"
$studyRoot = Join-Path (Split-Path $PSScriptRoot -Parent) ".benchmark-results/2026-10-26-fft200-exploratory-fit-comparison"
New-Item -ItemType Directory -Force -Path $studyRoot | Out-Null
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class ExploratoryFitPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if ([ExploratoryFitPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
    throw "Could not temporarily inhibit automatic sleep"
}
try {
    & wsl.exe -e /home/yash/ratcliff-rtx4060-baseline/.venv/bin/python /mnt/c/Users/Yash/repos/ratcliff/scripts/run_exploratory_fft_fits.py 2>&1 |
        Tee-Object -FilePath (Join-Path $studyRoot "run.log") -Append
    $studyExitCode = $LASTEXITCODE
} finally {
    [void][ExploratoryFitPower]::SetThreadExecutionState([uint32]2147483648)
}
exit $studyExitCode
