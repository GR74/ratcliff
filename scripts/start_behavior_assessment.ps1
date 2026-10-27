param([switch]$Resume)
$ErrorActionPreference = "Stop"
$studyRoot = Join-Path (Split-Path $PSScriptRoot -Parent) ".benchmark-results/2026-10-26-behavioral-assessment-20-runs"
New-Item -ItemType Directory -Force -Path $studyRoot | Out-Null
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class BehaviorPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if ([BehaviorPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
    throw "Could not temporarily inhibit automatic sleep"
}
try {
    $studyArgs = @('-e', '/home/yash/ratcliff-rtx4060-baseline/.venv/bin/python', '/mnt/c/Users/Yash/repos/ratcliff/scripts/run_behavior_assessment.py')
    if ($Resume) { $studyArgs += '--resume' }
    & wsl.exe @studyArgs 2>&1 | Tee-Object -FilePath (Join-Path $studyRoot "run.log") -Append
    $studyExitCode = $LASTEXITCODE
} finally {
    [void][BehaviorPower]::SetThreadExecutionState([uint32]2147483648)
}
exit $studyExitCode
