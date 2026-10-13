"""Independent Windows watchdog for one explicitly named WSL benchmark run.

Stops only matching benchmark parent/workers, never WSL or other GPU applications.
Run with a normal Windows Python process, separate from the GPU driver process.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess
import time


def matches_run(argv, cwd, script, output):
    expected = PurePosixPath(script)
    script_matches = (str(expected) in argv or
        (f"scripts/{expected.name}" in argv and cwd == str(expected.parent.parent)))
    try:
        return script_matches and argv[argv.index("--output") + 1] == output
    except (ValueError, IndexError):
        return False


WSL_SCAN = r'''
import json, os, signal, sys
from pathlib import Path
script, output, action = sys.argv[1:]
matches = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = (entry / 'cmdline').read_bytes().decode().split('\0')
        cwd = str((entry / 'cwd').resolve())
        is_script = script in argv or ('scripts/' + Path(script).name in argv and cwd == str(Path(script).parent.parent))
        if is_script and '--output' in argv and argv[argv.index('--output') + 1] == output:
            matches.append((int(entry.name), '--worker' in argv))
    except (OSError, ValueError, IndexError):
        continue
for pid, worker in sorted(matches, key=lambda item: not item[1]):
    if action != 'scan':
        try:
            os.kill(pid, signal.SIGTERM if action == 'term' else signal.SIGKILL)
        except ProcessLookupError:
            pass
print(json.dumps(matches))
'''


def scan(args, action="scan"):
    result = subprocess.run(["wsl.exe", "-d", args.distribution, "--exec", "python3",
        "-c", WSL_SCAN, args.script, args.output, action], capture_output=True,
        text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(args):
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.utcoffset() is None:
        raise ValueError("deadline must include a timezone")
    started = time.monotonic()
    seen_run = False
    state = dict(deadline_utc=deadline.isoformat(), script=args.script, output=args.output)
    while True:
        try:
            matches = scan(args)
            now = datetime.now(timezone.utc)
            state.update(last_checked_utc=now.isoformat(), matching_pids=matches, status="watching")
            seen_run = seen_run or bool(matches)
            if now >= deadline:
                state["terminated_pids"] = matches
                if matches:
                    scan(args, "term")
                    time.sleep(3)
                    if scan(args):
                        scan(args, "kill")
                        time.sleep(2)
                state.update(status="deadline_enforced", matching_pids=scan(args))
                save(args.state_file, state)
                if state["matching_pids"]:
                    raise RuntimeError("matching GPU benchmark processes remain")
                return
            if not matches and (seen_run or time.monotonic() - started > args.start_grace):
                state["status"] = "benchmark_processes_exited" if seen_run else "benchmark_never_started"
                save(args.state_file, state)
                return
            save(args.state_file, state)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            state.update(status="scan_error_retrying", error=str(error))
            save(args.state_file, state)
        time.sleep(min(args.poll_seconds, max(1., (deadline - datetime.now(timezone.utc)).total_seconds())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", default="Ubuntu")
    parser.add_argument("--script", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--start-grace", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=15)
    main(parser.parse_args())
