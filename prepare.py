"""Apply the setup fixes to existing PCLA and Bench2Drive folders."""

import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def git(checkout, *arguments):

    return subprocess.run(['git', '-C', str(checkout), *arguments],
                          capture_output=True, text=True)


def apply_patch(checkout, name):

    patch = str(ROOT / 'patches' / name)

    already_applied = git(checkout, 'apply', '--reverse', '--check', patch)
    if already_applied.returncode == 0:
        print(name + ': already applied')
        return

    result = git(checkout, 'apply', patch)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    print(name + ': applied')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcla', type=Path, default=ROOT / 'PCLA')
    parser.add_argument('--b2d', type=Path, default=ROOT / 'Bench2Drive')
    args = parser.parse_args()

    apply_patch(args.pcla.resolve(), 'minddrive-weights.patch')

    apply_patch(args.b2d.resolve(), 'b2d-python310.patch')
