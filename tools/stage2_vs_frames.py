"""
Stage-2 rms as a function of how many frames were stacked.

Runs the real pipeline (stack -> platesolve -> distortion fit) on the first N frames
for a ladder of N, then fits rms(N)^2 = static^2 + random^2/N. The asymptote is the
error stacking can never remove (optics + model + anything static); the 1/N part is
what more frames still buy. Together with the track-level scaling in error_budget.py
this closes the loop: the track analysis predicts the random part, and the pipeline
measures where the curve actually flattens.

    python tools/stage2_vs_frames.py "I:/MEE test frames/fits/zwo3/field/*.fit" \
        --darks "I:/MEE test frames/fits/zwo3/darks/*.fit" --counts 2 3 5 9 \
        --out docs/bench/psf/stage2_vs_frames_zwo3

Those two folders are a curated selection, not whole captures: 9 of the 60 frames in the
source `Zenith-01-0.2s` and 38 of its 64 darks. Aim this at the raw capture instead and it
stacks five wrong pointings against the wrong darks, and the ladder will not match. Every
frame is listed with its hash in `docs/bench/TEST_FRAMES.md`.
"""

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def run_pipeline(lights, darks, workdir, events_path):
    command = [sys.executable, '-m', 'mee2024.main', 'run', *lights,
               '--no-display', '--quiet', '-o', str(workdir),
               '--events-jsonl', str(events_path)]
    if darks:
        command += ['--dark', *darks]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-12:]
        raise RuntimeError('pipeline failed:\n' + '\n'.join(tail))
    metrics = None
    with open(events_path, encoding='utf-8') as fp:
        for line in fp:
            event = json.loads(line)
            if event.get('type') == 'metrics' and event.get('stage') == 'distortion':
                metrics = event
    if metrics is None:
        raise RuntimeError('no stage-2 metrics event found')
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pattern')
    ap.add_argument('--darks', default=None)
    ap.add_argument('--counts', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.pattern))
    darks = sorted(glob.glob(args.darks)) if args.darks else []
    print(f'{len(files)} frames available; ladder {args.counts}', flush=True)

    results = []
    for n in args.counts:
        if n > len(files):
            print(f'N={n}: skipped, only {len(files)} frames')
            continue
        t0 = time.time()
        workdir = args.out / f'n{n}'
        workdir.mkdir(exist_ok=True)
        try:
            metrics = run_pipeline(files[:n], darks, workdir, workdir / 'events.jsonl')
        except RuntimeError as error:
            print(f'N={n}: {error}')
            continue
        results.append({'n': n, 'rms_mas': metrics['rms_mas'],
                        'n_stars': metrics['n_stars'],
                        'platescale': metrics['platescale']})
        print(f'N={n}: rms {metrics["rms_mas"]:.1f} mas, {metrics["n_stars"]} stars, '
              f'platescale {metrics["platescale"]:.3f} "/px  ({time.time()-t0:.0f}s)',
              flush=True)

    if len(results) >= 3:
        ns = np.array([r['n'] for r in results], dtype=float)
        rms = np.array([r['rms_mas'] for r in results])
        # linear in 1/N: rms^2 = a2 + b2 / N
        coeffs = np.polyfit(1.0 / ns, rms ** 2, 1)
        b2, a2 = coeffs[0], coeffs[1]
        static = float(np.sqrt(max(a2, 0.0)))
        random1 = float(np.sqrt(max(b2, 0.0)))
        summary = {'results': results, 'static_mas': static,
                   'random_single_frame_mas': random1}
        print(f'\nfit rms(N)^2 = static^2 + random^2/N:')
        print(f'  static (never stacks away) : {static:.1f} mas')
        print(f'  random at N=1              : {random1:.1f} mas')
        print(f'  predicted rms at N=inf     : {static:.1f} mas; '
              f'at N=100: {np.sqrt(a2 + b2/100):.1f} mas')
    else:
        summary = {'results': results}

    with open(args.out / 'summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2)
    print(f'\nresults in {args.out}')


if __name__ == '__main__':
    main()
