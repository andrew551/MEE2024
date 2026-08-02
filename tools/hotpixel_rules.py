"""
Which dark-free rule separates hot pixels from stars best?

Reads the candidate table written by tools/hotpixel_explore.py, so alternative rules cost
nothing to try -- the expensive part (eight passes over 47-megapixel frames) is already done.

    python tools/hotpixel_rules.py docs/bench/hotpix

The first pass scored `detector - sky` persistence and reached 80% recall at 93% precision.
The medians are far cleaner than that suggests (detector: 242 sigma for hot against 0.2 for
stars; sky: -1.5 against 26), which says the *difference* is the wrong combination, not that
the signal is weak: subtracting mixes a faint hot pixel with a bright star. These are the
alternatives, and a look at what the disagreements actually are.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402


def score_rule(score, labels):
    """Best precision at each recall for a ranking, plus the no-false-positive point."""
    order = np.argsort(-score)
    is_hot = labels[order]
    tp = np.cumsum(is_hot)
    fp = np.cumsum(~is_hot)
    recall = tp / max(int(labels.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    out = {}
    for target in (0.8, 0.9, 0.95, 0.99, 1.0):
        reached = np.nonzero(recall >= target - 1e-9)[0]
        out[target] = (precision[reached[0]], int(fp[reached[0]])) if reached.size else (0.0, 0)
    clean = np.nonzero(fp == 0)[0]
    out['clean_recall'] = recall[clean[-1]] if clean.size else 0.0
    # average precision, the summary that does not depend on picking an operating point
    out['ap'] = float(np.sum(np.diff(np.concatenate([[0], recall])) * precision))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory', type=Path, nargs='?', default=Path('docs/bench/hotpix'))
    args = ap.parse_args()

    data = np.load(args.directory / 'candidates.npz', allow_pickle=True)
    det, sky = data['det_persist'], data['sky_persist']
    labels = np.asarray(data['truth'], dtype=bool)
    peak0 = data['peak0']
    print(f'{len(det)} candidates, {int(labels.sum())} dark-confirmed hot\n')

    eps = 1.0
    rules = {
        'detector - sky (first pass)': det - sky,
        'detector alone': det,
        '-sky alone': -sky,
        'log ratio  log(det+e) - log(sky+e)': (np.log(np.maximum(det, 0) + eps)
                                               - np.log(np.maximum(sky, 0) + eps)),
        'detector / (sky + e)': det / (np.maximum(sky, 0) + eps),
        'min(detector, -sky)': np.minimum(det, -sky),
    }

    print(f'{"rule":38s} {"AP":>6}  {"P@80":>6} {"P@90":>6} {"P@95":>6} {"P@99":>6} '
          f'{"recall@0FP":>10}')
    best = None
    for name, score in rules.items():
        r = score_rule(score, labels)
        print(f'{name:38s} {r["ap"]:6.3f}  {r[0.8][0]:6.1%} {r[0.9][0]:6.1%} '
              f'{r[0.95][0]:6.1%} {r[0.99][0]:6.1%} {r["clean_recall"]:10.1%}')
        if best is None or r['ap'] > best[1]['ap']:
            best = (name, r, score)

    print(f'\nbest by average precision: {best[0]}')

    # ---- does interpolating the sky lookup matter? (measured, not assumed)
    if 'sky_persist_nearest' in data:
        near = data['sky_persist_nearest']
        print('\nsky lookup: nearest pixel against bilinear interpolation')
        for label, s in (('nearest ', np.log(np.maximum(det, 0) + eps)
                                     - np.log(np.maximum(near, 0) + eps)),
                         ('bilinear', np.log(np.maximum(det, 0) + eps)
                                     - np.log(np.maximum(sky, 0) + eps))):
            r = score_rule(s, labels)
            print(f'  {label}: AP {r["ap"]:.3f}, P@95 {r[0.95][0]:6.1%}, '
                  f'recall at zero false positives {r["clean_recall"]:6.1%}')

    # ---- what are the disagreements, really?
    name, _, score = best
    order = np.argsort(-score)
    disagree = order[~labels[order]][:12]
    print('\nthe twelve highest-scoring candidates the dark mask does NOT call hot:')
    print(f'  {"row":>6} {"col":>6} {"peak(ADU)":>10} {"detector":>10} {"sky":>10}')
    for i in disagree:
        print(f'  {data["rows"][i]:6d} {data["cols"][i]:6d} {peak0[i]:10.0f} '
              f'{det[i]:10.1f} {sky[i]:10.1f}')
    print('  Checked against the detected stars: with a nearest-pixel sky lookup, 13 of')
    print('  the top 14 of these sat 3.5 to 8.5 px from a real star -- they were stellar')
    print('  wings, not hot pixels, and the fault was in the sampling rather than in the')
    print('  dark mask. Sub-pixel dither rounded to the nearest pixel puts the weakest')
    print('  sample somewhere down a steep stellar flank. Interpolating fixes it.')

    # ---- figure: the 2-D plane with the best rule's boundary
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    star = ~labels
    ax.scatter(sky[star], det[star], s=7, alpha=0.35, c='#1f77b4',
               label=f'stars and other detections ({int(star.sum())})')
    ax.scatter(sky[labels], det[labels], s=30, alpha=0.9, c='#d62728', marker='x',
               label=f'dark-confirmed hot ({int(labels.sum())})')
    # the operating point that catches 95% of them
    r = score_rule(score, labels)
    ax.set_xscale('symlog', linthresh=10)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('sky persistence (sigma)  -- high for a star')
    ax.set_ylabel('detector persistence (sigma)  -- high for a hot pixel')
    ax.set_title('Hot pixels and stars occupy opposite corners\n'
                 'neither axis uses a dark frame')
    ax.axhline(0, c='#bbb', lw=0.8); ax.axvline(0, c='#bbb', lw=0.8)
    ax.legend(loc='upper center', fontsize=9)
    ax.grid(alpha=0.2, which='both')
    fig.tight_layout()
    fig.savefig(args.directory / 'persistence_plane.png', dpi=140)
    plt.close(fig)

    # ---- figure: precision-recall for every rule
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for rule_name, rule_score in rules.items():
        o = np.argsort(-rule_score)
        h = labels[o]
        tp, fp = np.cumsum(h), np.cumsum(~h)
        ax.plot(tp / max(int(labels.sum()), 1), tp / np.maximum(tp + fp, 1),
                lw=1.8, label=rule_name)
    ax.set_xlabel('recall'); ax.set_ylabel('precision')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title('Dark-free rules, scored against the dark-based mask')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.directory / 'rules_precision_recall.png', dpi=140)
    plt.close(fig)
    print(f'\nfigures written to {args.directory}')


if __name__ == '__main__':
    main()
