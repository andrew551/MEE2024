"""Publishing charts into RECORD without destroying what was there.

CLAUDE.md: "Never overwrite a chart revision."  Cells 1-3 honour that with the construction
in `tools/step3_charts_record.py` -- anything in RECORD that DIFFERS from what is about to be
written is moved into a dated `superseded_*` folder first, and files that are byte-identical
are left alone so a re-run cannot bury identical copies under a new date.

This module exists because cell 4's chart tools wrote straight into RECORD and did overwrite a
revision: `record_covariance.png` was written once from a broken parse (it read L = 64 "/px)
and then overwritten by the corrected chart.  The broken one is not recoverable -- there was
no copy -- and it is recorded here as the reason this helper exists rather than quietly fixed.

    publish(['record_covariance.png'], src_dir)
"""
import filecmp
import os
import shutil
import datetime

RECORD = r'D:\MEE2024 output\MEE_output\RECORD\husillos2026'


def publish(names, src, record=RECORD):
    """Copy `names` from `src` into the record, superseding anything different first.

    Returns the supersede directory used, or None if nothing had to be moved.
    """
    os.makedirs(record, exist_ok=True)
    old = [f for f in names
           if os.path.exists(os.path.join(record, f))
           and not (os.path.exists(os.path.join(src, f))
                    and filecmp.cmp(os.path.join(src, f), os.path.join(record, f),
                                    shallow=False))]
    sup = None
    if old:
        sup = os.path.join(record, 'superseded_'
                           + datetime.datetime.now().strftime('%Y-%m-%d_%H%M'))
        os.makedirs(sup, exist_ok=True)
        for f in old:
            shutil.move(os.path.join(record, f), os.path.join(sup, f))
            print('   superseded: %s -> %s' % (f, os.path.basename(sup)))
    for f in names:
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(record, f))
            print('   published:  %s' % os.path.join(record, f))
    return sup
