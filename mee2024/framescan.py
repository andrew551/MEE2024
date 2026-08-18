"""
Which frames of a sequence are usable, and where the transients are.

An eclipse capture rarely starts and stops on the science. The Sun is there before totality
and back afterwards, so a sequence can open saturated, settle into the corona, and end
saturated again -- or end in blank frames where the capture was cut. Trimming that has been
a manual step in a separate program, which costs a duplicate of a 15 GB file and loses the
container's metadata on the way through.

This measures the sequence and *suggests* a range. It never edits anything: the answer is a
pair of indices the run records, so the trim is a parameter rather than a second copy of the
data, and changing your mind costs nothing.

**The trap worth knowing about.** The obvious rule -- "drop frames until the picture stops
changing" -- fails, and fails silently. While a frame is saturated its median is pinned at
full scale and the frame-to-frame difference is exactly zero, so a search for "settled"
starting at the beginning stops on the *first* frame and keeps the entire saturated run.
Saturation is indistinguishable from perfect stability by that measure. So stability alone
is not enough: a usable frame must also not be saturated well above the sequence's own
floor, which is what separates a pinned photosphere from a corona that legitimately clips a
few percent of the frame at every exposure.

The second obvious rule -- "drop all-black and all-white frames" -- also fails. Measured on
a real capture it got the blank tail exactly right and the saturated head completely wrong,
because the frames either side of second contact are not *all* white; they decay smoothly
through every value in between.

Both ends are treated the same way. Nothing here assumes the Sun is at the start.
"""

from pathlib import Path

import numpy as np

#: Floor under the settled-frame test, as a fraction. The test itself is *adaptive* (see
#: :func:`suggest`) because a fixed per-frame threshold cannot work across exposures: at
#: 315 ms, consecutive frames are a third of a second apart and a stable stretch changes by
#: well under 1% a frame; at 2 s they are 2.4 s apart and the same sky changes 1.3-2.6% a
#: frame, which a fixed 1% cut rejects as unusable -- measured, on a real calibration
#: sequence where it discarded every frame. This floor only stops the adaptive threshold
#: collapsing on a sequence that is unusually uniform.
CHANGE_TOLERANCE = 0.01

#: How far above the sequence's typical frame-to-frame change a frame must sit to count as
#: part of a transient. The usual robust outlier distance -- median plus three scaled MADs.
CHANGE_SIGMAS = 3.0

#: How far above the sequence's own saturation floor a frame may sit and still be usable, in
#: percentage points of the frame. A totality exposure legitimately clips a few percent --
#: the inner corona -- at every frame, so the test has to be relative to that floor rather
#: than to zero.
SATURATION_TOLERANCE = 0.02

#: A frame this empty is blank: the capture was cut, or the camera returned nothing.
BLANK_FRACTION = 0.99

#: Rows sampled through the middle of each frame. Enough to measure a level, few enough that
#: scanning a 22 GB container costs a few hundred megabytes of reads rather than all of it.
STRIP_ROWS = 48


def _sample_ser(container, index, rows):
    """A horizontal strip through the middle of one SER frame, without reading the frame."""
    from mee2024 import ser

    handle = ser.open_ser(container)
    header = handle.header
    if header['planes'] != 1:
        return handle.read(index)[::8, ::8]
    width, height = header['width'], header['height']
    rows = min(rows, height)
    top = (height - rows) // 2
    itemsize = header['bytes_per_sample']
    with open(container, 'rb') as fp:
        fp.seek(ser.HEADER_BYTES + index * header['frame_bytes']
                + top * width * itemsize)
        raw = fp.read(rows * width * itemsize)
    return np.frombuffer(raw, dtype=handle.dtype).reshape(rows, width)


def sample(frame, rows=STRIP_ROWS):
    """A representative sample of one frame, cheaply."""
    from mee2024 import ser

    if ser.is_ser(frame):
        container, index = ser.parse_ref(frame)
        return _sample_ser(container, index or 0, rows)
    from mee2024.stacker_implementation import open_image
    return open_image(frame)[::4, ::4]


def scan(frames, full_scale=None, progress=None, rows=STRIP_ROWS):
    """Per-frame level statistics for a sequence.

    Returns a list of dicts with ``median``, ``saturated`` and ``blank`` fractions, in the
    order given. ``full_scale`` defaults to the largest value seen, which is right for the
    16-bit data this handles and avoids assuming a depth the frames have not stated.
    """
    frames = [str(f) for f in frames]
    if progress is not None and hasattr(progress, 'start'):
        progress.start(len(frames), 'Measuring frames')
    samples = []
    peak = 0.0
    for index, frame in enumerate(frames):
        data = np.asarray(sample(frame, rows))
        samples.append(data)
        peak = max(peak, float(data.max()) if data.size else 0.0)
        if progress is not None:
            progress.update(index + 1)
    if progress is not None and hasattr(progress, 'finish'):
        progress.finish()

    ceiling = float(full_scale) if full_scale else peak
    out = []
    for index, data in enumerate(samples):
        data = np.asarray(data, dtype=np.float64)
        size = max(data.size, 1)
        out.append({
            'index': index, 'frame': frames[index],
            'median': float(np.median(data)) if data.size else 0.0,
            'mean': float(data.mean()) if data.size else 0.0,
            'max': float(data.max()) if data.size else 0.0,
            'saturated': float(np.count_nonzero(data >= ceiling) / size) if ceiling else 0.0,
            'blank': float(np.count_nonzero(data <= 0) / size),
        })
    return out


def suggest(levels, change_tolerance=CHANGE_TOLERANCE,
            saturation_tolerance=SATURATION_TOLERANCE, blank_fraction=BLANK_FRACTION):
    """The usable range of a scanned sequence: ``(start, stop, info)``, stop inclusive.

    A frame is usable when it is not blank, not saturated far above the sequence's own
    floor, and not in the middle of a rapid change. The answer is the **longest run** of
    such frames, which is what makes this symmetric: whichever end the transient is at, the
    science is the long stable stretch in between.

    ``info`` says what was excluded and why, because a silent trim is worse than none.
    """
    n = len(levels)
    if n == 0:
        return None, None, {'reason': 'no frames'}
    median = np.array([lv['median'] for lv in levels], dtype=np.float64)
    saturated = np.array([lv['saturated'] for lv in levels], dtype=np.float64)
    blank = np.array([lv['blank'] for lv in levels], dtype=np.float64) >= blank_fraction

    live = ~blank
    if not live.any():
        return None, None, {'reason': 'every frame is blank'}

    # the sequence's own saturation floor: what it clips even when behaving. The 10th
    # percentile rather than the minimum, so one unusually clean frame cannot set the bar
    floor = float(np.percentile(saturated[live], 10))
    over = saturated > floor + saturation_tolerance

    change = np.zeros(n)
    if n > 1:
        change[1:] = np.abs(np.diff(median)) / np.maximum(median[:-1], 1.0)
        change[0] = change[1]

    # The threshold calibrates itself against the sequence's own typical change, because
    # what counts as "settled" depends on the exposure and cadence, not on an absolute
    # number. Measured on the frames that are neither blank nor saturated -- including the
    # transient would inflate the scale until nothing looked like an outlier.
    usable = live & ~over
    reference = change[usable] if usable.any() else change
    typical = float(np.median(reference)) if reference.size else 0.0
    spread = 1.4826 * float(np.median(np.abs(reference - typical))) if reference.size else 0.0
    threshold = max(change_tolerance, typical + CHANGE_SIGMAS * spread)
    settled = change <= threshold

    good = live & ~over & settled
    if not good.any():
        return None, None, {'reason': 'no frame is both settled and unsaturated',
                            'saturation_floor': floor}

    # the longest contiguous run of usable frames
    best_start = best_len = run_start = run_len = 0
    for i, ok in enumerate(good):
        if ok:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    start, stop = best_start, best_start + best_len - 1

    info = {
        'n_frames': n, 'kept': best_len,
        'dropped_leading': start, 'dropped_trailing': n - 1 - stop,
        'saturation_floor': floor,
        'n_blank': int(blank.sum()),
        'n_saturated': int((over & live).sum()),
        'n_unsettled': int((~settled & live & ~over).sum()),
        'change_tolerance': float(threshold),
        'typical_change': typical,
    }
    return start, stop, info


def describe(start, stop, info):
    """One line for the log, saying what was left out and why."""
    if start is None:
        return f'could not suggest a frame range: {info.get("reason", "unknown")}'
    bits = [f'frames {start}-{stop} of {info["n_frames"]} look usable']
    if info['dropped_leading'] or info['dropped_trailing']:
        bits.append(f'{info["dropped_leading"]} dropped at the start, '
                    f'{info["dropped_trailing"]} at the end')
    reasons = []
    if info['n_blank']:
        reasons.append(f'{info["n_blank"]} blank')
    if info['n_saturated']:
        reasons.append(f'{info["n_saturated"]} saturated above the '
                       f'{100 * info["saturation_floor"]:.1f}% floor')
    if info['n_unsettled']:
        reasons.append(f'{info["n_unsettled"]} still changing')
    if reasons:
        bits.append('(' + ', '.join(reasons) + ')')
    return '; '.join(bits)


# ------------------------------------------- does a frame's level match what it claims?

#: How far a frame's level may sit from its exposure group's typical value before it is
#: worth a warning, in robust standard deviations. Loose, because sky brightness genuinely
#: drifts through a sequence and the point is to catch a frame carrying the *wrong
#: exposure's pixels*, not to police ordinary variation.
LEVEL_SIGMAS = 6.0

#: With fewer frames than this in an exposure group there is nothing to compare against.
MIN_GROUP = 3

#: How many same-exposure neighbours the backstop compares a frame against. Local rather
#: than whole-group, so a sky that brightens steadily through a sequence is seen as the
#: trend it is instead of as a hundred anomalous frames.
NEIGHBOURS = 8

#: Beyond this many, the messages stop being useful and start being a wall. Past it the
#: sequence itself is changing and the frame range is the thing to look at.
MAX_MESSAGES = 12


def stated_exposure(frame):
    """The exposure a frame's header claims, or None."""
    from mee2024 import ser

    try:
        if ser.is_ser(frame):
            container, index = ser.parse_ref(frame)
            value = ser.open_ser(container).fits_header(index or 0).get('EXPTIME')
            return float(value) if value is not None else None
        from astropy.io import fits
        with fits.open(frame) as hdul:
            header = hdul['PRIMARY'].header if 'PRIMARY' in hdul else hdul[0].header
        value = header.get('EXPTIME', header.get('EXPOSURE'))
        return float(value) if value not in (None, '') else None
    except Exception:
        return None


def check_exposures(frames, levels=None, sigmas=LEVEL_SIGMAS):
    """Frames whose brightness disagrees with the exposure their header states.

    Capture software that changes exposure mid-sequence can write the *new* exposure into
    the header of a frame that still holds the *previous* exposure's pixels -- the camera
    had not applied the change when the frame was read out. Six frames of one real eclipse
    ladder are like that. Nothing downstream can detect it: the file is well-formed, the
    header is self-consistent, and the frame simply carries half the signal it claims. It
    would then be stacked into the wrong exposure tier and matched to the wrong master dark.

    The test is comparison, not physics: group the frames by the exposure they *claim*, and
    ask whether each one's level fits its group. When a frame fits some *other* group better,
    say which -- that is the signature of this fault and it names the likely true exposure.

    Deliberately reports rather than corrects. The data is scarce enough to be worth keeping
    whatever its labels say, and an automatic relabel would be a silent edit of somebody's
    science. It also serves a second purpose: a run that has not been checked by hand says
    so, in the log, before anyone trusts its output.

    Returns a list of message strings, empty when everything is consistent.

    Two tests, because one is not enough. The **transition** test is the targeted one: at
    every point where the stated exposure changes, ask whether the new frame's level looks
    like the frame *before* it rather than like its own exposure. That is local -- the two
    frames are seconds apart -- so a sky that brightens through the sequence cannot hide the
    fault. A whole-group comparison can and does: on one real ladder the sky rose enough
    within a single exposure group to inflate its spread until a mislabelled member sat
    comfortably inside it.

    The **group** test is the backstop, for a frame that is simply wrong without being at a
    transition -- cloud, or a one-off.
    """
    frames = [str(f) for f in frames]
    if len(frames) < MIN_GROUP:
        return []
    if levels is None:
        levels = scan(frames)
    exposures = [stated_exposure(f) for f in frames]
    if not any(e is not None for e in exposures):
        return []

    groups = {}
    for index, exposure in enumerate(exposures):
        if exposure is not None:
            groups.setdefault(round(float(exposure), 6), []).append(index)
    level_of = [lv['median'] for lv in levels]

    def group_centre(exposure, exclude=()):
        members = [i for i in groups.get(exposure, []) if i not in exclude]
        if not members:
            return None
        return float(np.median([level_of[i] for i in members]))

    messages = []
    flagged = set()

    # --- the transition test
    for index in range(1, len(frames)):
        here, before = exposures[index], exposures[index - 1]
        if here is None or before is None or here == before:
            continue
        level = level_of[index]
        own = group_centre(round(here, 6), exclude={index})
        if own is None:
            continue
        previous = level_of[index - 1]
        # it should look like its own exposure, not like the frame before it
        if abs(level - previous) < abs(level - own) and abs(level - own) > 0.15 * max(own, 1):
            flagged.add(index)
            messages.append(
                f'{Path(frames[index]).name} says EXPTIME {here:g} s, but it is the first '
                f'frame after a change from {before:g} s and its level ({level:.0f}) looks '
                f'like the {before:g} s frame before it ({previous:.0f}) rather than like '
                f'the other {here:g} s frames ({own:.0f}). Capture software can write the '
                f'new exposure into the header of a frame that still holds the previous '
                f'one. Check it before this frame is stacked or dark-matched.')

    # --- the group backstop, compared LOCALLY
    #
    # Against the whole group this drowns any sequence whose sky is changing: on one real
    # capture the sky brightened steadily from first frame to last, so most of the frames
    # sat far from the group median and every one of them was reported -- thirty warnings
    # about a field where nothing was wrong. A brightening sky is a trend, not a fault. An
    # isolated frame at the wrong exposure is different: it stands out from the frames
    # either side of it, whatever the sky is doing.
    for exposure, members in sorted(groups.items()):
        if len(members) < NEIGHBOURS + 1:
            continue
        for position, index in enumerate(members):
            if index in flagged:
                continue
            lo = max(0, position - NEIGHBOURS // 2)
            nearby = [i for i in members[lo:lo + NEIGHBOURS + 1] if i != index]
            if len(nearby) < MIN_GROUP:
                continue
            values = np.array([level_of[i] for i in nearby], dtype=np.float64)
            centre = float(np.median(values))
            spread = max(1.4826 * float(np.median(np.abs(values - centre))),
                         max(abs(centre), 1.0) * 0.05)
            if abs(level_of[index] - centre) <= sigmas * spread:
                continue
            flagged.add(index)
            messages.append(
                f'{Path(frames[index]).name} says EXPTIME {exposure:g} s but its level '
                f'({level_of[index]:.0f}) is out of step with the {exposure:g} s frames '
                f'around it ({centre:.0f}). It may have been taken through cloud, at a '
                f'different exposure, or during a transient.')
    if len(messages) > MAX_MESSAGES:
        extra = len(messages) - MAX_MESSAGES
        messages = messages[:MAX_MESSAGES]
        messages.append(f'...and {extra} more frame(s) whose level disagrees with the '
                        f'exposure stated. That many suggests the sequence itself is '
                        f'changing rather than any one frame being mislabelled; check the '
                        f'suggested frame range instead.')
    return messages


def parse_range(text, n_frames=None):
    """``'50-172'`` or ``'50:172'`` or ``'50'`` -> ``(start, stop)``, 0-based and inclusive.

    Returns ``(None, None)`` for empty input. Raises ValueError on anything unparseable, so
    a typo stops the run rather than silently processing everything.
    """
    if text in (None, '', 'all'):
        return None, None
    raw = str(text).strip().replace(':', '-')
    parts = [p.strip() for p in raw.split('-') if p.strip() != '']
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f'cannot read {text!r} as a frame range; use e.g. 50-172')
    if len(values) == 1:
        start = stop = values[0]
    elif len(values) == 2:
        start, stop = values
    else:
        raise ValueError(f'cannot read {text!r} as a frame range; use e.g. 50-172')
    if start > stop:
        raise ValueError(f'frame range {text!r} runs backwards')
    if n_frames is not None:
        if start >= n_frames:
            raise ValueError(f'frame range {text!r} starts past the end of a '
                             f'{n_frames}-frame sequence')
        stop = min(stop, n_frames - 1)
    return start, stop


def apply_range(frames, start, stop):
    """The frames a range selects. A missing range selects everything."""
    if start is None:
        return list(frames)
    return list(frames)[start:stop + 1]
