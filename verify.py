#!/usr/bin/env python3
"""Check dcm2niix's NIfTI output in Out/ against gen_rgb.expected(): pixels,
not only a previous run. Lossless cases must match exactly. Lossy JPEG 2000
must equal OpenJPEG's decode (within 1); lossy JPEG must be as close to the
truth as DCMTK's decode (within 0.5 mean). Exit 1 on any mismatch.

usage: verify.py [Out]
"""
import glob
import os
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import pydicom
from pydicom.encaps import generate_frames

from gen_rgb import FRAMES, expected

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Out")

# case -> (expected rows x cols x 3 [x frames], max abs error, mean abs error)
CASES = {
    "rgb_interleaved": (expected(), 0, 0),
    "rgb_planar": (expected(), 0, 0),
    "rgb_implicit": (expected(), 0, 0),
    "ybr_full": (expected(), 1, 0.4),  # YBR_FULL rounded, then back to RGB
    "rgb_multiframe": (np.stack([expected(f) for f in range(FRAMES)], -1), 0, 0),
    # dcm2niix does not apply PALETTE COLOR: the NIfTI holds the stored indices.
    "palette": (expected()[..., 0], 0, 0),
    "jpeg_lossless_sv1": (expected(), 0, 0),
    "jpeg_baseline_ybr422": (expected(), 90, 6.0),  # lossy, 4:2:2 chroma on sharp edges (measured 81, 4.7)
    "jpegls_lossless": (expected(), 0, 0),
    "rle": (expected(), 0, 0),
    "j2k_rct": (expected(), 0, 0),
    "j2k_rgb_nomct": (expected(), 0, 0),
    "j2k_ict": (expected(), 50, 6.0),  # lossy 9/7 at 8:1 (measured 36, 4.6)
}


def dcmdjpeg(case):
    """DCMTK's decode of a JPEG case, as rows x cols x 3 RGB."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "d.dcm")
        subprocess.run(["dcmdjpeg", os.path.join(HERE, "In", case, case + ".dcm"), out], check=True)
        return pydicom.dcmread(out).pixel_array


def opj(case):
    """OpenJPEG's decode of a JPEG 2000 case (it inverts the MCT itself)."""
    ds = pydicom.dcmread(os.path.join(HERE, "In", case, case + ".dcm"))
    with tempfile.TemporaryDirectory() as tmp:
        j2k, ppm = os.path.join(tmp, "f.j2k"), os.path.join(tmp, "f.ppm")
        with open(j2k, "wb") as f:
            f.write(next(generate_frames(ds.PixelData)))
        subprocess.run(["opj_decompress", "-i", j2k, "-o", ppm], check=True, stdout=subprocess.DEVNULL)
        raw = open(ppm, "rb").read()
    return np.frombuffer(raw[-ds.Rows * ds.Columns * 3 :], np.uint8).reshape(ds.Rows, ds.Columns, 3)


# Lossy cases against an independent decoder. JPEG 2000 decoding is exact, so
# dcm2niix must equal OpenJPEG. Baseline JPEG decoders may upsample 4:2:2 chroma
# differently (dcm2niix and DCMTK differ on ~70% of pixels), so dcm2niix must
# instead be as close to the truth as DCMTK is (mean error 4.7 vs 4.5).
EXACT = {"j2k_ict": opj}
AS_GOOD = {"jpeg_baseline_ybr422": dcmdjpeg}


def rgb_of(path):
    """NIfTI RGB24 as rows x cols x 3 [x slices], in DICOM row order."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    if data.dtype.names:  # RGB24 comes back as a structured array
        data = np.stack([data[n] for n in data.dtype.names], -1)
    # x, y, [z,] rgb -> y, x, rgb [, z]; dcm2niix stores DICOM row 0 last (NIfTI is bottom-up).
    data = np.squeeze(data)
    if data.ndim == 4:
        data = np.moveaxis(data, 2, -1)
    return np.flip(np.swapaxes(data, 0, 1), 0)


def main():
    failed = 0
    for case, (want, max_tol, mean_tol) in CASES.items():
        hits = glob.glob(os.path.join(OUT, f"{case}_*.nii"))
        if len(hits) != 1:
            print(f"FAIL {case}: {len(hits)} outputs")
            failed += 1
            continue
        got = rgb_of(hits[0])
        if got.shape != want.shape:
            print(f"FAIL {case}: shape {got.shape}, want {want.shape}")
            failed += 1
            continue
        d = np.abs(got.astype(np.int32) - want.astype(np.int32))
        ok = d.max() <= max_tol and d.mean() <= mean_tol
        note = ""
        if case in EXACT:
            e = np.abs(got.astype(np.int32) - EXACT[case](case).astype(np.int32)).max()
            ok = ok and e <= 1
            note = f"; vs {EXACT[case].__name__} max {e} (<= 1)"
        if case in AS_GOOD:
            theirs = np.abs(AS_GOOD[case](case).astype(np.int32) - want.astype(np.int32)).mean()
            ok = ok and d.mean() <= theirs + 0.5
            note = f"; {AS_GOOD[case].__name__} mean {theirs:.2f} vs truth (dcm2niix <= +0.5)"
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {case}: max {d.max()} (<= {max_tol}), mean {d.mean():.2f} (<= {mean_tol}){note}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
