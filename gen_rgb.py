#!/usr/bin/env python3
"""Write dcm_qa_rgb/In: tiny synthetic colour DICOMs with known pixels.

Every case is built from one deterministic RGB pattern (expected()), so
verify.py can check dcm2niix's NIfTI output against ground truth, not only
against a previous run. UIDs are derived from the case name, so rerunning
gives byte-identical files (given the same encoder versions).

Needs pydicom 3, numpy, and on PATH: dcmcjpeg, dcmcjpls, dcmcrle (DCMTK 3.7)
and opj_compress (OpenJPEG 2.5).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000,
    JPEG2000Lossless,
    SecondaryCaptureImageStorage,
    generate_uid,
)

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "In")
ROWS, COLS = 48, 64
FRAMES = 3  # for the multi-frame case


def expected(frame=0):
    """The RGB pattern: rows x cols x 3. Red ramps with x, green with y, blue is
    a checkerboard; frame shifts blue. The top row holds pure primaries, white
    and black, so a channel swap or a planar mix-up is unmistakable."""
    y, x = np.mgrid[0:ROWS, 0:COLS]
    rgb = np.zeros((ROWS, COLS, 3), np.float64)
    rgb[..., 0] = x * 4
    rgb[..., 1] = y * 5
    rgb[..., 2] = ((x // 8 + y // 8) % 2) * 160 + 20 + frame * 30
    swatches = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
    for i, c in enumerate(swatches):
        rgb[0:4, i * 8 : i * 8 + 8] = c
    return rgb.astype(np.uint8)


def rgb_to_ybr_full(rgb):
    """PS3.3 C.7.6.3.1.2 YBR_FULL (JFIF), rounded."""
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return np.clip(np.rint(np.stack([y, cb, cr], -1)), 0, 255).astype(np.uint8)


def base(name, series, ts=ExplicitVRLittleEndian):
    uid = lambda what: generate_uid(entropy_srcs=["dcm_qa_rgb", name, what])
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = uid("sop")
    meta.TransferSyntaxUID = ts
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid(entropy_srcs=["dcm_qa_rgb", "study"])
    ds.SeriesInstanceUID = uid("series")
    ds.FrameOfReferenceUID = generate_uid(entropy_srcs=["dcm_qa_rgb", "for"])
    ds.PatientName = "dcm_qa_rgb"
    ds.PatientID = "dcm_qa_rgb"
    ds.StudyDate = "20261106"
    ds.Modality = "OT"
    ds.Manufacturer = "dcm_qa_rgb"
    ds.SeriesNumber = series
    ds.InstanceNumber = 1
    ds.SeriesDescription = name
    ds.ProtocolName = name
    ds.PatientPosition = "HFS"
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 1
    ds.Rows, ds.Columns = ROWS, COLS
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    return ds


def save(ds, name):
    os.makedirs(os.path.join(IN, name), exist_ok=True)
    path = os.path.join(IN, name, name + ".dcm")
    ds.save_as(path, enforce_file_format=True)
    return path


def tool(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def j2k(rgb, name, series, lossless, mct):
    """Encode with opj_compress and encapsulate: RCT (lossless) or ICT (lossy)
    when mct, else RGB components coded independently."""
    with tempfile.TemporaryDirectory() as tmp:
        ppm = os.path.join(tmp, "in.ppm")
        with open(ppm, "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (COLS, ROWS) + rgb.tobytes())
        out = os.path.join(tmp, "out.j2k")
        cmd = ["opj_compress", "-i", ppm, "-o", out, "-mct", "1" if mct else "0"]
        if not lossless:
            cmd += ["-I", "-r", "8"]
        tool(*cmd)
        codestream = open(out, "rb").read()
    ds = base(name, series, JPEG2000Lossless if lossless else JPEG2000)
    ds.PhotometricInterpretation = ("YBR_RCT" if lossless else "YBR_ICT") if mct else "RGB"
    ds.PixelData = encapsulate([codestream])
    ds["PixelData"].VR = "OB"
    if not lossless:
        ds.LossyImageCompression = "01"
    save(ds, name)


def main():
    for t in ("dcmcjpeg", "dcmcjpls", "dcmcrle", "opj_compress"):
        if not shutil.which(t):
            sys.exit(f"{t} not on PATH")
    shutil.rmtree(IN, ignore_errors=True)
    rgb = expected()

    # Uncompressed.
    ds = base("rgb_interleaved", 1)
    ds.PixelData = rgb.tobytes()
    src = save(ds, "rgb_interleaved")

    ds = base("rgb_planar", 2)
    ds.PlanarConfiguration = 1
    ds.PixelData = np.ascontiguousarray(rgb.transpose(2, 0, 1)).tobytes()
    save(ds, "rgb_planar")

    ds = base("rgb_implicit", 3, ImplicitVRLittleEndian)
    ds.PixelData = rgb.tobytes()
    save(ds, "rgb_implicit")

    ds = base("ybr_full", 4)
    ds.PhotometricInterpretation = "YBR_FULL"
    ds.PixelData = rgb_to_ybr_full(rgb).tobytes()
    save(ds, "ybr_full")

    # No 16-bit RGB case (see README).
    ds = base("rgb_multiframe", 6)
    ds.NumberOfFrames = FRAMES
    ds.PixelData = b"".join(expected(f).tobytes() for f in range(FRAMES))
    save(ds, "rgb_multiframe")

    # PALETTE COLOR: 8-bit indices into 16-bit LUTs; index i -> (i, 255 - i, i // 2) * 257.
    ds = base("palette", 7)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "PALETTE COLOR"
    del ds.PlanarConfiguration
    index = expected()[..., 0]  # the red ramp, 0..252
    ds.PixelData = index.tobytes()
    lut = np.arange(256)
    for tag, values in (("Red", lut), ("Green", 255 - lut), ("Blue", lut // 2)):
        setattr(ds, f"{tag}PaletteColorLookupTableDescriptor", [256, 0, 16])
        setattr(ds, f"{tag}PaletteColorLookupTableData", (values * 257).astype("<u2").tobytes())
    save(ds, "palette")

    # Compressed, from rgb_interleaved by DCMTK.
    for name, series, cmd in (
        ("jpeg_lossless_sv1", 8, ["dcmcjpeg", "+e1"]),
        ("jpeg_baseline_ybr422", 9, ["dcmcjpeg", "+eb"]),  # lossy, YBR_FULL_422
        ("jpegls_lossless", 10, ["dcmcjpls", "+el"]),
        ("rle", 11, ["dcmcrle"]),
    ):
        os.makedirs(os.path.join(IN, name), exist_ok=True)
        dst = os.path.join(IN, name, name + ".dcm")
        tool(*cmd, src, dst)
        # Distinct series: patch the copied identity with pydicom.
        ds = dcmread(dst)
        ds.SeriesNumber, ds.SeriesDescription, ds.ProtocolName = series, name, name
        ds.SeriesInstanceUID = generate_uid(entropy_srcs=["dcm_qa_rgb", name, "series"])
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid(
            entropy_srcs=["dcm_qa_rgb", name, "sop"]
        )
        ds.save_as(dst, enforce_file_format=True)

    # JPEG 2000, from the pattern by OpenJPEG.
    j2k(rgb, "j2k_rct", 12, lossless=True, mct=True)
    j2k(rgb, "j2k_rgb_nomct", 13, lossless=True, mct=False)
    j2k(rgb, "j2k_ict", 14, lossless=False, mct=True)
    print(f"wrote {len(os.listdir(IN))} cases to {IN}")


if __name__ == "__main__":
    main()
