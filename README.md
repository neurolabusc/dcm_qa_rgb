## About

Tiny synthetic colour DICOMs (64×48, 140 KB in all) for regression-testing dcm2niix's colour paths. Unlike most dcm_qa suites, the expected pixels are known: every case is built from one deterministic RGB pattern (`gen_rgb.py` `expected()`). `verify.py` therefore checks dcm2niix's output against the truth, not only against an earlier run.
- Red ramps with x and green with y.
- Blue is a checkerboard.
- The top rows hold red, green, blue, white and black swatches, so a channel swap or a planar mix-up is unmistakable.

| Case | Encoding | Expected |
|---|---|---|
| `rgb_interleaved`, `rgb_planar`, `rgb_implicit` | uncompressed RGB, PlanarConfiguration 0/1, explicit/implicit VR | exact |
| `ybr_full` | uncompressed YBR_FULL | within 1 (rounding) |
| `rgb_multiframe` | 3 frames | exact, 3 slices |
| `palette` | PALETTE COLOR, 8-bit indices | the indices: dcm2niix does not apply the palette |
| `jpeg_lossless_sv1` | DCMTK `dcmcjpeg +e1` | exact |
| `jpeg_baseline_ybr422` | DCMTK `dcmcjpeg +eb` (YBR_FULL_422) | lossy: within 24 of DCMTK's `dcmdjpeg` (mean ≤ 2.5; the two decoders upsample chroma differently and are equally close to the truth) |
| `jpegls_lossless` | DCMTK `dcmcjpls +el` | exact |
| `rle` | DCMTK `dcmcrle` | exact |
| `j2k_rct`, `j2k_rgb_nomct` | OpenJPEG lossless with RCT, and without MCT (RGB) | exact |
| `j2k_ict` | OpenJPEG lossy 9/7 with ICT | equal to `opj_decompress` (within 1) |

There is no 16-bit RGB case: dcm2niix refuses one ("Unsupported DICOM bit-depth 16 with 3 samples per pixel") and exits non-zero, which would stop `batch.sh`. DICOM-WSI tile colour, including ICC profiles, is tested in `wsi_frames` and in `@niivue/dcm2niix`'s `npm test`: dcm2niix's NIfTI output is never colour-managed.

`In/` was written by `gen_rgb.py` (pydicom 3.0, numpy, DCMTK 3.7.0, OpenJPEG 2.5.4). UIDs are derived from the case names, so rerunning it rewrites the same files, given the same encoder versions.

## Running

Run `batch.sh` with dcm2niix built with OpenJPEG and JPEG-LS (the CMake build); plain `make` lacks OpenJPEG.
1. It converts `In/` to `Out/` and diffs against `Ref/`.
2. It then runs `verify.py`, which needs Python with nibabel and pydicom, plus `dcmdjpeg` and `opj_decompress` on the PATH.
