# libdmtx shim (preserved, NOT in the shipped package)

Retired from the package in the zxing re-anchor, kept here for **ensemble
experiments** (libdmtx is complementary to zxing on ink-thickened images —
it decoded scan_432 that zxing missed). Not a runtime dependency.

To use: copy `binding.py` + `_build_dmtx.py` into `src/dmtxslide/`, then
`CPATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib python -m dmtxslide._build_dmtx`
and `mv dmtxslide/_dmtx.*.so src/dmtxslide/`. Needs `brew install libdmtx`.
