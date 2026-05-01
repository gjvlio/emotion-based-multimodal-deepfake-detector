import h5py

files = {
    "COVAREP":   "acoustics/CMU_MOSEI_COVAREP.csd",
    "Labels":    "labels/CMU_MOSEI_Labels.csd",
    "Phones":    "languages/CMU_MOSEI_TimestampedPhones.csd",
    "Words":     "languages/CMU_MOSEI_TimestampedWords.csd",
    "GloVe":     "languages/CMU_MOSEI_TimestampedWordVectors.csd",
    "Facet42":   "visuals/CMU_MOSEI_VisualFacet42.csd",
    "OpenFace2": "visuals/CMU_MOSEI_VisualOpenFace2.csd",
}

for name, path in files.items():
    try:
        with h5py.File(path, "r") as f:
            root = list(f.keys())[0]
            n = len(f[root]["data"])
            sample_id = list(f[root]["data"].keys())[0]
            shape = f[root]["data"][sample_id]["features"].shape
            print(f"{name:10s} [OK]   videos={n:5d}  sample dim={shape[1]}")
    except Exception as e:
        print(f"{name:10s} [FAIL] {e}")