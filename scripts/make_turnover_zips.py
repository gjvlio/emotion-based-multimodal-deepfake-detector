import os
import zipfile
from pathlib import Path

# Workspace paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "turnover_zips"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Creating zips in: {OUTPUT_DIR}\n")

    # Define zipping specs
    # Format: (zip_filename, [ list of (source_path, arc_root) ], [ list of exclusion keywords ])
    specs = [
        (
            "crema_d_raw.zip",
            [
                (DATA_DIR / "raw" / "CREMA-D", "data/raw/CREMA-D"),
                (DATA_DIR / "processed" / "actor_portraits" / "actor_portraits_manifest.csv", "data/processed/actor_portraits/actor_portraits_manifest.csv")
            ],
            [".git", ".gitattributes", ".gitignore"]
        ),
        (
            "tracks_1_2_3_4.zip",
            [
                (DATA_DIR / "synthetic" / "track1_fakes", "data/synthetic/track1_fakes"),
                (DATA_DIR / "synthetic" / "track2_fakes", "data/synthetic/track2_fakes"),
                (DATA_DIR / "synthetic" / "track3_fakes", "data/synthetic/track3_fakes"),
                (DATA_DIR / "synthetic" / "track4_fakes", "data/synthetic/track4_fakes"),
                (DATA_DIR / "processed" / "track1_manifests", "data/processed/track1_manifests")
            ],
            ["smoke", "wav_tmp"] # Exclude smoketests, temporary wav directories
        ),
        (
            "meld_raw.zip",
            [
                (DATA_DIR / "raw" / "MELD", "data/raw/MELD"),
                (DATA_DIR / "processed" / "meld_manifests", "data/processed/meld_manifests")
            ],
            ["smoke"]
        ),
        (
            "mustard.zip",
            [
                (DATA_DIR / "raw" / "MUStARD", "data/raw/MUStARD")
            ],
            ["smoke"]
        ),
        (
            "fakeavceleb.zip",
            [
                (DATA_DIR / "raw" / "FakeAVCeleb_v1.2", "data/raw/FakeAVCeleb_v1.2")
            ],
            ["smoke"]
        )
    ]

    for zip_name, paths, exclusions in specs:
        zip_path = OUTPUT_DIR / zip_name
        print(f"Packing {zip_name}...")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for src_path, arc_root in paths:
                if not src_path.exists():
                    print(f"  Warning: Path does not exist: {src_path}")
                    continue
                
                if src_path.is_file():
                    zf.write(src_path, arc_root)
                    print(f"  Added file: {src_path.name} -> {arc_root}")
                else:
                    count = 0
                    for root, dirs, files in os.walk(src_path):
                        # Filter out excluded directory names
                        dirs[:] = [d for d in dirs if not any(exc in d.lower() for exc in exclusions)]
                        
                        for file in files:
                            if any(exc in file.lower() for exc in exclusions):
                                continue
                            if any(exc in root.lower() for exc in exclusions):
                                continue
                            
                            file_path = Path(root) / file
                            # Re-derive archive name relative to the source directory root
                            rel_path = file_path.relative_to(src_path)
                            arcname = Path(arc_root) / rel_path
                            zf.write(file_path, arcname.as_posix())
                            count += 1
                    print(f"  Added directory: {src_path.name} ({count} files) -> {arc_root}")
        
        size_gb = zip_path.stat().st_size / (1024 ** 3)
        print(f"Finished {zip_name} ({size_gb:.3f} GB)\n")

    print("All zip files successfully created.")

if __name__ == "__main__":
    main()
