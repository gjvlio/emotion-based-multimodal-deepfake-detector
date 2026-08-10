import zipfile
import sys
from pathlib import Path

# Try to find the drive mount
drive_path = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/cmumosei.zip")
fallback_path = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/cmumosei.zip")

zip_path = None
for p in [drive_path, fallback_path]:
    if p.exists():
        zip_path = p
        break

if not zip_path:
    print("ERROR: cmumosei.zip not found on Google Drive!")
    print("Please make sure Google Drive is mounted and the file exists at:")
    print("  /content/drive/MyDrive/THESIS_MOTHERFILE/datasets/cmumosei.zip")
    sys.exit(1)

print("=" * 60)
print(f"HEALTH CHECKING DRIVE ARCHIVE: {zip_path.name}")
print(f"Path: {zip_path}")
print(f"Size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
print("=" * 60)

print("Opening zip and running CRC-32 integrity validation...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        total_files = len(names)
        print(f"Total entries in archive: {total_files}")
        
        # Run testzip (checks CRC and headers)
        print("Testing archive headers...")
        bad_file = zf.testzip()
        
        if bad_file is None:
            print("\nSUCCESS: All files in the Drive ZIP archive are 100% HEALTHY!")
            print("The zip file is not corrupted.")
        else:
            print("\n[WARNING] Found corrupted files in Google Drive ZIP!")
            corrupt_count = 0
            for name in names:
                try:
                    # Attempt to read data to verify CRC checksum
                    zf.read(name)
                except Exception as e:
                    corrupt_count += 1
                    if corrupt_count <= 20:
                        print(f"  Corrupt: {name} ({e})")
                    elif corrupt_count == 21:
                        print("  ... (more errors truncated)")
            
            percent_corrupt = (corrupt_count / total_files) * 100
            print("-" * 60)
            print(f"Result: {corrupt_count} / {total_files} files are corrupted ({percent_corrupt:.4f}%)")
            print("Please re-upload a healthy copy of the ZIP.")
            print("-" * 60)
            
except Exception as e:
    print(f"\n[FATAL ERROR] Failed to read ZIP structure: {e}")
    print("This means the ZIP file is completely broken or still uploading/syncing.")
