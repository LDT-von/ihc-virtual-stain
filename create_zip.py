import os
from pathlib import Path

zip_file = Path("E:/aic/ihc-virtual-stain/results_test_pix2pix.zip")
if zip_file.exists():
    print(f"ZIP file: {zip_file}")
    print(f"Size: {os.path.getsize(zip_file) / 1024 / 1024:.2f} MB")
else:
    print("ZIP file not found!")
    
    # Check if we need to create it manually
    import zipfile
    results_dir = Path("E:/aic/ihc-virtual-stain/results/test/HLA-DR")
    if results_dir.exists():
        print(f"Creating ZIP from {results_dir}...")
        
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(results_dir.glob("*.jpg")):
                zf.write(f, f"test/HLA-DR/{f.name}")
        
        print(f"ZIP created: {os.path.getsize(zip_file) / 1024 / 1024:.2f} MB")
