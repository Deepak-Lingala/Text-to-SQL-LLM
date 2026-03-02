"""
Download Spider SQLite databases for execution accuracy evaluation.
Run this script before evaluation notebooks (03 and 05).
"""
import os
import urllib.request
import zipfile

def download_spider_databases(target_dir="spider_databases"):
    """Download and extract Spider SQLite databases."""
    if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 10:
        print(f"✅ Spider databases already exist at {target_dir}/")
        return target_dir
    
    print("Downloading Spider databases...")
    zip_path = "/tmp/spider.zip"
    
    # Download from the Spider dataset mirror
    url = "https://drive.usercontent.google.com/download?id=1iRDVHLr6a7w74J9F2gVLjGv9rkzG90LP&confirm=t"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception:
        # Fallback: try gdown
        print("Direct download failed. Trying gdown...")
        os.system(f"pip install -q gdown")
        os.system(f"gdown 1iRDVHLr6a7w74J9F2gVLjGv9rkzG90LP -O {zip_path}")
    
    if not os.path.exists(zip_path):
        print("❌ Download failed. Please download manually from:")
        print("   https://yale-lily.github.io/spider")
        print(f"   Extract the 'database/' folder as '{target_dir}/'")
        return None
    
    print("Extracting...")
    os.makedirs(target_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Spider zip contains spider/database/ with all the SQLite files
        for member in z.namelist():
            if '/database/' in member and member.endswith('.sqlite'):
                # Extract to target_dir/db_id/db_id.sqlite
                parts = member.split('/')
                db_idx = parts.index('database')
                if db_idx + 1 < len(parts):
                    relative = '/'.join(parts[db_idx + 1:])
                    out_path = os.path.join(target_dir, relative)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with z.open(member) as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())
    
    os.remove(zip_path)
    
    num_dbs = len([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    print(f"✅ Extracted {num_dbs} databases to {target_dir}/")
    return target_dir


if __name__ == "__main__":
    download_spider_databases()
