import os
from huggingface_hub import hf_hub_download

def download_sinpa_data():
    """Download SINPA dataset from Hugging Face"""
    repo_id = "Huaiwu/SINPA"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Files to download
    files_to_download = [
        "prob_full_occupy.npy",
        "region/assignment.npy",
        "region/mask.npy", 
        "base/dist.npy",
        "sensor_graph/adj_mx_base.pkl"
    ]
    
    print("Downloading SINPA dataset files...")
    
    for file_path in files_to_download:
        try:
            print(f"Downloading {file_path}...")
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(f"data/{file_path}"), exist_ok=True)
            
            # Download file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir="data",
                local_dir_use_symlinks=False
            )
            
            print(f"Successfully downloaded {file_path}")
            
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
    
    print("Download completed!")

if __name__ == "__main__":
    download_sinpa_data() 