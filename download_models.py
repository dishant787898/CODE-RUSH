import os
import sys
import requests
from tqdm import tqdm
import re

# Function to extract file ID from Google Drive link
def extract_file_id(drive_link):
    """Extract Google Drive file ID from a sharing link"""
    # Pattern for Google Drive links
    patterns = [
        r"https://drive\.google\.com/file/d/([\w-]+)",  # /file/d/ format
        r"https://drive\.google\.com/open\?id=([\w-]+)",  # open?id= format
        r"https://docs\.google\.com/[\w/]+/d/([\w-]+)"   # docs format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_link)
        if match:
            return match.group(1)
    
    # If it's already just the ID
    if re.match(r"^[\w-]+$", drive_link):
        return drive_link
    
    return None

# Create models directory
MODEL_FOLDER = os.getenv('MODEL_PATH', 'models/')
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive using the file ID.
    
    Args:
        file_id: The ID of the file on Google Drive
        destination: Local path where the file should be saved
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}, skipping download.")
        return
    
    print(f"Downloading file to {destination}...")
    
    # Google Drive API endpoint
    URL = "https://drive.google.com/uc?export=download"
    
    # Initial request to get the confirmation token
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    
    # Check if file is large and requires confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    
    # Add confirmation token if needed
    params = {'id': file_id}
    if token:
        params['confirm'] = token
    
    # Download the file with progress bar
    response = session.get(URL, params=params, stream=True)
    
    # Get file size if available
    file_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(destination, 'wb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download complete: {destination}")

model_links = {
    'vgg16_waste_classification_tf.h5': 'https://drive.google.com/file/d/14euZ5Ka8TaVM56YUwxiT8RLt94LTLSrd/view?usp=drive_link',
    'waste_classification22_model.h5': 'https://drive.google.com/file/d/1babBBQeQTT-CADo-WFEDBeMxVReEJAwb/view?usp=drive_link',
    'inceptionv3_waste_classification_tf.h5': 'https://drive.google.com/file/d/17WY01-emhtiC2WvDecKYw680GlBGe0Gq/view?usp=drive_link'
}

def main():
    print("Starting model downloads from Google Drive...")
    
    for model_name, link in model_links.items():
        file_id = extract_file_id(link)
        
        if not file_id:
            print(f"⚠️ Could not extract file ID for {model_name}. Please check your Google Drive link.")
            continue
            
        destination = os.path.join(MODEL_FOLDER, model_name)
        try:
            download_file_from_google_drive(file_id, destination)
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            print("Will try to continue with other models...")
    
    print("Model download process completed.")

if __name__ == "__main__":
    main()
