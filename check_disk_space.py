import os
import shutil
import platform

def check_disk_space():
    """Check if there's enough disk space for models"""
    # Calculate total required space (add 10% buffer)
    required_mb = sum(model["size_mb"] for model in MODELS.values())
    required_mb = int(required_mb * 1.1)  # Add 10% buffer
    
    # Get available space
    if platform.system() == 'Windows':
        # For Windows
        free_bytes = shutil.disk_usage(".").free
    else:
        # For Unix/Linux/Mac
        stat = os.statvfs(".")
        free_bytes = stat.f_bavail * stat.f_frsize
    
    free_mb = free_bytes / (1024 * 1024)
    
    print(f"Required disk space: {required_mb} MB")
    print(f"Available disk space: {free_mb:.0f} MB")
    
    if free_mb < required_mb:
        print(f"\n⚠️ Warning: Not enough disk space. Need {required_mb} MB, but only {free_mb:.0f} MB available.")
        return False
    
    return True

# Add this function to the imports at the top of setup_models.py
# and call it before downloading models
