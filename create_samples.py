import os
from PIL import Image, ImageDraw, ImageFont

def create_placeholder(path, text, color=(200, 200, 200)):
    """Create placeholder image with text"""
    img = Image.new('RGB', (400, 300), color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, or default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text((20, 140), text, fill="white", font=font)
    except:
        draw.text((20, 140), text, fill="white")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Created: {path}")

# Create NFT samples
create_placeholder("static/images/nft/plastic_bottle_1.jpg", "Plastic Bottle Art", (65, 105, 225))
create_placeholder("static/images/nft/paper_art_1.jpg", "Recycled Paper Art", (200, 180, 100))
create_placeholder("static/images/nft/metal_1.jpg", "Metal Sculpture", (180, 180, 180))
create_placeholder("static/images/nft/glass_1.jpg", "Glass Mosaic", (70, 200, 170))

# Create sample uploads for analytics page
create_placeholder("static/uploads/plastic_bottle.jpg", "Plastic Bottle", (65, 105, 225))
create_placeholder("static/uploads/banana_peel.jpg", "Banana Peel", (220, 180, 50))
create_placeholder("static/uploads/newspaper.jpg", "Newspaper", (220, 220, 220))
create_placeholder("static/uploads/apple_core.jpg", "Apple Core", (180, 30, 30))

print("Sample images created successfully!")
