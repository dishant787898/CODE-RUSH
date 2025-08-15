import os

# Create necessary directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/images/nft', exist_ok=True)

print("Created necessary directories")

# Create placeholder NFT images
try:
    from PIL import Image, ImageDraw, ImageFont
    
    def create_placeholder_image(filename, text, color=(200, 200, 200)):
        img = Image.new('RGB', (400, 300), color=color)
        draw = ImageDraw.Draw(img)
        
        # Try to use a font if available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            draw.text((50, 150), text, fill=(255, 255, 255), font=font)
        except:
            draw.text((50, 150), text, fill=(255, 255, 255))
            
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)
        print(f"Created placeholder image: {filename}")
    
    # Create NFT placeholder images
    create_placeholder_image('static/images/nft/plastic_bottle_1.jpg', "Plastic Bottle Art", (100, 200, 230))
    create_placeholder_image('static/images/nft/paper_art_1.jpg', "Paper Art", (230, 220, 180))
    create_placeholder_image('static/images/nft/metal_1.jpg', "Metal Sculpture", (180, 180, 180))
    create_placeholder_image('static/images/nft/glass_1.jpg', "Glass Mosaic", (200, 230, 240))
    
    # Create sample uploads for analytics page
    create_placeholder_image('static/uploads/plastic_bottle.jpg', "Plastic Bottle", (100, 200, 230))
    create_placeholder_image('static/uploads/banana_peel.jpg', "Banana Peel", (220, 220, 100))
    create_placeholder_image('static/uploads/newspaper.jpg', "Newspaper", (230, 230, 230))
    create_placeholder_image('static/uploads/apple_core.jpg', "Apple Core", (180, 220, 120))

except ImportError:
    print("PIL not available. Skipping image creation.")

print("Setup complete!")
