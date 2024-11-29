from PIL import Image, ImageDraw, ImageFont

# Create a blank image with light blue background
width, height = 800, 400
light_blue = (135, 206, 250)  # A lighter shade of blue
image = Image.new('RGB', (width, height), light_blue)
draw = ImageDraw.Draw(image)

# Load a font
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)

# Draw the title
draw.text((10, 10), "Architecture Overview", fill=black, font=font)

# Draw the Backend box
draw.rectangle([50, 50, 350, 300], outline=black, fill=white)
draw.text((60, 60), "Backend", fill=black, font=font)

# Draw the Frontend box
draw.rectangle([450, 50, 750, 300], outline=black, fill=white)
draw.text((460, 60), "Frontend", fill=black, font=font)

# Draw the components inside Backend
draw.rectangle([60, 100, 120, 150], outline=black)
draw.text((70, 110), "Database", fill=black, font=font)

draw.rectangle([60, 160, 120, 210], outline=black)
draw.text((70, 170), "Vectorization\n(Sentence Transformer)", fill=black, font=font)

draw.rectangle([60, 220, 120, 270], outline=black)
draw.text((70, 230), "File Storage", fill=black, font=font)

# Draw the Search Engine box inside Backend
draw.rectangle([150, 100, 340, 270], outline=black)
draw.text((160, 110), "Search Engine", fill=black, font=font)
draw.text((160, 130), "Endpoints", fill=black, font=font)
draw.text((160, 200), "Request Handler", fill=black, font=font)

# Draw the components inside Frontend
draw.rectangle([460, 100, 740, 270], outline=black)
draw.text((470, 110), "User Interface \n  (single search bar ", fill=black, font=font)

# Draw arrows for data flow
draw.line([400, 170, 450, 170], fill=black, width=2)
draw.polygon([(450, 170), (440, 165), (440, 175)], fill=black)  # Arrowhead

draw.line([110, 150, 110, 160], fill=black, width=2)
draw.line([110, 185, 110, 195], fill=black, width=2)
draw.line([110, 220, 110, 230], fill=black, width=2)

draw.line([120, 175, 150, 175], fill=black, width=2)
draw.line([150, 175, 150, 225], fill=black, width=2)
draw.line([150, 225, 120, 225], fill=black, width=2)
draw.line([150, 150, 150, 110], fill=black, width=2)

# Save the image to the specified path
image_path = "C:\\Users\\Acer\\OneDrive\\Desktop\\HMI PROJECT\\architecture_overview_final_light_blue.png"
image.save(image_path)

print(f"Image saved at {image_path}")
