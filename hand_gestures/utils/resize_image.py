from PIL import Image

# -------- Configuration --------
input_path = "test_image2.jpg"         # Path to your input image
output_path = "test_image2.jpg"      # Path to save the resized image
new_size = (256, 256)            # New size: (width, height)

# -------- Load and Resize --------
image = Image.open(input_path)
resized_image = image.resize(new_size)

# -------- Save Result --------
resized_image.save(output_path)

print(f"Image resized to {new_size} and saved as '{output_path}'")
