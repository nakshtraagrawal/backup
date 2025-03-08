import os
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load model and processor
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on device: {device}")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Folder paths
image_folder = "//home/nakshtra/Desktop/GroundingDINO/detection"  # Path to the folder containing input images
output_folder = "home/nakshtra/Desktop/GroundingDINO/outputcheck"  # Path to save annotated images
crops_folder = "/home/nakshtra/Desktop/GroundingDINO/crops1"  # New folder for cropped images

# Create output directories
os.makedirs(output_folder, exist_ok=True)
os.makedirs(crops_folder, exist_ok=True)

# Define the objects to detect
# text = ".a grey bed. a grey mattres.a person .a small suitcase.a small trolly bag.a person sleeping.a black pentagon umbrella.a orange ball."
text = "white plane.brown miniature car.white umbrella.orange cricket bat"

# Thresholds for detection
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

print("Thresholds:")
print(f"BOX_THRESHOLD: {BOX_THRESHOLD}")
print(f"TEXT_THRESHOLD: {TEXT_THRESHOLD}")

def save_crop(image, box, label, score, image_name, crop_number):
    """Save a cropped region of the image"""
    # Convert box coordinates to integers
    box = [int(coord) for coord in box]
    
    # Ensure coordinates are within image boundaries
    width, height = image.size
    box = [
        max(0, box[0]),  # x1
        max(0, box[1]),  # y1
        min(width, box[2]),  # x2
        min(height, box[3])  # y2
    ]
    
    # Crop the image
    cropped_image = image.crop(box)
    
    # Create filename with detection information
    clean_label = "".join(c if c.isalnum() else "_" for c in label)  # Clean label for filename
    filename = f"{os.path.splitext(image_name)[0]}_{clean_label}_{score:.2f}_{crop_number}.jpg"
    crop_path = os.path.join(crops_folder, filename)
    
    # Save the crop
    cropped_image.save(crop_path)
    return crop_path

# Process all images in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing: {image_file}")
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs for the model
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process the results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]]
        )
        
        if len(results) == 0:
            print(f"No detections for {image_file}. Try adjusting thresholds or the description.")
            continue
        
        # Annotate the image and save crops
        draw = ImageDraw.Draw(image)
        for result in results:
            for crop_number, (box, label, score) in enumerate(zip(result["boxes"], result["labels"], result["scores"])):
                if score >= BOX_THRESHOLD:
                    # Draw bounding box and label on the original image
                    box_coords = box.tolist()
                    label_text = f"{label} ({score:.2f})"
                    draw.rectangle(box_coords, outline="red", width=3)
                    draw.text((box_coords[0], box_coords[1] - 10), label_text, fill="red")
                    
                    # Save the cropped region
                    crop_path = save_crop(image, box_coords, label, score, image_file, crop_number)
                    print(f"Saved crop: {crop_path}")
        
        # Save the annotated image
        output_path = os.path.join(output_folder, image_file)
        image.save(output_path)
        print(f"Processed and saved annotated image: {output_path}")

print("Processing complete. Annotated images and crops are saved in their respective folders.")
