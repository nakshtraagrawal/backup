import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# [Previous functions remain unchanged up to process_images]

def crop_and_save_object(image_pil, box, label, output_dir, filename):
    """
    Crop detected object from image and save it
    Args:
        image_pil: PIL Image
        box: tensor [x_center, y_center, width, height]
        label: string
        output_dir: directory to save cropped image
        filename: original image filename
    """
    # Convert from center format to corner format
    W, H = image_pil.size
    box = box * torch.Tensor([W, H, W, H])
    x_center, y_center, width, height = box.tolist()
    
    x0 = int(x_center - width/2)
    y0 = int(y_center - height/2)
    x1 = int(x_center + width/2)
    y1 = int(y_center + height/2)
    
    # Ensure coordinates are within image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    
    # Crop the image
    cropped_img = image_pil.crop((x0, y0, x1, y1))
    
    # Clean the label for filename (remove confidence score and area percentage)
    clean_label = label.split('(')[0].strip().replace(' ', '_')
    
    # Create filename for cropped image
    base_filename = os.path.splitext(filename)[0]
    crop_filename = f"{base_filename}_{clean_label}.jpg"
    
    # Save cropped image
    crop_path = os.path.join(output_dir, crop_filename)
    cropped_img.save(crop_path, "JPEG")
    return crop_path

def process_images(image_folder, model, caption, output_dir, crops_dir, box_threshold, text_threshold, max_area_threshold, iou_threshold=0.5, cpu_only=False):
    """
    Process images with improved error handling, logging, and object cropping
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for cropped objects
    os.makedirs(crops_dir, exist_ok=True)
    
    print(f"Processing images with area threshold: {max_area_threshold}, IoU threshold: {iou_threshold}")
    print(f"Saving crops to: {crops_dir}")
    
    processed_count = 0
    error_count = 0
    total_crops = 0
    
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            image_path = os.path.join(image_folder, filename)
            image_pil, image = load_image(image_path)
            
            boxes_filt, pred_phrases = get_grounding_output(
                model=model,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_area_threshold=max_area_threshold,
                iou_threshold=iou_threshold,
                cpu_only=cpu_only
            )

            # Verify boxes and labels match
            if len(boxes_filt) != len(pred_phrases):
                print(f"Warning: Mismatch in boxes ({len(boxes_filt)}) and labels ({len(pred_phrases)}) for {filename}")
                continue

            if len(boxes_filt) == 0:
                print(f"No valid detections in {filename}")
                continue

            # Crop and save individual objects
            for box, label in zip(boxes_filt, pred_phrases):
                crop_path = crop_and_save_object(image_pil, box, label, crops_dir, filename)
                total_crops += 1

            # Save annotated image
            pred_dict = {
                "boxes": boxes_filt,
                "size": [image_pil.size[1], image_pil.size[0]],
                "labels": pred_phrases,
            }

            image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            output_path = os.path.join(output_dir, f"pred_{filename}")
            image_with_box.save(output_path)
            print(f"Processed {filename}: found {len(pred_phrases)} objects")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_count += 1
            continue
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_count} images")
    print(f"Total objects cropped: {total_crops}")
    print(f"Errors encountered: {error_count} images")

if __name__ == "__main__":
    # Parameters
    CONFIG_FILE = "/home/nakshtra/Desktop/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    CHECKPOINT_PATH = "/home/nakshtra/Desktop/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    IMAGE_FOLDER = "/home/nakshtra/Desktop/GroundingDINO/largeinput"
    OUTPUT_DIR = "/home/nakshtra/Desktop/GroundingDINO/largeoutput"
    CROPS_DIR = "/home/nakshtra/Desktop/GroundingDINO/largeoutput/crops"  # Add this line
    TEXT_PROMPT = "black mattress with a small white label. white snowboard.white aeroplane with a tail.two white snow skis lying parallel to each other.lugguage bag.person with black hair walking"
    
    # Thresholds
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.35
    MAX_AREA_THRESHOLD = 0.110
    IOU_THRESHOLD = 0.05
    CPU_ONLY = False

    # Load model
    model = load_model(CONFIG_FILE, CHECKPOINT_PATH, cpu_only=CPU_ONLY)

    # Process images
    process_images(
        image_folder=IMAGE_FOLDER,
        model=model,
        caption=TEXT_PROMPT,
        output_dir=OUTPUT_DIR,
        crops_dir=CROPS_DIR,  # Add this parameter
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        max_area_threshold=MAX_AREA_THRESHOLD,
        iou_threshold=IOU_THRESHOLD, 
        cpu_only=CPU_ONLY
    )