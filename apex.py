import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import csv

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x_center, y_center, width, height] format
    """
    # Convert from center format to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def apply_nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression
    Args:
        boxes: tensor of boxes in [x_center, y_center, width, height] format
        scores: tensor of confidence scores
        iou_threshold: IoU threshold for considering boxes as overlapping
    Returns:
        indices of kept boxes
    """
    if len(boxes) == 0:
        return []
    
    # Convert boxes to numpy for easier handling
    boxes = boxes.numpy()
    scores = scores.numpy()
    
    # Sort boxes by score
    indices = np.argsort(scores)[::-1]
    kept_indices = []
    
    while len(indices) > 0:
        # Keep the box with highest score
        current_idx = indices[0]
        kept_indices.append(current_idx)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU of the kept box with all remaining boxes
        ious = [calculate_iou(boxes[current_idx], boxes[idx]) for idx in indices[1:]]
        
        # Filter out boxes with IoU > threshold
        filtered_indices = [idx for i, idx in enumerate(indices[1:]) 
                          if ious[i] <= iou_threshold]
        indices = np.array(filtered_indices)
    
    return kept_indices

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have the same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # Create a larger font - specify size 24
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # Fallback to default font if custom font is not available
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = map(int, box)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        
        # Get text size for the background rectangle
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            # Approximate the bbox size for older Pillow versions
            text_width, text_height = draw.textsize(str(label), font=font)
            bbox = (x0, y0, x0 + text_width, y0 + text_height)

        # Draw background rectangle for text
        draw.rectangle(bbox, fill=color)
        # Draw text with the larger font
        draw.text((x0, y0), str(label), fill="white", font=font)
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, max_area_threshold, iou_threshold=0.1, cpu_only=False):
    """
    Get grounding output with NMS and area filtering
    Returns additional confidence scores for CSV logging
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0].cpu()
    boxes = outputs["pred_boxes"][0].cpu()

    # Filter based on box threshold
    scores = logits.max(dim=1)[0]
    boxes_filt = boxes.clone()
    logits_filt = logits.clone()
    filt_mask = scores > box_threshold
    
    scores = scores[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    logits_filt = logits_filt[filt_mask]

    # Apply area filtering
    areas = boxes_filt[:, 2] * boxes_filt[:, 3]
    area_mask = areas <= max_area_threshold
    scores = scores[area_mask]
    boxes_filt = boxes_filt[area_mask]
    logits_filt = logits_filt[area_mask]

    # Get phrases before NMS
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    # Create phrases list with confidence scores
    phrases_with_scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if pred_phrase:
            area_percentage = (box[2] * box[3] * 100).item()
            conf_score = logit.max().item()
            phrases_with_scores.append({
                'phrase': pred_phrase,
                'score': conf_score,
                'area': area_percentage,
                'box': box
            })

    # Sort by confidence score
    phrases_with_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply NMS
    kept_indices = []
    kept_phrases = []
    kept_boxes = []
    kept_scores = []  # Add this to store confidence scores
    
    for i, current in enumerate(phrases_with_scores):
        should_keep = True
        current_box = current['box']
        
        for kept_idx in kept_indices:
            kept_box = phrases_with_scores[kept_idx]['box']
            iou = calculate_iou(current_box, kept_box)
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            kept_indices.append(i)
            kept_phrases.append(
                f"{current['phrase']} ({current['score']:.2f}, area: {current['area']:.1f}%)"
            )
            kept_boxes.append(current_box)
            kept_scores.append(current['score'])  # Store the confidence score
    
    # Convert kept boxes to tensor
    kept_boxes = torch.stack(kept_boxes) if kept_boxes else torch.zeros((0, 4))
    
    return kept_boxes, kept_phrases, kept_scores  # Return scores as well

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

def process_images(image_folder, model, caption, output_dir, box_threshold, text_threshold, max_area_threshold, iou_threshold=0.5, cpu_only=False):
    """
    Process images and save detection results to CSV
    """
    # Create main output directory and crops directory
    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    # Create CSV file for detection results
    csv_path = os.path.join(output_dir, "detection_results.csv")
    csv_header = ["image_name", "object_class", "center_x", "center_y", "confidence_score"]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        
        print(f"Processing images with area threshold: {max_area_threshold}, IoU threshold: {iou_threshold}")
        
        processed_count = 0
        error_count = 0
        total_crops = 0
        
        for filename in os.listdir(image_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                image_path = os.path.join(image_folder, filename)
                image_pil, image = load_image(image_path)
                
                boxes_filt, pred_phrases, confidence_scores = get_grounding_output(
                    model=model,
                    image=image,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    max_area_threshold=max_area_threshold,
                    iou_threshold=iou_threshold,
                    cpu_only=cpu_only
                )

                if len(boxes_filt) != len(pred_phrases):
                    print(f"Warning: Mismatch in boxes ({len(boxes_filt)}) and labels ({len(pred_phrases)}) for {filename}")
                    continue

                if len(boxes_filt) == 0:
                    print(f"No valid detections in {filename}")
                    continue

                # Write detection results to CSV
                W, H = image_pil.size
                for box, label, score in zip(boxes_filt, pred_phrases, confidence_scores):
                    # Get center coordinates (normalized coordinates * image dimensions)
                    center_x = int(box[0].item() * W)
                    center_y = int(box[1].item() * H)
                    
                    # Clean label (remove confidence score and area percentage)
                    clean_label = label.split('(')[0].strip()
                    
                    # Write to CSV
                    writer.writerow([filename, clean_label, center_x, center_y, f"{score:.3f}"])

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
        print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    # Parameters
    CONFIG_FILE = "/home/nakshtra/Desktop/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    CHECKPOINT_PATH = "/home/nakshtra/Desktop/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    IMAGE_FOLDER = "/home/nakshtra/Desktop/GroundingDINO/images"
    OUTPUT_DIR = "/home/nakshtra/Desktop/GroundingDINO/test"
    #assignment data set TEXT_PROMPT = "small brown miniature car. white plane. white inverted umbrella. brown suitcase luggage bag. black mattress. red mattress. blue badminton racket. orange cricket bat. blue umbrella.basketball.black tennis racket cover"
    # apex snowboard data TEXT_PROMPT = "black mattress with a small white label. white snowboard.white aeroplane with a tail.two white snow skis lying parallel to each other.lugguage bag.person with black hair walking"
    TEXT_PROMPT = "person.brown lugguage bag. blue drone. basket ball.red thick bed mattress"

    # Thresholds
    BOX_THRESHOLD = 0.34
    TEXT_THRESHOLD = 0.30
    MAX_AREA_THRESHOLD = 0.015
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
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        max_area_threshold=MAX_AREA_THRESHOLD,
        iou_threshold=IOU_THRESHOLD, 
        cpu_only=CPU_ONLY
    )