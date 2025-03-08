import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.ops as ops



def plot_boxes_to_image(image_pil, tgt, conf_threshold=0.5, iou_threshold=0.5):
    """
    Plot boxes on image with confidence filtering and NMS
    Args:
        image_pil: PIL image
        tgt: dict containing boxes, labels, and scores
        conf_threshold: minimum confidence score to show box
        iou_threshold: IOU threshold for NMS
    """
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    scores = tgt.get("scores", torch.ones(len(boxes)))  # Default all scores to 1 if not provided
    
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = [label for i, label in enumerate(labels) if mask[i]]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return image_pil, None
    
    # Convert boxes to expected format for NMS
    boxes_for_nms = boxes.clone()
    boxes_for_nms = boxes_for_nms * torch.tensor([W, H, W, H])
    boxes_for_nms[:, :2] -= boxes_for_nms[:, 2:] / 2
    boxes_for_nms[:, 2:] += boxes_for_nms[:, :2]
    
    # Apply NMS
    keep_indices = ops.nms(boxes_for_nms, scores, iou_threshold)
    boxes = boxes[keep_indices]
    labels = [labels[i] for i in keep_indices]
    scores = scores[keep_indices]

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # Use a fixed color palette for better visualization
    color_palette = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        box = box * torch.tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        
        # Use color from palette (cycle through colors if more boxes than colors)
        color = color_palette[idx % len(color_palette)]
        
        x0, y0, x1, y1 = map(int, box)
        
        # Draw box with thinner lines
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Draw label with score
        font = ImageFont.load_default()
        label_text = f"{label} ({score:.2f})"
        bbox = draw.textbbox((x0, y0-15), label_text, font) if hasattr(font, "getbbox") else (x0, y0-15, x0 + 100, y0)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0-15), label_text, fill="white")

    return image_pil, mask

def get_grounding_output(model, image, caption, device, conf_threshold=0.5):
    """Modified to include confidence scores in output"""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    
    # Get confidence scores
    scores = logits.max(dim=1)[0]
    
    # Filter by confidence threshold
    mask = scores >= conf_threshold
    logits = logits[mask]
    boxes = boxes[mask]
    scores = scores[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    for logit in logits:
        pred_phrase = get_phrases_from_posmap(logit > conf_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase)

    return boxes, pred_phrases, scores

def process_images(image_folder, model, caption, output_dir, device, conf_threshold=0.5, iou_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            image_path = os.path.join(image_folder, filename)
            image_pil, image = load_image(image_path, device)
            
            # Get predictions with confidence scores
            boxes, pred_phrases, scores = get_grounding_output(
                model, image, caption, device, conf_threshold
            )

            # Move tensors to CPU for plotting
            boxes = boxes.cpu()
            scores = scores.cpu()

            pred_dict = {
                "boxes": boxes,
                "size": [image_pil.size[1], image_pil.size[0]],
                "labels": pred_phrases,
                "scores": scores
            }
            
            # Plot with confidence filtering and NMS
            image_with_box, _ = plot_boxes_to_image(
                image_pil, 
                pred_dict,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            output_path = os.path.join(output_dir, f"pred_{filename}")
            image_with_box.save(output_path)
            print(f"Processed and saved: {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Update main script with new parameters
if __name__ == "__main__":
    # ... (previous parameter definitions) ...
    
    # Add new parameters for filtering
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score to keep a prediction
    IOU_THRESHOLD = 0.5        # IOU threshold for NMS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG_FILE, CHECKPOINT_PATH, device)
    
    process_images(
        IMAGE_FOLDER, 
        model, 
        TEXT_PROMPT, 
        OUTPUT_DIR, 
        device,
        conf_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )