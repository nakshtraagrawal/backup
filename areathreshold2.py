import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

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

def get_grounding_output(model, image, caption, box_threshold, text_threshold, max_area_threshold, cpu_only=False):
    """
    Get grounding output with synchronized area-based filtering.
    
    Args:
        max_area_threshold (float): Maximum allowed area as fraction of image area (0.0 to 1.0)
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # Filter based on box threshold first
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    # Get all phrases first
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    # Store all phrases and their corresponding boxes/logits
    all_phrases = []
    final_boxes = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if pred_phrase:
            # Calculate area for this box
            width, height = box[2], box[3]
            area = width * height
            
            # Only add if area is within threshold
            if area <= max_area_threshold:
                area_percentage = (area * 100).item()
                all_phrases.append(f"{pred_phrase} ({logit.max().item():.2f}, area: {area_percentage:.1f}%)")
                final_boxes.append(box)

    # Convert final boxes to tensor
    if final_boxes:
        final_boxes = torch.stack(final_boxes)
    else:
        final_boxes = torch.zeros((0, 4))  # Empty tensor with correct shape

    return final_boxes, all_phrases

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have the same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        # Convert from [x_center, y_center, width, height] to [x0, y0, x1, y1]
        x_center, y_center = box[0] * W, box[1] * H
        width, height = box[2] * W, box[3] * H
        
        x0 = x_center - width/2
        y0 = y_center - height/2
        x1 = x_center + width/2
        y1 = y_center + height/2

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            text_width, text_height = draw.textsize(str(label), font=font)
            bbox = (x0, y0, x0 + text_width, y0 + text_height)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def process_images(image_folder, model, caption, output_dir, box_threshold, text_threshold, max_area_threshold, cpu_only):
    """
    Process images with synchronized area threshold filtering.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing images with area threshold: {max_area_threshold}")

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping unsupported file: {filename}")
            continue

        try:
            image_path = os.path.join(image_folder, filename)
            image_pil, image = load_image(image_path)
            
            # Get predictions with synchronized filtering
            boxes_filt, pred_phrases = get_grounding_output(
                model=model,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_area_threshold=max_area_threshold,
                cpu_only=cpu_only
            )

            # Skip if no valid detections
            if len(boxes_filt) == 0:
                print(f"No valid detections in {filename} after area filtering")
                continue

            # Create prediction dictionary
            pred_dict = {
                "boxes": boxes_filt,
                "size": [image_pil.size[1], image_pil.size[0]],  # [H, W]
                "labels": pred_phrases,
            }

            # Draw boxes and save
            image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            output_path = os.path.join(output_dir, f"pred_{filename}")
            image_with_box.save(output_path)
            print(f"Processed and saved: {output_path}")
            # print(f"Detected {len(pred_phrases)} objects in {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    # Parameters
    CONFIG_FILE = "/home/nakshtra/Desktop/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    CHECKPOINT_PATH = "/home/nakshtra/Desktop/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    IMAGE_FOLDER = "/home/nakshtra/Desktop/GroundingDINO/input"
    OUTPUT_DIR = "/home/nakshtra/Desktop/GroundingDINO/outputnew"
    TEXT_PROMPT = "miniature car. white plane. white inverted umbrella. brown suitcase. black mattress. red mattress. blue badminton racket. orange cricket bat. blue umbrella"

    # Thresholds
    BOX_THRESHOLD = 0.40
    TEXT_THRESHOLD = 0.40
    MAX_AREA_THRESHOLD = 0.20
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
        cpu_only=CPU_ONLY
    )