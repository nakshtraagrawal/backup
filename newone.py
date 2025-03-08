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

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, cpu_only=False):
    assert text_threshold is not None, "text_threshold must not be None!"
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases

def process_images(image_folder, model, caption, output_dir, box_threshold, text_threshold, cpu_only):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png")):
            print(f"Skipping unsupported file: {filename}")
            continue

        try:
            image_pil, image = load_image(image_path)
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, caption, box_threshold, text_threshold, cpu_only=cpu_only
            )

            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],
                "labels": pred_phrases,
            }
            image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            output_path = os.path.join(output_dir, f"pred_{filename}")
            image_with_box.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Parameters
    CONFIG_FILE = "/home/nakshtra/Desktop/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    CHECKPOINT_PATH = "/home/nakshtra/Desktop/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    IMAGE_FOLDER = "/home/nakshtra/Desktop/GroundingDINO/input"
    OUTPUT_DIR = "/home/nakshtra/Desktop/GroundingDINO/output"
    TEXT_PROMPT = "red miniature car. white plane.white inverted umbrella. brown suitcase. mattress. red mattress with yellow pattern. blue badminton racket. orange cricket bat.blue umbrella"

    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.35
    CPU_ONLY = False

    # Load model
    model = load_model(CONFIG_FILE, CHECKPOINT_PATH, cpu_only=CPU_ONLY)

    # Process images
    process_images(IMAGE_FOLDER, model, TEXT_PROMPT, OUTPUT_DIR, BOX_THRESHOLD, TEXT_THRESHOLD, CPU_ONLY)