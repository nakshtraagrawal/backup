import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
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

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = map(int, box)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        font = ImageFont.load_default()
        bbox = draw.textbbox((x0, y0), str(label), font) if hasattr(font, "getbbox") else (x0, y0, x0 + 50, y0 + 15)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_image(image_path, device):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    image = image.to(device)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)  # Changed to load directly to correct device
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model = model.to(device)
    model.eval()
    return model

def get_grounding_output(model, image, caption, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # Ensure everything is on the same device
    logits = logits.to(device)
    boxes = boxes.to(device)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    for logit, box in zip(logits, boxes):
        pred_phrase = get_phrases_from_posmap(logit > 0.5, tokenized, tokenizer)
        pred_phrases.append(pred_phrase)

    return boxes, pred_phrases

def process_images(image_folder, model, caption, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png")):
            print(f"Skipping unsupported file: {filename}")
            continue

        try:
            image_pil, image = load_image(image_path, device)
            boxes, pred_phrases = get_grounding_output(
                model, image, caption, device
            )

            # Move boxes to CPU for plotting
            boxes = boxes.cpu()

            size = image_pil.size
            pred_dict = {
                "boxes": boxes,
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
    OUTPUT_DIR = "/home/nakshtra/Desktop/GroundingDINO/output2"
    TEXT_PROMPT = "miniature car.plane.umbrella.red suitcase.blue mattress.red mattress.blue racket"
 
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(CONFIG_FILE, CHECKPOINT_PATH, device)

    # Process images
    process_images(IMAGE_FOLDER, model, TEXT_PROMPT, OUTPUT_DIR, device)