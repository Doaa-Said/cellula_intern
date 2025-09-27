from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_caption(image_path: str) -> str:
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_length=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
