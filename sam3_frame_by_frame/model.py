from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path="/content/weights/sam3.pt",  # SAM 3 image weights; see colab_setup.py
    load_from_HF=False,
)
# Only keep detections scoring above this
processor = Sam3Processor(model, confidence_threshold=0.5)
