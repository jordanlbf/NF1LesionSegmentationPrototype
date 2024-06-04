# segformer_app/segformer_model.py
import torch
from transformers import SegformerForSemanticSegmentation
import os

def load_model():
    script_dir = os.path.dirname(__file__)
    # Construct the full path to the model file
    model_path = os.path.join(script_dir, 'SegFormer_b4_Final_512.pth')
    model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

