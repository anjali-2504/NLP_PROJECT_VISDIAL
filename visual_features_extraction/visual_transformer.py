from transformers import ViTForImageClassification
import torch
from PIL import Image
import requests
from transformers import ViTFeatureExtractor


if __name__ == "__main__":

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
   model.eval()
   model.to(device)
  
   
   feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
   encoding = feature_extractor(images=im, return_tensors="pt")
   encoding.keys()
   pixel_values = encoding['pixel_values'].to(device)
   outputs = model(pixel_values)
   logits = outputs.logits
   
