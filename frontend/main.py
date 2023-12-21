import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import torch
import extcolors


processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)


def getColorDominantFromMask(image, mask):
  # apply mask
  binary_mask = (mask * 255).astype(np.uint8)

  # convert PIL image to openCV image
  imageCV = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA)

  # add the forth dimension (opacity)
  four_channel_mask = cv2.merge([binary_mask] * 4)

  # Apply mask
  result = cv2.bitwise_and(imageCV, four_channel_mask)

  # Convert to PIL image
  pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGBA))

  # get dominant color
  colors = extcolors.extract_from_image(pil_image)
  return colors[0][0]

def get_skin_color(image):
  inputs = processor(images=image, return_tensors="pt").to(device)
  outputs = model(**inputs)
  logits = outputs.logits.cpu()

  upsampled_logits = nn.functional.interpolate(
      logits,
      size=image.size[::-1],
      mode="bilinear",
      align_corners=False,
  )

  # Get the segmentation prediction
  pred_seg = upsampled_logits.argmax(dim=1)[0]
  rows, cols = pred_seg.shape

  skin_mask = np.full((rows,cols), False, dtype=bool)
  segmentation = pred_seg.detach().cpu().numpy()
  has_skin = False

  for i in range(rows):
      for j in range(cols):
        v = segmentation[i, j]
        if v in [11,12,13,14,15]: #body parts
          skin_mask[i,j]=  True
          has_skin = True

  return getColorDominantFromMask(image,skin_mask)

#pipeline = pipeline(task="skin-color-detection", model="mattmdjaga/segformer_b2_clothes")

st.title("My title")

file_name = st.file_uploader("Upload a full body image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = get_skin_color(image)

    col2.header("Probabilities")
    st.write(predictions)
    # for p in predictions:
    #     col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")