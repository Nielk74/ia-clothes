import streamlit as st
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


def get_occurences_by_skin_cluster(skin_cluster, occurences):
    occurences_by_skin_cluster = {}
    for key, value in occurences.items():
        if value['skin_cluster'] == skin_cluster:
            occurences_by_skin_cluster[key] = value
    occurences_by_skin_cluster = {k: v for k, v in sorted(occurences_by_skin_cluster.items(), key=lambda item: item[1]['occurences'], reverse=True)}
    return occurences_by_skin_cluster


def write_color(color):
  return f'<p style="background-color:rgb{color}; width:20px; height: 20px"></p>'


def get_result(file_name, result_container):
  if (file_name is None):
    st.error("Please upload an image")
    return

  # double column layout
  col1, col2 = result_container.columns(2)

  # display the image
  image = Image.open(file_name)
  col1.image(image, use_column_width=True)

  # display the detected skin color
  skin_color, pixel_count = get_skin_color(image)
  col2.header("Detected skin color")
  col2.markdown(write_color(skin_color), unsafe_allow_html=True)

  col2.header("Clothing colors occurences")
  col2.write("TODO")


def render_page():
  st.title("ENSIMAG AI project")

  formCol1, formCol2 = st.columns(2)
  formCol1.radio("Gender", ["Male", "Female"])
  formCol2.radio("Number of clothing color clusters", ["20", "40"])

  file_name = st.file_uploader("Upload a full body image")

  with st.container():
    if st.button("Submit"):
      get_result(file_name, st)


if __name__ == "__main__":
  render_page()