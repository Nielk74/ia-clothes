import streamlit as st
from PIL import Image
import cv2
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import torch
import extcolors
import colour
from colour.models import RGB_COLOURSPACE_sRGB
import requests

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
color_space = (RGB_COLOURSPACE_sRGB,)


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


def normalize_to_srgb(rgb):
    return np.dot(1/255,rgb)


def rgb_to_lab(rgb):
    rgb = normalize_to_srgb(rgb)
    rgb = colour.RGB_to_XYZ(rgb, *color_space)
    rgb = colour.XYZ_to_Lab(rgb)
    return rgb


def lab_to_rgb(lab):
    lab = np.array(lab)
    lab = np.reshape(lab, (1,3))
    lab = np.apply_along_axis(colour.Lab_to_XYZ, 1, lab)
    lab = np.apply_along_axis(colour.XYZ_to_RGB, 1, lab, *color_space)
    lab = np.apply_along_axis(np.dot, 1, lab, 255)
    return lab[0]


def get_closest_skin_cluster_index(skin_color, clusters_centers_skin):
    cluster_centers_skin = clusters_centers_skin
    lab_input = rgb_to_lab(skin_color)
    # find the closest skin cluster index
    closest_skin_cluster_index = 0
    closest_skin_cluster_distance = 10000
    for skin_cluster_index in range(len(cluster_centers_skin)):
        lab_skin_cluster = cluster_centers_skin[skin_cluster_index]
        distance = np.linalg.norm(lab_input - lab_skin_cluster)
        if distance < closest_skin_cluster_distance:
            closest_skin_cluster_distance = distance
            closest_skin_cluster_index = skin_cluster_index
    return closest_skin_cluster_index


def get_occurences_by_skin_cluster_index(skin_cluster_index, occurences):
    occurences_by_skin_cluster = {}
    for key, value in occurences.items():
        if value['skin_cluster'] == skin_cluster_index:
            occurences_by_skin_cluster[key] = value
    occurences_by_skin_cluster = {k: v for k, v in sorted(occurences_by_skin_cluster.items(), key=lambda item: item[1]['occurences'], reverse=True)}
    return occurences_by_skin_cluster


def write_color(color):
  if (type(color) is np.ndarray):
     color = "(" + ",".join(color.astype(str)) + ")"
  return f'<p style="background-color:rgb{color}; width:40px; height: 40px"></p>'


def get_result(file_name, dataset, gender, cluster_size):
  if file_name is None:
    st.error("Please upload an image of yourself")
    return
  
  if dataset == "Style du Monde":
    clusters = requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/style-du-monde/clusters-sdm-20.json').json()
    occurences = requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/style-du-monde/occurences-sdm-20.json').json()
  else:
    clusters = requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/deepfashion/clusters-{gender.lower()}-{cluster_size}.json').json()
    occurences = requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/deepfashion/occurences-{gender.lower()}-{cluster_size}.json').json()

  # display the image
  image = Image.open(file_name)
  st.image(image)

  # display the detected skin color
  skin_color, pixel_count = get_skin_color(image)
  st.header("Detected skin tone")
  st.markdown(write_color(skin_color), unsafe_allow_html=True)

  # display the clothing colors occurences matching the detected skin color
  st.header("Most popular clothing colors matching your skin tone")
  closest_skin_cluster_index = get_closest_skin_cluster_index(skin_color, clusters["cluster_centers_skin"])
  occurences_by_skin_cluster = get_occurences_by_skin_cluster_index(closest_skin_cluster_index, occurences)
  cluster_centers_upper = clusters["cluster_centers_upper"]
  cluster_centers_lower = clusters["cluster_centers_lower"]
  for i in range(10):
    key = list(occurences_by_skin_cluster.keys())[i]
    value = occurences_by_skin_cluster[key]
    rgb_upper = lab_to_rgb(cluster_centers_upper[value['upper_cluster']])
    rgb_lower = lab_to_rgb(cluster_centers_lower[value['lower_cluster']])
    col1, col2, col3, col4 = st.columns(4)
    col1.write(f"{i+1}. Occurences: {value['occurences']}")
    col2.markdown(f"Upper color: {write_color(rgb_upper)}", unsafe_allow_html=True)
    col3.markdown(f"Lower color: {write_color(rgb_lower)}", unsafe_allow_html=True)
    col4.write("Example image:")
    if (dataset == "DeepFashion-MultiModal"):
      col4.image(requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/deepfashion/example-images/{value["path"]}.jpg').content, width=100)
    else:
      col4.image(requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/style-du-monde/example-images/{value["path"].split("/")[-1]}').content, width=100)
    # write path in a file
    # f = open(f"images-path-{dataset}", "a")
    # f = open(f"images-path-{gender}-{cluster_size}", "a")
    # f.write(f"{value['path']}\n")


def render_page():
  st.title("ENSIMAG AI project")
  st.header("Clothing color matching")
  st.write("This is a demo of our clothing color matching app. It will detect your skin tone and suggest you the most popular clothing colors matching your skin tone.")
  st.write("You can choose a dataset which is used to retrieve the clothing colors. We used two datasets: [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) and [Style du Monde](https://styledumonde.com).")
  st.write("When choosing the DeepFashion-MultiModal dataset, you can also choose your gender and the number of clothing color clusters you want to use.")
  st.write("The number of clothing color clusters is the number of colors used to represent the clothing colors. The higher the number, the more precise the result colors will be.")

  formCol1, formCol2, formCol3 = st.columns(3)
  dataset = formCol1.radio("Dataset", ["DeepFashion-MultiModal", "Style du Monde"])
  gender = formCol2.radio("Gender", ["Man", "Woman"], disabled=(dataset == "Style du Monde"))
  cluster_size = formCol3.radio("Number of clothing color clusters", ["10", "20"], disabled=(dataset == "Style du Monde"))

  file_name = st.file_uploader("Upload a full body image")

  with st.container():
    if st.button("Submit"):
      with st.spinner('Processing...'):
        get_result(file_name, dataset, gender, cluster_size)


if __name__ == "__main__":
  render_page()