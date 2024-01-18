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
  return colors[0][0][0]


def get_clothes_and_skin_colors(image):
  # Load model
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
  upper_mask = np.full((rows, cols), False, dtype=bool)
  lower_mask = np.full((rows,cols), False, dtype=bool)
  skin_mask = np.full((rows,cols), False, dtype=bool)
  dress_mask = np.full((rows,cols), False, dtype=bool)
  has_upper = False
  has_lower = False
  has_skin = False
  has_dress = False
  # Iterate through the 2D tensor array with indices

  # Convert pytorch tensor to numpy array to optimize
  segmentation = pred_seg.detach().cpu().numpy()

  for i in range(rows):
      for j in range(cols):
        v = segmentation[i, j]
        if v == 4: # upper clothes
          upper_mask[i, j] = True
          has_upper = True
        elif v == 6 or v == 5: # pants or skirt
          lower_mask[i, j] = True
          has_lower = True
        elif v in [11, 12, 13, 14, 15]: #body parts
          skin_mask[i, j]=  True
          has_skin = True
        elif v == 7: # dress:
          dress_mask[i, j] =  True
          has_dress = True


  if not has_skin:
    return None
  if has_dress:
    dress_color = getColorDominantFromMask(image, dress_mask)
    return (dress_color, dress_color,getColorDominantFromMask(image, skin_mask))
  elif not has_upper or not has_lower:
    return None
  else:
    return (getColorDominantFromMask(image, upper_mask), getColorDominantFromMask(image, lower_mask), getColorDominantFromMask(image, skin_mask))


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


def get_closest_cluster_index(lab_color, clusters):
    lab_input = rgb_to_lab(lab_color)
    # find the closest cluster index to the input color
    closest_cluster_index = 0
    closest_cluster_distance = 10000
    for cluster_index in range(len(clusters)):
        lab_cluster = clusters[cluster_index]
        distance = np.linalg.norm(lab_input - lab_cluster)
        if distance < closest_cluster_distance:
            closest_cluster_distance = distance
            closest_cluster_index = cluster_index
    return closest_cluster_index


def get_occurences_by_skin_cluster_index(skin_cluster_index, occurences):
    occurences_by_skin_cluster = {}
    for key, value in occurences.items():
        if value['skin_cluster'] == skin_cluster_index:
            occurences_by_skin_cluster[key] = value
    occurences_by_skin_cluster = {k: v for k, v in sorted(occurences_by_skin_cluster.items(), key=lambda item: item[1]['occurences'], reverse=True)}
    return occurences_by_skin_cluster


def get_max_occ(cluster):
  max_key=max(cluster, key=lambda k: cluster[k]['occurences'])
  return cluster[max_key]['occurences']


def compute_score(upper_clusters, lower_clusters, skin_clusters, colors_set, occurences_by_skin_cluster, occurences):
    max_occ = get_max_occ(occurences_by_skin_cluster)

    # if there is an undetected color, we dont give a score
    if colors_set == None:
      return None
    upper_color, lower_color, skin_color = colors_set
    upper_lab, lower_lab, skin_lab = rgb_to_lab(upper_color), rgb_to_lab(lower_color), rgb_to_lab(skin_color)
    closest_upper_cluster, closest_lower_cluster,closest_skin_cluster = get_closest_cluster_index(upper_lab, upper_clusters), get_closest_cluster_index(lower_lab, lower_clusters), get_closest_cluster_index(skin_lab, skin_clusters)
    key = str(closest_skin_cluster) + ',' + str(closest_upper_cluster) +',' +str(closest_lower_cluster)
    print(key)
    if key in occurences:
        return (occurences[key]['occurences']/max_occ) * 30 + 70
    else:
        return None


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
  image = Image.open(file_name).convert("RGB")
  st.image(image)

  # display the detected skin color
  colors_set = get_clothes_and_skin_colors(image)
  if (colors_set is None):
    st.error("Please upload a full body image")
    return
  skin_color = colors_set[2]
  st.header("Detected skin tone")
  st.markdown(write_color(skin_color), unsafe_allow_html=True)

  # display a score
  cluster_centers_upper = clusters["cluster_centers_upper"]
  cluster_centers_lower = clusters["cluster_centers_lower"]
  cluster_centers_skin = clusters["cluster_centers_skin"]
  closest_skin_cluster_index = get_closest_cluster_index(skin_color, cluster_centers_skin)
  occurences_by_skin_cluster = get_occurences_by_skin_cluster_index(closest_skin_cluster_index, occurences)
  score = compute_score(cluster_centers_upper,  cluster_centers_lower, cluster_centers_skin, colors_set, occurences_by_skin_cluster, occurences)
  st.header("Score")
  if score is None:
    st.write("We couldn't determine your score. Please try again with another image.")
  else:
    st.write(f"{score}%")

  # display the clothing colors occurences matching the detected skin color
  st.header("Most popular clothing colors matching your skin tone")
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
      col4.image(requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/deepfashion/example-images/{value["path"]}.jpg').content)
    else:
      col4.image(requests.get(f'https://raw.githubusercontent.com/Nielk74/ia-clothes/master/data/results/style-du-monde/example-images/{value["path"].split("/")[-1]}').content)


def render_page():
  st.title("ENSIMAG AI project")
  st.header("Clothing color matching")
  st.write("This is a demo of our clothing color matching app. It will detect your skin tone and suggest you the most popular clothing colors matching your skin tone.")
  st.write("You can choose the dataset used to retrieve the clothing colors. We used two datasets: [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) and [Style du Monde](https://styledumonde.com).")
  st.write("When choosing the Style du Monde dataset, be aware that it is more suited for women.")
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