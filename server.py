# import necessary libraries 
import tensorflow as tf
import cv2
from six import BytesIO
import numpy as np
from PIL import Image
import flask 
from flask import request, jsonify 
import pandas as pd
  
# create a Flask app object 
app = flask.Flask(__name__) 

# load model from disk using the saved path of the model file (.pd)  
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/")

@app.route("/detect", methods=["POST"])
def predict():
  image = request.files["image"]
  deltaX = request.form.get("deltaX")
  deltaY = request.form.get("deltaY")

  if(isinstance(deltaX, str)):
    deltaX = int(deltaX)
  else:
    deltaX = 1
  
  if(isinstance(deltaY, str)):
    deltaY = int(deltaY)
  else:
    deltaY = 1

  if(deltaX < 1):
    deltaX = 1

  if(deltaY < 1):
    deltaY = 1

  image_np = np.frombuffer(image.read(), dtype=np.uint8) 
  img = cv2.imdecode(image_np, cv2.IMREAD_COLOR) 
  preds = run_inference_for_single_image(model, img, deltaX, deltaY)
  return pd.Series(preds).to_json(orient="values")

def run_inference_for_single_image(model, image, deltaX, deltaY):
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures["serving_default"]
  output_dict = model_fn(input_tensor)

  # Extract the relevant information from the output dictionary
  num_detections = int(output_dict["num_detections"][0])
  classes = [int(output_dict["detection_classes"][0, i]) for i in range(num_detections)]
  scores = [float(output_dict["detection_scores"][0, i]) for i in range(num_detections)]
  boxes = [[int(output_dict["detection_boxes"][0, i, j] * image.shape[0]) for j in range(4)] for i in range(num_detections)]

  # Filter detections with score lower than 0.8
  filtered_detections = []
  for i in range(num_detections):
    if scores[i] >= 0.8:
      filtered_detections.append({
        "class": classes[i],
        "score": scores[i],
        "box": boxes[i]
      })

  # Find detection with highest probability in each cluster
  highest_probability_boxes = find_highest_probability_box(filtered_detections, deltaX, deltaY)

  return highest_probability_boxes

def find_highest_probability_box(detections, delta_x, delta_y):
  detection_clusters = {}
  
  # Group detections into clusters based on delta_x and delta_y
  for detection in detections:
    cluster_key = (detection["box"][0] // delta_x, detection["box"][1] // delta_y)
    if cluster_key not in detection_clusters:
      detection_clusters[cluster_key] = []
    detection_clusters[cluster_key].append(detection)
  
  # Find detection with highest probability in each cluster
  highest_score_detections = []
  for cluster_detections in detection_clusters.values():
    highest_score_detection = max(cluster_detections, key=lambda box: box["score"])
    highest_score_detections.append(highest_score_detection)
  
  return highest_score_detections

if __name__ == "__main__":
  app.run(debug=True, port=443, host="0.0.0.0")
    
