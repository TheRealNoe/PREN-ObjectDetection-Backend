# import necessary libraries 
import tensorflow as tf
import cv2
import numpy as np
import flask 
from flask import request 
import pandas as pd
import ssl
import os

# openssl variables
OPENSSL_PW = os.environ.get("OPENSSL_PW")
OPENSSL_CERTFILE = os.environ.get("OPENSSL_CERTFILE")
OPENSSL_KEYFILE = os.environ.get("OPENSSL_KEYFILE")

# create a Flask app object 
app = flask.Flask(__name__) 

# load model from disk using the saved path of the model file (.pd)  
model = tf.saved_model.load("models/3_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/")

@app.route("/detect", methods=["POST"])
def predict():
  image = request.files["image"]

  image_np = np.frombuffer(image.read(), dtype=np.uint8) 
  img = cv2.imdecode(image_np, cv2.IMREAD_COLOR) 
  preds = run_inference_for_single_image(model, img)
  return pd.Series(preds).to_json(orient="values")

def run_inference_for_single_image(model, image):
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

  return filtered_detections

if __name__ == "__main__":   
  context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
  context.load_cert_chain(certfile=OPENSSL_CERTFILE, keyfile=OPENSSL_KEYFILE, password=OPENSSL_PW) 
  app.run(port=443, host="0.0.0.0", ssl_context=context)