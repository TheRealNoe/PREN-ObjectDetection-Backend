import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

url = "http://prenh22-naufdenb.el.eee.intern:443/detect"
image = "test_img.jpg"
post_data = {"deltaX": 20, "deltaY": 20}

with open(image, "rb") as file:
    # Create a dictionary containing the file as the value
    files = {"image": file}

    # Send the POST request with the image file and post data as the payload
    response = requests.post(url, files=files, data=post_data)

    # Check the response status code
    if response.status_code == 200:
        parsed = response.json()
      
        im = Image.open(image)

        # Display the image
        plt.imshow(im)

        # Get the current reference
        ax = plt.gca()
        ax.invert_yaxis()

        for detection in parsed:
            print(detection)
            
            x = detection["box"][0]
            y = detection["box"][1]
            width = abs(x - detection["box"][2])
            height = abs(detection["box"][1] - detection["box"][3])
            
            # Create a Rectangle patch
            rect = Rectangle((y, x), height, width, linewidth = 1, edgecolor = "r", facecolor = "none")
            ax.add_patch(rect)
        
        plt.show()
        
        print(parsed)
    else:
        print("Failed to upload image. Status code:", response.status_code)
