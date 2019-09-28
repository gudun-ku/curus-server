import io
import json
import torch
import base64
from io import BytesIO

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
imagenet_class_index = json.load(open("./imagenet_class_index.json"))
device = torch.device("cpu")
model = torch.load("./model3.h5", map_location=device)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)

    outputs = model.forward(tensor)
    model.eval()
    gprob, pred = outputs.max(1)
    prob = gprob.clone()
    nnprob = prob.detach().numpy()[0]

    if nnprob > 0.5:
        return "Danger"
    else:
        return "Normal"
    # return imagenet_class_index[predicted_idx]


@app.route("/predimg", methods=["POST"])
def predimg():
    if request.method == "POST":
        jsonData = request.get_json()
        file = jsonData["img"]
        starter = file.find(",")
        image_data = file[starter + 1 :]

        image_data = bytes(image_data, encoding="ascii")
        img_bytes = BytesIO(base64.b64decode(image_data))
        im = Image.open(img_bytes)
        # im.save("./image.jpg")
        # return jsonify("OK")
        class_name = get_prediction(image_bytes=base64.b64decode(image_data))
        return jsonify({"class_name": class_name})


@app.route("/test")
def hello():
    return "test passed!"


if __name__ == "__main__":
    app.run()
