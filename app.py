import io
import json
import torch
import base64
from io import BytesIO

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

# from pyfcm import FCMNotification


# push_service = FCMNotification(
#     api_key="AAAAicLngtE:APA91bEMDXGcE5suWeWd5JD1fJY1S2MJHdXMZYl4tbCCiHGfE5nvriAO05svXCJ4rBMd2aaDjtTfv5o0D9lrDfl9e2gKEIK_DbNKeCchESJ8Hk7Uj3FoxXvMa5_HOGAnrX7x60KvoMJ0"
# )

# # OR initialize with proxies

# proxy_dict = {"http": "http://127.0.0.1", "https": "http://127.0.0.1"}
# push_service = FCMNotification(
#     api_key="AAAAicLngtE:APA91bEMDXGcE5suWeWd5JD1fJY1S2MJHdXMZYl4tbCCiHGfE5nvriAO05svXCJ4rBMd2aaDjtTfv5o0D9lrDfl9e2gKEIK_DbNKeCchESJ8Hk7Uj3FoxXvMa5_HOGAnrX7x60KvoMJ0",
#     proxy_dict=proxy_dict,
# )


app = Flask(__name__)

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
    result = "Normal"
    if nnprob > 0.5:
        result = "Danger"

    print(result)
    return result
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

    # Your api-key can be gotten from:  https://console.firebase.google.com/project/sosclick/settings/cloudmessaging

    registration_id = "cHmvVSrhyaw:APA91bFmKwjvYyzugixggC2NWQTGdbbwz_XyOO-8Zcws4xfYPRvbTQRqKdyYBzKwOg8E9qIkBpauzEjYrilC9qKbec7FzU4_3PaM0uL8IsuoZrMXJ6dCmY3qxx04wLKK_g8-nqTu3U0P"
    message_title = "Uber update"
    message_body = "Hi john, your customized news for today is ready"
    result = push_service.notify_single_device(
        registration_id=registration_id,
        message_title=message_title,
        message_body=message_body,
    )

    print(result)

    return "test passed!"


if __name__ == "__main__":
    app.run()
