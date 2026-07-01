from PIL import Image
from io import BytesIO
import pickle
import traceback
from reward_server.ours import load_ours
import numpy as np
import os

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_ours()

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"]) 
def inference():
    print(f"received POST request from {request.remote_addr}")
    data = request.get_data()

    try:

        data = pickle.loads(data)


        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
        meta_datas = data["meta_datas"]
        prompts = data['prompts']

        print(f'{data.keys()}')
        print(f"Got {len(images)} images")



        batch_tensors = []
        batch_obj_details = []
        batch_spatial_details = []

        for img, meta, prompt in zip(images, meta_datas, prompts):


            r_tensor, r_obj_det, r_spatial_det = INFERENCE_FN(img, meta, prompt)



            batch_tensors.append(r_tensor.cpu().tolist())


            batch_obj_details.append(r_obj_det)
            batch_spatial_details.append(r_spatial_det)


        response = {
            "reward_tensors": batch_tensors,
            "detailed_object_rewards": batch_obj_details,
            "detailed_spatial_scores": batch_spatial_details
        }


        response = pickle.dumps(response)
        returncode = 200

    except Exception as e:
        response = traceback.format_exc()
        print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode

HOST = "127.0.0.1"
PORT = 8095

if __name__ == "__main__":
    create_app().run(HOST, PORT)