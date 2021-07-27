from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests, json, urllib.request,os
from PIL import Image, ImageDraw, ImageFont
import cv2
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import json

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def FaceRecog(localfileName):
    # 別途指定する値に書き換える
    subscription_key = ''
    face_api_url = ''

    # 顔認識させる画像
    image_file_path = localfileName
    image_file = open(image_file_path, "rb")
    body = image_file.read()
    image_file.close()

    # ヘッダ設定
    headers = {
        'Content-Type' : 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    # パラメーターの設定
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'recognitionModel' : 'recognition_04',
        'returnRecognitionModel' : 'false',
        'detectionModel' : 'detection_01',
        'faceIdTimeToLive' : '86400',
    }

    # POSTリクエストの試行
    try:
        request = requests.post(face_api_url, params=params, headers=headers, data=body, timeout=30)
        response = request.json()

        raw_img = cv2.imread(localfileName)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        for item in response:
            x = item["faceRectangle"]["left"]
            y = item["faceRectangle"]["top"]
            w = item["faceRectangle"]["width"]
            h = item["faceRectangle"]["height"]

            raw_img = cv2.putText(raw_img, "age:"+str(int(item["faceAttributes"]["age"])), (x+6, y+20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, "gen:"+str(item["faceAttributes"]["gender"]), (x+6, y+39), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            raw_img = cv2.rectangle(raw_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # figsizeとdpiを変化させることで、写真の大きさを変更することが可能
        plt.figure(figsize=(4, 4), dpi=1000)
        plt.imshow(raw_img)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)    
        plt.show()

    except requests.exceptions.ConnectionError:
        print("Site not rechable", face_api_url)
        
if __name__ == "__main__":
    FaceRecog("sample1.jpg")
