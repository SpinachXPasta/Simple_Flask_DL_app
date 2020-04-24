import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template, session, send_from_directory, Markup
import tensorflow.keras as keras

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from PIL import Image
import re
import cv2
from labels import labels
import os
import time
import pandas as pd





app = Flask(__name__, static_url_path = '/user_upload', static_folder = 'user_upload')


def setup():
    global model
    model = keras.applications.mobilenet.MobileNet(weights = 'imagenet')
    print ("Process Complete \n\n")

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:

        print ("\n\n\n",request.files,"\n\n\n")

        if request.files.get('filename') != None:
            Img_upload = Image.open(request.files['filename'].stream).convert('RGB')

            temp_fname = 'temp' + str(np.random.randint(100,999)) + ".jpg"
            original_fname = 'original' + str(np.random.randint(100,999)) + ".jpg"

            print (temp_fname)

            print ("Img", Img_upload)

            open_cv_image = np.array(Img_upload)

            dir_path = os.path.dirname(os.path.realpath(__file__))

            cv2.imwrite(dir_path+"/user_upload/"+original_fname,open_cv_image[:,:,::-1])

            height, width, _ = open_cv_image.shape

            im = cv2.resize(open_cv_image, (224,224))
            im2 = keras.applications.mobilenet.preprocess_input(im)

            Pred = model.predict(im2.reshape(1,224,224,3))
            Prediction = np.argmax(Pred)



            df = pd.DataFrame({'ID':range(len(Pred.reshape(-1))),'Score':Pred.reshape(-1)})
            df['Score'] = df['Score'].round(5)
            labeldf = pd.DataFrame({'ID':list(labels.keys()),'Values':list(labels.values())})
            df = df.sort_values(by = 'Score', ascending = False)
            df = df.merge(labeldf, on ='ID')
            vizDf = df.loc[0:10]

            f = plt.figure(figsize = (6,6))
            plt.barh(vizDf.Values,vizDf.Score)
            f.savefig(dir_path+"/user_upload/"+temp_fname,bbox_inches = 'tight')


            query =labels[Prediction]
            idx = query.find(",")
            if idx > 0:
                query = query[0:idx]
            query = query.replace(" ","_")

            search = "https://en.wikipedia.org/wiki/"+query

            time.sleep(2)
            data = requests.get(search)

            soup = BeautifulSoup(data.text, 'html.parser')



            output_text = ""
            for e,i in enumerate(soup.findAll('p')):
                if e > 0:
                    if remove_html_tags(str(i)).strip() != "":
                        output_text += remove_html_tags(str(i))
                        break




            return render_template('output.html', Words = output_text, original = original_fname, result = temp_fname)

    except Exception as e:
        print ("\n\n\n",str(e),"\n\n\n")
        pass

    return render_template('upload.html')





if __name__ == "__main__":
    setup()
    app.run()
