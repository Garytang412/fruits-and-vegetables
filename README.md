WGU BSCS capstone  
fruits-and-vegetables<br>
Author: Yachong Tang<br>
Student ID: 011851202<br>

## This project include Two parts:
### 1 The training-model:
This part 
    1. This part can **ONLY RUN** on a device with Macos and apple M-chip.<br>
    2. To train the model, please download the zip of dataset from https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition . Unzip the the file. Make sure the train, validation, and test folders are under training-model/dataset/<br>
    3. Change directory to training-model `cd training-model`<br>
    4. Create virtual environment `python -m venv .venv-train`<br>
    5. Activate virtual enviroment `source .venv-train/bin/activate`<br>
    6. Install libraries. `pip install -r requirements-macos-Mchip.txt`<br>
    7. Run `python main.py` wait untill the training process to finish. It takes about 30 - 60 mins.<br>
    8. You will get some plots, and saved model 'MyModel.keras'<br>

### 2 The web-application:
This part is the Application part. The website uses the trained model to make predictions on the images uploaded by users.  
    1. Change directory to web-application `cd web-application`<br>
    2. If you installed requirements-macos-Mchip.txt from training-model, skip 3-5 <br>
    3. Create virtual enviroment `python -m venv .venv-web`<br>
    4. Activate virtual enviroment `source .venv-web/bin/activate`<br>
    5. Install libraries. `pip install -r requirements-web.txt`<br>
    6. Run `python app.py`<br>
    7. Visite http://127.0.0.1:5000/ and start use application.<br>

