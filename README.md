WGU BSCS capstone
fruits-and-vegetables
Author: Yachong Tang
Student ID: 011851202

## This project include Two parts:
### 1 The training-model:
This part 
    1. This part can **ONLY RUN** on a device with Macos and apple M-chip.
    2. To train the model, please download the zip of dataset from https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition . Unzip the the file. Make sure the train, validation, and test folders are under training-model/dataset/
    3. Change directory to training-model `cd training-model`
    4. Create virtual environments `python -m venv .venvtrain`
    5. Install libraties. `pip install -r requirements-macos-Mchip.txt`
    6. Run `python main.py` wait untill the training process to finish. It takes about 30 - 60 mins.

### 2 The web-application:
This part is the Application part. The website uses the trained model to make predictions on the images uploaded by users.
    1. Change directory to web-application `cd web-application`
    2. Use python virtual environments or directly install requirement libraries `pip install -r requirements-web.txt`
    3. 