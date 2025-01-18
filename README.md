WGU BSCS capstone  
fruits-and-vegetables  
Author: Yachong Tang  
Student ID: 011851202  

## This project include Two parts:
### 1. The training-model:
This part is used to train the model using the dataset. The tensorflow library used in this project has strict requirements for hardware and operating system, so this part of the code can **ONLY** run normally on Mac OS devices using Apple M-chips.  
    1. This part can **ONLY RUN** on a device with Mac OS and Apple M-chip.  
    2. To train the model, please download the zip of dataset from https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition . Unzip the the file. Make sure the train, validation, and test folders are under training-model/dataset/  
    3. Change directory to training-model `cd training-model`  
    4. Create virtual environment `python -m venv .venv-train`  
    5. Activate virtual enviroment `source .venv-train/bin/activate`  
    6. Install libraries. `pip install -r requirements-macos-Mchip.txt`  
    7. Run `python main.py` wait untill the training process to finish. It takes about 30 - 60 mins.  
    8. You will get some plots, and saved model 'MyModel.keras'  

### 2. The web-application:
This part is the Web Application part. The website uses the trained model to make predictions on the images uploaded by users.  
This part works for Mac OS, Windows, and Linux with python 3.9 and +.  
    1. Change directory to web-application `cd web-application`  
    2. If you installed requirements-macos-Mchip.txt from training-model, skip 3-5   
    3. Create virtual enviroment `python -m venv .venv-web`  
    4. Activate virtual enviroment `source .venv-web/bin/activate` for terminal.    
                    or `.\.venv-web\Scripts\activate.ps1` for windos Powershell.  
    5. Install libraries. `pip install -r requirements-web.txt`  
    6. Run `python app.py`  
    7. Go to http://127.0.0.1:5000/ and start to use the application. (First time run the application may not fully load, please use control + c to stop the flask project and re-run `python app.py` ).  
    8. There are a few images under /test-images for user to try the application.