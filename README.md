# Sign Language Detection Using Mediapipe and ANN

This repository contains code and resources for a Sign Language Detection project utilizing Mediapipe and Artificial Neural Networks (ANN). The project aims to build a sign language recognition system capable of real-time gesture detection.

## Repository Structure

- **`collect_data.ipynb`**: A Jupyter Notebook for collecting and preprocessing sign language data. This notebook handles data acquisition from video feeds and stores it in an organized format.
- **`split_data.py`**: A Python script for splitting the collected data into training and testing datasets. It ensures that data is appropriately divided for model training and evaluation.
- **`train_model.ipynb`**: A Jupyter Notebook for training an artificial neural network model on the preprocessed sign language data. This notebook includes model architecture definition, training, and evaluation steps.
- **`real_time_detection.py`**: A Python script for performing real-time sign language detection using the trained model. It captures live video, preprocesses the input, and provides predictions.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nomanjaffar1/Sign_Launguage_detection_using_mediapipe_ANN.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd Sign_Launguage_detection_using_mediapipe_ANN
   ```

3. **Set Up the Environment**

   Create a virtual environment and install the required dependencies. You can use `pip` to install the necessary packages. A `requirements.txt` file can be added to specify the dependencies.

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `.\env\Scripts\activate`
   pip install -r requirements.txt
   ```

   **Note:** Ensure you have the following packages installed:
   - TensorFlow
   - OpenCV
   - Mediapipe
   - scikit-learn
   - joblib

## Usage

1. **Data Collection**

   Run `collect_data.ipynb` to collect sign language gesture data. Follow the instructions within the notebook to gather and save data.

2. **Data Splitting**

   Execute `split_data.py` to split the collected data into training and testing datasets. This script prepares the data for model training and evaluation.

   ```bash
   python split_data.py
   ```

3. **Model Training**

   Open and run `train_model.ipynb` to train the ANN model on the preprocessed data. The notebook includes steps for defining, training, and evaluating the model.

4. **Real-Time Detection**

   Use `real_time_detection.py` to perform real-time sign language detection. This script will use the trained model to predict gestures from live video input.

   ```bash
   python real_time_detection.py
   ```

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes. Ensure that your code adheres to the project's coding standards and includes appropriate documentation.


Feel free to adjust the content to better fit your specific needs or additional details about the project.
