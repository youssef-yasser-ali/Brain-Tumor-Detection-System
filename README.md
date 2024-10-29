## Brain Tumor Detection System

This is a FastAPI project designed for brain tumor classification and segmentation using PyTorch. The project is containerized with Docker and can be deployed.

### Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Using the API](#using-the-api)

### Project Overview

This API provides endpoints for:

1. **Classification**: Determines the type of brain tumor present in an image.
   - **Classes**: The classification model distinguishes between four types:
     - `No Tumor`
     - `Glioma`
     - `Meningioma`
     - `Pituitary Tumor`
2. **Segmentation**: Identifies the tumor's location within the brain image.

This project is built using:

- **FastAPI** for API management.
- **PyTorch** for machine learning model handling.
- **Docker** for containerization.

---

### Folder Structure

The project structure is organized as follows:

```plaintext
.
├── app
│   ├── api
│   │   ├── classification.py         # Classification endpoint logic
│   │   └── segmentation.py           # Segmentation endpoint logic
│   ├── config.py                     # Configuration file
│   ├── index.html                    # Basic UI for testing
│   ├── main.py                       # FastAPI entry point
│   ├── models
│   │   ├── classification_model.py   # Classification model definition
│   │   ├── segmentation_model.py     # Segmentation model definition
│   │   └── pre_trained
│   │       ├── class_brain_tumor     # Classification model weights
│   │       └── seg_brain_tumor_model.pth  # Segmentation model weights
│   └── utils.py                      # Helper functions
├── notebooks                         # Folder for model training notebooks
│   ├── classification_training.ipynb # Notebook for training classification model
│   └── segmentation_training.ipynb   # Notebook for training segmentation model
├── Dockerfile                        # Docker configuration file
└── requirements.txt                  # Python dependencies
```

---

### Setup and Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd brain-tumor-detection-api
   ```

2. **Install Dependencies**:
   Ensure Python 3.9+ is installed, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the FastAPI app locally:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

---

### Using the API

The API has two main endpoints:

- **`POST /api/classification`**: Accepts an image file and returns one of the following classification labels:

  - `No Tumor`
  - `Glioma`
  - `Meningioma`
  - `Pituitary Tumor`

- **`POST /api/segmentation`**: Accepts an image file and returns a segmented image showing the tumor location.

#### Example Request (Classification):

```bash
curl -X POST "http://127.0.0.1:8000/api/classification" -F "file=@<path_to_image>"
```

#### Example Request (Segmentation):

```bash
curl -X POST "http://127.0.0.1:8000/api/segmentation" -F "file=@<path_to_image>"
```

---

---

### Model Training Notebooks

The `notebooks` folder contains Jupyter notebooks used for training the classification and segmentation models:

- **classification_training.ipynb**: Contains code for data preprocessing, model architecture, training, and evaluation for the classification task.
- **segmentation_training.ipynb**: Covers the data preparation, model setup, training, and performance evaluation for the segmentation model.

To run these notebooks, navigate to the `notebooks` folder and open them with Jupyter:

```bash
jupyter notebook notebooks/classification_training.ipynb
```

These notebooks document the model's training and provide steps to retrain or fine-tune the models as needed.

---

### Docker

#### Building and Running with Docker

1. **Build the Docker Image**:

   ```bash
   docker build -t brain_tumor_detection_api .
   ```

2. **Run the Docker Container**:

   ```bash
   docker run -d -p 8000:8000 brain_tumor_detection_api
   ```

   Access the application at `http://localhost:8000/`.

### Acknowledgements

- **FastAPI** for the web framework.
- **PyTorch** for deep learning models.
