# DL Hackathon – Eye Vessel Segmentation and Health Indicator Web App

Welcome to our submission for the IIT Mandi Hackathon! This project is all about making retinal vessel analysis more accessible for everyone. With just an image of your eye, our web app can segment out the blood vessels and give you a potential health indicator—all in a few seconds.

---

## What does this app do?

Our web application takes an eye image and does all the heavy lifting:

1. **Uploads your eye image** through a simple browser interface.
2. **(Optionally) Detects the region of interest** (your eye) with a Haar Cascade to focus processing.
3. **Segments the vessels** using a U-Net (with ResNet34 backbone) deep learning model.
4. **Refines the segmentation** with post-processing (morphological ops like connected components and closing).
5. **Extracts key vessel features**: density, branch/endpoints, tortuosity, length, width, and more.
6. **Normalizes the features** with a pre-trained MinMaxScaler.
7. **Predicts a health cluster** using a pre-trained KMeans model.
8. **Returns** the segmented vessel mask (as an image) and a health indicator (as text).

---

## How does it look?

Below are a few screenshots from the web app and model outputs:

### Architecture

![Screenshot from 2025-06-27 08-57-57](https://github.com/user-attachments/assets/6da9584a-f71c-4322-b815-c296b708e590)

### Website Interface

![Screenshot from 2025-06-27 08-56-29](https://github.com/user-attachments/assets/8268817d-9876-4399-b2c5-49e1f90632a3)

### Example: Original vs Segmented Output

![Screenshot from 2025-06-27 08-47-24](https://github.com/user-attachments/assets/06f3887a-3419-493e-b14e-487839adbb50)

---

## 🛠️ Getting Started

This repo does **not** include the trained models (they’re too big for GitHub). You’ll need to set up the models yourself. Here’s how:

### 1. Prepare your data & models

- Place your dataset at the same level as `test_images/`.
- Open `unet_feat/unet_training_independent.py` and check the `BASE_PATH` variable.
- For faster experimentation, try reducing the image size from `(1024, 1024)` to `(512, 512)` or `(256, 256)`. Don’t set `batch_size` above 4.
- Run the training script. It will save three models (`*.pth`, `*.joblib`) in the `models/` folder and predictions in `predictions/`.
- For standalone scripts:
  - U-Net training: `unet_feat_combined.py`
  - Feature extraction: `feature_extractor_independent.py`

### 2. Required files (must be in the right folders)

- `bestmodel.pth` – U-Net weights (vessel segmentation)
- `scaler.joblib` – Pre-trained MinMaxScaler (feature normalization)
- `kmeans.joblib` – Pre-trained KMeans (clustering)
- `haarcascade_eye.xml` – Haar Cascade for eye region detection

### 3. Environment setup

Make sure you have:

- **Python 3.x**
- **pip**
- **Git** (for cloning the repo)

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

Navigate to the `dl/dlhackathon` directory (where `app.py` is) and start the server:

```bash
python app.py
# or
python3 app.py
```

You should see output like:

```
Starting Flask server on port 5001...
 * Serving Flask app 'app'
 * Debug mode: on
...
```

If port 5001 is busy, either free it up or change the port in both `app.py` and `templates/index.html` (switch to 5002, for example).

### 5. See it in action

Open your browser to [http://127.0.0.1:5002/](http://127.0.0.1:5002/) (or whatever address appears in the terminal).  
You’ll see the web interface where you can upload an eye image and get instant results!

---

## Repo Structure

- **unet_feat/**: Model training, feature extraction scripts
- **models/**: Where trained models are saved (you'll generate these)
- **predictions/**: Stores segmented output images
- **test_images/**: Example input images
- **templates/index.html**: Main web page
- **app.py**: Flask application (the backend)

---

## ✨ Features At a Glance

- **Easy-to-use web interface**
- **Accurate vessel segmentation** with deep learning
- **Automated health indicator** (clustering-based)
- **Modular code**: train your own models, extract features, or use as a complete app

---

## ⚠️ Notes

- You’ll need your own data/models to run this (see above).
- If you get a port error, just switch to another port in both the backend and frontend files.
- For any issues, make sure all dependencies in `requirements.txt` are installed.

---

## 👏 Credits

This project was created as part of the IIT Mandi Hackathon by the awesome team at [@JohnPrice11/IITMandiHackathon](https://github.com/JohnPrice11/IITMandiHackathon).

---

Feel free to fork, experiment, or reach out with feedback. Happy segmenting!
