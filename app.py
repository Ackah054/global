import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress TensorFlow logs

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import uuid

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load Models
tb_model = load_model("tb_detection_model.h5")
stroke_model = load_model("stroke_detection_model.h5")

IMG_SIZE = (224, 224)  # Assuming both models use this input size
CONF_THRESHOLD = 0.7   # Confidence threshold

# =========================
# Helper: Validate if uploaded image looks like medical scan
# =========================
def validate_medical_image(filepath):
    """Basic heuristic check: reject colorful or unrealistic images for scans"""
    try:
        # Load grayscale
        gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            return False

        # Load color
        color_img = cv2.imread(filepath)

        # Check if too colorful (X-rays/CT are mostly grayscale)
        if np.std(color_img.reshape(-1, 3), axis=0).mean() > 45:  # threshold can be tuned
            return False

        # Brightness check (medical scans are not too dark/bright)
        mean_intensity = np.mean(gray_img)
        if mean_intensity < 30 or mean_intensity > 220:
            return False

        return True
    except:
        return False

# =========================
# Routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

# ---------- TB FORM ----------
@app.route('/tbforms', methods=['GET', 'POST'])
def tbforms():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        request.environ['tb_form_data'] = form_data
        return redirect(url_for('tb_camera'))
    return render_template('tbforms.html')

@app.route('/tb_camera')
def tb_camera():
    return render_template('camera.html', analysis_type="tb")

# ---------- Stroke FORM ----------
@app.route('/stkforms', methods=['GET', 'POST'])
def stkforms():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        request.environ['stk_form_data'] = form_data
        return redirect(url_for('stk_camera'))
    return render_template('stkforms.html')

@app.route('/stk_camera')
def stk_camera():
    return render_template('camera.html', analysis_type="stroke")

# ---------- TB ANALYSIS ----------
@app.route('/analyze_tb', methods=['POST'])
def analyze_tb():
    file = request.files['image']
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    form_data = getattr(request.environ, 'tb_form_data', {})

    # Validate image before running model
    if not validate_medical_image(filepath):
        return render_template(
            'tbresults.html',
            label="Error: Uploaded image does not appear to be a valid Chest X-ray.",
            confidence=None,
            uploaded_image=filename,
            gradcam_image=None,
            regions=[],
            report_id='TB-' + datetime.now().strftime('%Y%m%d%H%M%S'),
            **form_data
        )

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = tb_model.predict(img_array)
    confidence = float(np.max(preds))

    if confidence < CONF_THRESHOLD:
        label = "Error: Low confidence. Image may not be valid for TB analysis."
    else:
        label = "TB Detected" if np.argmax(preds) == 1 else "Normal"

    return render_template(
        'tbresults.html',
        label=label,
        confidence=round(confidence * 100, 2),
        uploaded_image=filename,
        gradcam_image=None,
        regions=[],
        report_id='TB-' + datetime.now().strftime('%Y%m%d%H%M%S'),
        **form_data
    )

# ---------- STROKE ANALYSIS ----------
@app.route('/analyze_stroke', methods=['POST'])
def analyze_stroke():
    file = request.files['image']
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    form_data = getattr(request.environ, 'stk_form_data', {})

    # Validate image before running model
    if not validate_medical_image(filepath):
        return render_template(
            'stkresults.html',
            label="Error: Uploaded image does not appear to be a valid Brain CT scan.",
            confidence=None,
            uploaded_image=filename,
            gradcam_image=None,
            regions=[],
            report_id='STK-' + datetime.now().strftime('%Y%m%d%H%M%S'),
            **form_data
        )

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = stroke_model.predict(img_array)
    confidence = float(np.max(preds))

    if confidence < CONF_THRESHOLD:
        label = "Error: Low confidence. Image may not be valid for Stroke analysis."
    else:
        label = "Stroke Detected" if np.argmax(preds) == 1 else "Normal"

    return render_template(
        'stkresults.html',
        label=label,
        confidence=round(confidence * 100, 2),
        uploaded_image=filename,
        gradcam_image=None,
        regions=[],
        report_id='STK-' + datetime.now().strftime('%Y%m%d%H%M%S'),
        **form_data
    )

# =========================
# Run Server
# =========================
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
