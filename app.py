
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from datetime import datetime
from PIL import Image
import tensorflow as tf
import cv2

# ===== Prevent TensorFlow from using all memory =====
physical_devices = tf.config.list_physical_devices('CPU')
try:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
except Exception as e:
    print(f"⚠ Memory growth setup failed: {e}")
# ====================================================

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model paths
TB_MODEL_PATH = "tb_detection_model.h5"
STROKE_MODEL_PATH = "stroke_detection_resnet50.h5"

tb_model = None
stroke_model = None

# === Confidence threshold for rejecting invalid images ===
CONFIDENCE_THRESHOLD = 0.6

def get_tb_model():
    global tb_model
    if tb_model is None:
        if not os.path.exists(TB_MODEL_PATH):
            raise FileNotFoundError(f"TB model not found: {TB_MODEL_PATH}")
        tb_model = load_model(TB_MODEL_PATH, compile=False, custom_objects={'InputLayer': InputLayer})
    return tb_model

def get_stroke_model():
    global stroke_model
    if stroke_model is None:
        if not os.path.exists(STROKE_MODEL_PATH):
            raise FileNotFoundError(f"Stroke model not found: {STROKE_MODEL_PATH}")
        stroke_model = load_model(STROKE_MODEL_PATH, compile=False, custom_objects={'InputLayer': InputLayer})
    return stroke_model

@app.route('/')
def home():
    return render_template('choice.html')

@app.route('/tb')
def tb_form():
    return render_template('tbforms.html')

@app.route('/camera')
def show_tb_camera():
    return render_template('camera.html')

@app.route('/predict_tb', methods=['POST'])
def predict_tb():
    try:
        model = get_tb_model()

        form_data = {k: request.form.get(k) for k in
                     ['firstName', 'lastName', 'age', 'gender', 'phone', 'email', 'address', 'city']}

        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        filename = f'tb_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_input(np.expand_dims(image.img_to_array(
            Image.open(filepath).convert('RGB').resize((224, 224))), axis=0))

        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = float(prediction if prediction > 0.5 else (1 - prediction))

        # 🔹 Reject invalid images
        if confidence < CONFIDENCE_THRESHOLD:
            return render_template(
                'tbresults.html',
                label="Invalid Image: Please upload a proper Chest X-ray/CT scan.",
                confidence=round(confidence * 100, 1),
                uploaded_image=filename,
                gradcam_image=None,
                regions=[],
                report_id='TB-' + datetime.now().strftime('%Y%m%d%H%M%S'),
                **form_data
            )

        result = "TB Detected - High Confidence" if prediction > 0.5 else "No TB Detected - Low Risk"
        gradcam_filename, regions = generate_gradcam(model, img_array, filepath, layer_name='out_relu')

        return render_template(
            'tbresults.html',
            label=result,
            confidence=round(confidence * 100, 1),
            uploaded_image=filename,
            gradcam_image=gradcam_filename,
            regions=regions,
            report_id='TB-' + datetime.now().strftime('%Y%m%d%H%M%S'),
            **form_data
        )
    except Exception as e:
        print(f"❌ TB Prediction Error: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/stroke')
def stroke_form():
    return render_template('stkforms.html')

@app.route('/predictAction', methods=['POST'])
def predictAction():
    return redirect(url_for('show_stroke_camera'))

@app.route('/stkcamera')
def show_stroke_camera():
    return render_template('stkcamera.html')

@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
    try:
        model = get_stroke_model()

        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        filename = f'stk_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_input(np.expand_dims(image.img_to_array(
            Image.open(filepath).convert('RGB').resize((224, 224))), axis=0))

        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = float(prediction if prediction > 0.5 else (1 - prediction))

        # 🔹 Reject invalid images
        if confidence < CONFIDENCE_THRESHOLD:
            return render_template(
                'stkresults.html',
                label="Invalid Image: Please upload a proper Brain CT scan.",
                confidence=round(confidence * 100, 1),
                uploaded_image=filename,
                gradcam_image=None,
                stroke_regions=[],
                report_id='STK-' + datetime.now().strftime('%Y%m%d%H%M%S')
            )

        result = "Stroke Detected - High Confidence" if prediction > 0.5 else "No Stroke Detected - Low Risk"
        gradcam_filename, regions = generate_gradcam(model, img_array, filepath, layer_name='out_relu')

        return render_template(
            'stkresults.html',
            label=result,
            confidence=round(confidence * 100, 1),
            uploaded_image=filename,
            gradcam_image=gradcam_filename,
            stroke_regions=regions,
            report_id='STK-' + datetime.now().strftime('%Y%m%d%H%M%S')
        )
    except Exception as e:
        print(f"❌ Stroke Prediction Error: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

def generate_gradcam(model, img_array, img_path, layer_name, threshold=0.6):
    """Generate Grad-CAM heatmap and extract clickable regions."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Save superimposed Grad-CAM
    img = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + img

    gradcam_filename = f"gradcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(cam_path, superimposed_img)

    # Extract regions above threshold
    mask = np.uint8(heatmap_resized > threshold) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        explanation = f"AI detected unusual patterns in this region (size {w}x{h} px)"
        regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "explanation": explanation})

    return gradcam_filename, regions



from flask import Flask, render_template, request, jsonify

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    mode = data.get("mode")
    answers = data.get("answers", [])

    score = 0
    for ans in answers:
        if ans in ["yes", "y"]:  
            score += 1

    if mode == "stroke":
        if score >= 2:
            result = "⚠️ You may be at HIGH RISK of Stroke. Please take a scan to confirm."
        else:
            result = "✅ Your Stroke risk seems low, but consult a doctor if symptoms persist."
    elif mode == "tb":
        if score >= 2:
            result = "⚠️ You may be at HIGH RISK of TB. Please take a scan to confirm."
        else:
            result = "✅ Your TB risk seems low, but consult a doctor if symptoms persist."
    else:
        result = "❌ Invalid selection."

    return jsonify({"result": result})








if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
