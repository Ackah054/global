"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.keras.layers import InputLayer

app = Flask(__name__)

# Setup upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model paths
TB_MODEL_PATH = 'tb_detection_model.h5'
STROKE_MODEL_PATH = 'stroke_detection_resnet50.h5'

@app.route('/')
def home():
    return render_template('choice.html')

@app.route('/tb')
def tb_form():
    return render_template('tbforms.html')

@app.route('/camera', methods=['GET'])
def show_tb_camera():
    return render_template('camera.html')


@app.route('/predict_tb', methods=['POST'])
def predict_tb():
    try:
        # ======== 1. GET FORM DATA ========
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        email = request.form.get('email')
        address = request.form.get('address')
        city = request.form.get('city')
        state = request.form.get('state')
        country = request.form.get('country')
        previous_tb = request.form.get('previousTB')
        symptom_duration = request.form.get('symptomDuration')
        additional_info = request.form.get('additionalInfo')
        symptoms = request.form.getlist('symptoms')
        risk_factors = request.form.getlist('riskFactors')

        # ======== 2. PROCESS IMAGE ========
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        filename = f'tb_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_pil = Image.open(filepath).convert('RGB')
        image_resized = image_pil.resize((224, 224))
        img_array = image.img_to_array(image_resized)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model(TB_MODEL_PATH, compile=False, custom_objects={'InputLayer': InputLayer})
       # model = load_model(TB_MODEL_PATH, compile=False)
        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction * 100) if prediction > 0.5 else float((1 - prediction) * 100)
        result = "TB Detected - High Confidence" if prediction > 0.5 else "No TB Detected - Low Risk"

        gradcam_filename = generate_gradcam(model, img_array, filepath, 'conv4_block6_out')

        # ======== 3. RETURN TO RESULTS PAGE ========
        return render_template(
            'tbresults.html',
            label=result,
            confidence=round(confidence, 1),
            uploaded_image=filename,
            gradcam_image=gradcam_filename,
            report_id='TB-' + datetime.now().strftime('%Y%m%d%H%M%S'),
            first_name=first_name,
            last_name=last_name,
            age=age,
            gender=gender,
            phone=phone,
            city=city
            
        )

    except Exception as e:
        print(f"❌ TB Prediction Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/stroke')
def stroke_form():
    return render_template('stkforms.html')

@app.route('/predictAction', methods=['POST'])
def predictAction():
    return redirect(url_for('show_stroke_camera'))

@app.route('/stkcamera', methods=['GET'])
def show_stroke_camera():
    return render_template('stkcamera.html')

@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
    try:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        filename = f'stk_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_pil = Image.open(filepath).convert('RGB')
        image_resized = image_pil.resize((224, 224))
        img_array = image.img_to_array(image_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model(STROKE_MODEL_PATH, compile=False)
        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction * 100) if prediction > 0.5 else float((1 - prediction) * 100)
        result = "Stroke Detected - High Confidence" if prediction > 0.5 else "No Stroke Detected - Low Risk"

        gradcam_filename = generate_gradcam(model, img_array, filepath, 'conv4_block6_out')

        return render_template(
            'stkresults.html',
            label=result,
            confidence=round(confidence, 1),
            uploaded_image=filename,
            gradcam_image=gradcam_filename,
            report_id='STK-' + datetime.now().strftime('%Y%m%d%H%M%S')
        )

    except Exception as e:
        print(f"❌ Stroke Prediction Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def generate_gradcam(model, img_array, img_path, layer_name='conv4_block6_out'):
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

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + img

    gradcam_filename = f"gradcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(cam_path, superimposed_img)

    return gradcam_filename

if __name__ == '__main__':
    app.run(debug=True)
"""








import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from datetime import datetime
from PIL import Image
import tensorflow as tf
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

TB_MODEL_PATH = "tb_detection_model.h5"
STROKE_MODEL_PATH = "stroke_detection_resnet50.h5"

# Lazy-loaded model variables
tb_model = None
stroke_model = None

def get_tb_model():
    global tb_model
    if tb_model is None:
        print("📥 Loading TB model...")
        tb_model = load_model(TB_MODEL_PATH, compile=False, custom_objects={'InputLayer': InputLayer})
    return tb_model

def get_stroke_model():
    global stroke_model
    if stroke_model is None:
        print("📥 Loading Stroke model...")
        stroke_model = load_model(STROKE_MODEL_PATH, compile=False, custom_objects={'preprocess_input': preprocess_input})
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

        # Form data
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        email = request.form.get('email')
        address = request.form.get('address')
        city = request.form.get('city')

        # Image
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        filename = f'tb_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_pil = Image.open(filepath).convert('RGB')
        image_resized = image_pil.resize((224, 224))
        img_array = image.img_to_array(image_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction * 100) if prediction > 0.5 else float((1 - prediction) * 100)
        result = "TB Detected - High Confidence" if prediction > 0.5 else "No TB Detected - Low Risk"

        gradcam_filename = generate_gradcam(model, img_array, filepath, 'conv4_block6_out')

        return render_template(
            'tbresults.html',
            label=result,
            confidence=round(confidence, 1),
            uploaded_image=filename,
            gradcam_image=gradcam_filename,
            report_id='TB-' + datetime.now().strftime('%Y%m%d%H%M%S'),
            first_name=first_name,
            last_name=last_name,
            age=age,
            gender=gender,
            phone=phone,
            city=city
        )
    except Exception as e:
        print(f"❌ TB Prediction Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        filename = f'stk_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_pil = Image.open(filepath).convert('RGB')
        image_resized = image_pil.resize((224, 224))
        img_array = image.img_to_array(image_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction * 100) if prediction > 0.5 else float((1 - prediction) * 100)
        result = "Stroke Detected - High Confidence" if prediction > 0.5 else "No Stroke Detected - Low Risk"

        gradcam_filename = generate_gradcam(model, img_array, filepath, 'conv4_block6_out')

        return render_template(
            'stkresults.html',
            label=result,
            confidence=round(confidence, 1),
            uploaded_image=filename,
            gradcam_image=gradcam_filename,
            report_id='STK-' + datetime.now().strftime('%Y%m%d%H%M%S')
        )
    except Exception as e:
        print(f"❌ Stroke Prediction Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def generate_gradcam(model, img_array, img_path, layer_name='conv4_block6_out'):
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

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + img

    gradcam_filename = f"gradcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    cam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(cam_path, superimposed_img)

    return gradcam_filename

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
