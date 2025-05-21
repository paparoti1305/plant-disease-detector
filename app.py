import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'model/model_epoch_15 (1).h5'
model = load_model(MODEL_PATH)

# Load class mapping
with open('model/class_indices_moi.json', 'r') as f:
    class_indices = json.load(f)

# Tạo dictionary phân loại và thông tin bệnh
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "description": "Bệnh đốm vảy trên táo do nấm Venturia inaequalis gây ra, xuất hiện các đốm nâu/xám trên lá và quả, làm giảm năng suất và chất lượng quả.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc gốc đồng hoặc thuốc chứa mancozeb, vệ sinh vườn cây."
    },
    "Apple___Cedar_apple_rust": {
        "name": "Apple Rust",
        "description": "Bệnh gỉ sắt trên táo, gây ra các đốm màu cam trên lá, có thể làm rụng lá sớm và ảnh hưởng đến sự phát triển của cây.",
        "treatment": "Loại bỏ lá bệnh, dùng thuốc diệt nấm (như mancozeb, myclobutanil), trồng giống kháng bệnh."
    },
    "Apple___healthy": {
        "name": "Apple Healthy",
        "description": "Lá táo khỏe mạnh, không có dấu hiệu bệnh lý.",
        "treatment": "Không có bệnh, tiếp tục chăm sóc tốt, bón phân hợp lý và tưới nước đầy đủ."
    },
    "Pepper,_bell___healthy": {
        "name": "Bell Pepper Healthy",
        "description": "Lá ớt chuông khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ để phát hiện bệnh sớm."
    },
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "description": "Bệnh đốm vi khuẩn trên ớt chuông, gây ra các đốm nâu, lõm trên lá và quả, có thể làm rụng lá.",
        "treatment": "Loại bỏ lá bệnh, phun thuốc gốc đồng, luân canh cây trồng, sử dụng giống kháng bệnh."
    },
    "Blueberry___healthy": {
        "name": "Blueberry Healthy",
        "description": "Lá việt quất khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, tưới nước và bón phân hợp lý."
    },
    "Cherry_(including_sour)___healthy": {
        "name": "Cherry Healthy",
        "description": "Lá anh đào khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Corn_(maize)___Common_rust_": {
        "name": "Corn Common Rust",
        "description": "Bệnh gỉ sắt thông thường trên ngô, xuất hiện các đốm màu nâu đỏ trên lá, làm giảm năng suất.",
        "treatment": "Sử dụng giống kháng bệnh, phun thuốc trừ nấm (như mancozeb, azoxystrobin) khi phát hiện bệnh."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Corn Northern Leaf Blight",
        "description": "Bệnh cháy lá phương Bắc trên ngô, gây ra các vết dài màu xám trên lá, làm giảm khả năng quang hợp.",
        "treatment": "Trồng giống kháng bệnh, luân canh cây trồng, phun thuốc trừ nấm khi cần thiết."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "description": "Bệnh đốm xám lá ngô do nấm Cercospora gây ra, xuất hiện các vết xám hình chữ nhật trên lá.",
        "treatment": "Sử dụng giống kháng bệnh, phun thuốc trừ nấm, vệ sinh tàn dư thực vật."
    },
    "Peach___healthy": {
        "name": "Peach Healthy",
        "description": "Lá đào khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": "Bệnh sớm trên khoai tây do nấm Alternaria solani, gây ra các đốm tròn nâu trên lá, có quầng vàng.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như mancozeb, chlorothalonil), luân canh cây trồng."
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "description": "Bệnh mốc sương trên khoai tây do nấm Phytophthora infestans, gây thối lá, thân và củ.",
        "treatment": "Phun thuốc trừ nấm (như metalaxyl, cymoxanil), tiêu hủy cây bệnh, trồng giống kháng bệnh."
    },
    "Raspberry___healthy": {
        "name": "Raspberry Healthy",
        "description": "Lá mâm xôi khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Soybean___healthy": {
        "name": "Soybean Healthy",
        "description": "Lá đậu tương khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Squash___Powdery_mildew": {
        "name": "Squash Powdery Mildew",
        "description": "Bệnh phấn trắng trên bí, xuất hiện lớp phấn trắng trên bề mặt lá, làm lá vàng và khô.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như sulfur, myclobutanil), tăng thông thoáng cho cây."
    },
    "Strawberry___healthy": {
        "name": "Strawberry Healthy",
        "description": "Lá dâu tây khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "description": "Bệnh sớm trên cà chua do nấm Alternaria solani, gây đốm nâu tròn trên lá, thân và quả.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như mancozeb, chlorothalonil), luân canh cây trồng."
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "description": "Bệnh đốm lá Septoria trên cà chua, xuất hiện các đốm nhỏ màu nâu xám trên lá.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc trừ nấm (như mancozeb, copper), vệ sinh vườn."
    },
    "Tomato___healthy": {
        "name": "Tomato Healthy",
        "description": "Lá cà chua khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "description": "Bệnh đốm vi khuẩn trên cà chua, gây đốm nhỏ màu nâu đen trên lá, thân và quả.",
        "treatment": "Loại bỏ lá bệnh, phun thuốc gốc đồng, sử dụng giống kháng bệnh."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "description": "Bệnh mốc sương trên cà chua do nấm Phytophthora infestans, gây thối lá, thân và quả.",
        "treatment": "Phun thuốc trừ nấm (như metalaxyl, cymoxanil), tiêu hủy cây bệnh, trồng giống kháng bệnh."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "Bệnh virus khảm trên cà chua, gây lá biến dạng, vàng, giảm năng suất.",
        "treatment": "Tiêu hủy cây bệnh, vệ sinh dụng cụ, sử dụng giống kháng virus."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": "Bệnh virus vàng xoăn lá trên cà chua, làm lá xoăn, vàng, cây còi cọc.",
        "treatment": "Tiêu hủy cây bệnh, kiểm soát bọ phấn, sử dụng giống kháng virus."
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description": "Bệnh mốc lá trên cà chua, xuất hiện các mảng vàng ở mặt trên lá, mặt dưới có lớp mốc màu ô liu.",
        "treatment": "Cắt bỏ lá bệnh, tăng thông thoáng, phun thuốc trừ nấm (như mancozeb, copper)."
    },
    "Grape___healthy": {
        "name": "Grape Healthy",
        "description": "Lá nho khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc, kiểm tra định kỳ."
    },
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "description": "Bệnh thối đen trên nho do nấm Guignardia bidwellii, gây đốm đen trên lá, quả bị thối đen.",
        "treatment": "Cắt bỏ lá và quả bệnh, phun thuốc trừ nấm (như mancozeb, myclobutanil), vệ sinh vườn."
    }
}
# Đặt thông tin mặc định khi không tìm thấy thông tin bệnh
default_info = {
    "name": "Unknown Disease",
    "description": "Không có thông tin chi tiết về bệnh này.",
    "treatment": "Tham khảo ý kiến chuyên gia nông nghiệp."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = float(prediction[0][predicted_class_index])
    
    # Lấy thông tin bệnh
    info = disease_info.get(predicted_class_name, default_info)
    
    return {
        "class_name": predicted_class_name,
        "disease_name": info["name"],
        "description": info["description"],
        "treatment": info["treatment"],
        "confidence": confidence
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict disease
            result = predict_disease(file_path)
            
            return render_template('result.html', 
                                  filename=filename, 
                                  disease_name=result["disease_name"],
                                  class_name=result["class_name"],
                                  description=result["description"],
                                  treatment=result["treatment"],
                                  )
    
    return render_template('index.html')

@app.route('/analyze_webcam', methods=['POST'])
def analyze_webcam():
    # Đây là phần xử lý ảnh từ webcam, sẽ được implement sau
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Predict disease
        result = predict_disease(file_path)
        
        return jsonify(result)

# @app.route('/analyze_esp32', methods=['POST'])
# def analyze_esp32():
#     # Phần này dành cho ESP32 camera, sẽ được implement sau khi cần
#     pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
