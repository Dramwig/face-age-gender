from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from face_detection import process_video
import time
import tensorflow as tf
import json
import threading

# 配置TensorFlow
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用TensorFlow日志

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 使用GPU 4
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# 确保上传和输出目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 存储处理进度的字典
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_thread(input_path, output_path, timestamp):
    try:
        process_video(input_path, output_path, processing_status, timestamp)
        processing_status[timestamp]['status'] = 'completed'
        processing_status[timestamp]['progress'] = 100
        processing_status[timestamp]['download_url'] = f'/download/{os.path.basename(output_path)}'
    except Exception as e:
        processing_status[timestamp]['status'] = 'error'
        processing_status[timestamp]['message'] = str(e)
    finally:
        # 清理上传的原始文件
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}.{filename}")
        output_path = os.path.join('outputs', f"{timestamp}.{filename}")
        
        file.save(input_path)
        
        # 初始化处理状态
        processing_status[timestamp] = {
            'status': 'processing',
            'progress': 0,
            'message': '开始处理视频...'
        }
        
        # 在后台线程中处理视频
        thread = threading.Thread(
            target=process_video_thread,
            args=(input_path, output_path, timestamp)
        )
        thread.start()
        
        # 立即返回timestamp
        return jsonify({
            'success': True,
            'message': '开始处理视频',
            'timestamp': timestamp
        })
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/status/<timestamp>')
def get_status(timestamp):
    timestamp = int(timestamp)
    if timestamp in processing_status:
        return jsonify(processing_status[timestamp])
    return jsonify({'error': '找不到处理状态'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join('outputs', filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True, port=5003) 