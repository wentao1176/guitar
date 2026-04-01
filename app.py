import base64
import logging
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from core.detector import GuitarFingeringRecognizer
from api.chords import chords_bp
import config

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO if not config.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- 初始化应用 ----------
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
CORS(app)

# 注册蓝图
app.register_blueprint(chords_bp, url_prefix='/api/chords')

# ---------- 模型文件检查 ----------
def check_model_files():
    missing = []
    if not os.path.exists(config.YOLO_MODEL_PATH):
        missing.append(config.YOLO_MODEL_PATH)
    if not os.path.exists(config.HAND_MODEL_PATH):
        missing.append(config.HAND_MODEL_PATH)
    if missing:
        logger.error(f"模型文件缺失: {missing}")
        raise FileNotFoundError(f"模型文件缺失: {missing}")
    logger.info("所有模型文件已找到")

check_model_files()

# ---------- 识别器初始化 ----------
recognizer = GuitarFingeringRecognizer(
    yolo_model_path=config.YOLO_MODEL_PATH,
    hand_model_path=config.HAND_MODEL_PATH
)

# 全局演奏记录
play_records = []
records_lock = threading.Lock()
MAX_RECORDS = 100

# ---------- 工具函数 ----------
def base64_to_cv2(image_base64: str):
    """将 base64 图片数据转换为 OpenCV 图像 (BGR 格式)"""
    try:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[-1]
        image_base64 = image_base64.strip()
        img_bytes = base64.b64decode(image_base64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("cv2.imdecode 返回 None，图片可能损坏")
        return frame
    except Exception as e:
        logger.error(f"Base64 转图片失败: {e}")
        return None

# ---------- HTTP 接口 ----------
@app.route('/api/solo/save_record', methods=['POST'])
def save_record():
    """保存演奏记录"""
    global play_records
    try:
        data = request.get_json()
        new_record = data.get('record', [])
        with records_lock:
            play_records = new_record
            if len(play_records) > MAX_RECORDS:
                play_records = play_records[-MAX_RECORDS:]
        logger.info(f"记录保存成功，当前条数: {len(play_records)}")
        return jsonify({'status': 'success', 'message': '记录保存成功'})
    except Exception as e:
        logger.exception("保存记录失败")
        return jsonify({'status': 'failed', 'message': '保存失败，请稍后重试'}), 500

@app.route('/api/process_landmarks', methods=['POST'])
def process_landmarks():
    """处理手部关键点（替代 SocketIO）"""
    try:
        data = request.get_json()
        landmarks = data.get('landmarks')
        timestamp = data.get('timestamp', time.time())
        img_width = data.get('img_width', 1280)
        img_height = data.get('img_height', 720)

        if not landmarks or len(landmarks) != 21:
            return jsonify({'status': 'failed', 'error': '关键点数据无效'}), 400

        result = recognizer.process_landmarks(landmarks, timestamp, img_width, img_height)
        return jsonify(result)
    except Exception as e:
        logger.exception("处理关键点时发生异常")
        return jsonify({'status': 'failed', 'error': '处理失败'}), 500

@app.route('/api/update_fretboard', methods=['POST'])
def update_fretboard():
    """更新指板参数（替代 SocketIO）"""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'status': 'failed', 'error': '图像数据为空'}), 400

        frame = base64_to_cv2(image_base64)
        if frame is None:
            return jsonify({'status': 'failed', 'error': '图片解码失败'}), 400

        success = recognizer.update_fretboard(frame)
        return jsonify({'status': 'success', 'success': success})
    except Exception as e:
        logger.exception("更新指板参数时发生异常")
        return jsonify({'status': 'failed', 'error': '处理失败'}), 500

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """处理图像帧（替代 SocketIO）"""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'status': 'failed', 'error': '图像数据为空'}), 400

        frame = base64_to_cv2(image_base64)
        if frame is None:
            return jsonify({'status': 'failed', 'error': '图片解码失败'}), 400

        timestamp = time.time()
        processed_frame, result = recognizer.process_frame(frame, timestamp)
        return jsonify(result)
    except Exception as e:
        logger.exception("处理帧时发生异常")
        return jsonify({'status': 'failed', 'error': '处理失败，请稍后重试'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "message": "吉他AI服务运行正常",
        "models": {
            "yolo": os.path.exists(config.YOLO_MODEL_PATH),
            "hand": os.path.exists(config.HAND_MODEL_PATH)
        }
    })

# ---------- 页面路由 ----------
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/tuning')
def tuning():
    return render_template('tuning.html')

@app.route('/solo')
def solo():
    return render_template('solo.html')

@app.route('/teach')
def teach():
    return render_template('teach.html')

# ---------- 启动 ----------
if __name__ == '__main__':
    logger.info(f"启动服务器，debug={config.DEBUG}")
    app.run(
        debug=config.DEBUG,
        host='0.0.0.0',
        port=5000
    ) 