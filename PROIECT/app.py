from io import BytesIO
import random
from utils.encode import encode
from utils.decode import decode
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from flask import Flask, render_template, request, send_file
from scipy.signal import correlate2d

app = Flask("Fourier Watermarking")

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')

@app.route('/apply-watermark', methods=['POST'])
def apply_watermark():
    # Get the uploaded image and watermark files
    image_file = request.files['image']
    watermark_file = request.files['watermark']

    # Save the uploaded files to disk
    image_path = 'image.png'
    watermark_path = 'watermark.png'
    image_file.save(image_path)
    watermark_file.save(watermark_path)

    encode(image_path, watermark_path, 'watermarked_image.png', 5)

    return send_file('watermarked_image.png', mimetype='image/png')

@app.route('/retrieve-watermark', methods=['POST'])
def retrieve_watermark():
    original_file = request.files['original']
    watermarked_file = request.files['watermarked']

    # Save the uploaded files to disk
    original_path = 'original.png'
    watermarked_path = 'watermarked.png'
    original_file.save(original_path)
    watermarked_file.save(watermarked_path)
    
    decode(original_path, watermarked_path, 'res_watermark.png', 5)
    
    return send_file('res_watermark.png', mimetype='image/png')

@app.rout("/rotate", methods=['POST'])
def rotate():
    # Get the uploaded image, rotate by 90 degrees and save it
    image_file = request.files['image']
    image_path = 'image.png'
    image_file.save(image_path)
    image = cv2.imread(image_path)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(image_path, image)

    return send_file(image_path, mimetype='image/png')

@app.route("/flip", methods=['POST'])
def flip():
    # Get the uploaded image, flip it and save it
    image_file = request.files['image']
    image_path = 'image.png'
    image_file.save(image_path)
    image = cv2.imread(image_path)
    image = cv2.flip(image, 1)
    cv2.imwrite(image_path, image)

    return send_file(image_path, mimetype='image/png')

@app.route("/scale", methods=['POST'])
def scale():
    # Get the uploaded image, scale it and save it
    image_file = request.files['image']
    image_path = 'image.png'
    image_file.save(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imwrite(image_path, image)

    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
