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

@app.route('/process_images')
def process_images():
    # Render the HTML template
    return render_template('processing.html')

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

@app.route("/rotate", methods=['POST'])
def rotate():
    # Get the uploaded image, rotate by 90 degrees and save it
    image_file = request.files['image']
    image_path = 'image.png'
    image_file.save(image_path)
    image = cv2.imread(image_path)
    image = cv2.rotate(image, cv2.ROTATE_180)
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
    # Get the uploaded image, scale it, and save it
    image_file = request.files['image']
    image_path = 'image.png'
    image_file.save(image_path)

    # Read the image
    image = cv2.imread(image_path)

    # Get the original dimensions of the image
    height, width = image.shape[:2]

    zoom_factor = 2

    # Calculate the center point of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the new dimensions after zooming
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Calculate the top-left corner of the zoomed-in region
    top_left_x = center_x - (new_width // 2)
    top_left_y = center_y - (new_height // 2)

    # Crop the image to the zoomed-in region
    zoomed_image = image[top_left_y:top_left_y + new_height, top_left_x:top_left_x + new_width]

    # Resize the zoomed-in region to the original size
    zoomed_image = cv2.resize(zoomed_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Save the zoomed-in image
    cv2.imwrite(image_path, zoomed_image)

    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
