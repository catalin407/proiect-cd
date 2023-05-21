from io import BytesIO
import random

import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from flask import Flask, render_template, request, send_file
from scipy.signal import correlate2d

app = Flask("pula mea")

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

    img = cv2.imread(image_path)
    img_f = np.fft.fft2(img)
    height, width, channel = np.shape(img)
    watermark = cv2.imread(watermark_path)
    wm_height, wm_width = watermark.shape[0], watermark.shape[1]
    x = list(range(height // 2))
    y = list(range(width))
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    alpha = 5

    for i in range(height // 2):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = watermark[x[i]][y[j]]
                tmp[height - 1 - i][width - 1 - j] = tmp[i][j]

    res_f = img_f + alpha * tmp
    res = np.fft.ifft2(res_f)
    res = np.real(res)

    buffer = BytesIO()

    # Convert the res image to a PIL Image
    res_image = Image.fromarray(np.uint8(res))

    # Save the res image to the buffer as JPEG
    res_image.save(buffer, format='PNG')

    # Reset the buffer's position to the start
    buffer.seek(0)

    # Return the buffer as a response
    return send_file(buffer, mimetype='image/png')

@app.route('/retrieve-watermark', methods=['POST'])
def retrieve_watermark():
    original_file = request.files['original']
    watermarked_file = request.files['watermarked']

    # Save the uploaded files to disk
    original_path = 'original.png'
    watermarked_path = 'watermarked.png'
    original_file.save(original_path)
    watermarked_file.save(watermarked_path)

    # Load the images using cv2.imread
    original_image = cv2.imread(original_path)
    watermarked_image = cv2.imread(watermarked_path)

    # Convert the images to grayscale if needed
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)

    ori_f = np.fft.fft2(original_gray)
    img_f = np.fft.fft2(watermarked_gray)
    alpha = 5
    watermark = (ori_f - img_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(watermark.shape)
    height, width = watermark.shape

    random.seed(height + width)
    x = list(range(height // 2))
    y = list(range(width))
    random.shuffle(x)
    random.shuffle(y)

    for i in range(height // 2):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]

    buffer = BytesIO()

    # Convert the res array to a PIL Image
    res_image = Image.fromarray(np.uint8(res))

    # Save the res image to the buffer as PNG
    res_image.save(buffer, format='PNG')

    # Reset the buffer's position to the start
    buffer.seek(0)

    # Return the buffer as a response
    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
