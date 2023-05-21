from io import BytesIO
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

    # Load the original image
    img = Image.open(image_file)

    # Convert the image to grayscale
    gray_img = img.convert('L')

    # Convert the grayscale image to a NumPy array
    gray_arr = np.array(gray_img)

    # Apply the 2D discrete Fourier transform to the image
    shiftedDFT = fftshift(fft2(gray_arr))

    # Load the watermark image and resize it to match the original image size
    watermark_img = Image.open(watermark_file)
    watermark_img = watermark_img.resize(gray_img.size)

    # Convert the watermark image to a binary NumPy array
    watermark_arr = np.array(watermark_img.convert('1'))

    # Apply the watermark to the image by multiplying the DFT of the watermark
    # by the DFT of the image
    # watermarked_dft = dft * np.fft.fft2(watermark_arr, s=gray_arr.shape)
    alpha = 0.5
    # # Apply the inverse DFT to the watermarked image
    # watermarked = np.fft.ifft2(watermarked_dft).real
    watermarkedDFT = shiftedDFT + alpha * watermark_arr

    watermarkedImage = ifft2(ifftshift(watermarkedDFT))

    # Convert the watermarked image to an 8-bit unsigned integer array
    watermarked = np.uint8(watermarkedImage)

    # Save the watermarked image to a buffer
    buffer = BytesIO()
    Image.fromarray(watermarked).save(buffer, format='JPEG')

    # Reset the buffer's position to the start
    buffer.seek(0)

    # Return the watermarked image as a response
    return send_file(buffer, mimetype='image/jpeg')


@app.route('/retrieve-watermark', methods=['POST'])
def retrieve_watermark():
    watermarked_image = request.files["watermarked_image"]
    original_image = request.files["original_image"]
    # Convert the images to grayscale
    watermarked_gray = watermarked_image.convert('L')
    original_gray = original_image.convert('L')

    # Convert the grayscale images to NumPy arrays
    watermarked_gray_arr = np.array(watermarked_gray)
    original_gray_arr = np.array(original_gray)

    # Apply the 2D discrete Fourier transform to the images
    watermarked_shiftedDFT = fftshift(fft2(watermarked_gray_arr))
    original_shiftedDFT = fftshift(fft2(original_gray_arr))

    # Calculate the DFT difference between the watermarked and original images
    alpha = 0.5
    watermark_shiftedDFT = (watermarked_shiftedDFT - original_shiftedDFT) / alpha

    # Apply the inverse DFT to the watermark DFT
    watermark_image = ifft2(ifftshift(watermark_shiftedDFT))

    watermark = np.uint8(np.abs(watermark_image))

    # Threshold the watermark to create a binary image
    thresh = 128
    watermark_binary = (watermark > thresh) * 255

    # Convert the binary watermark back to a PIL image
    watermark_img = Image.fromarray(watermarked_binary.astype(np.uint8))

    buffer = BytesIO()
    Image.fromarray(watermark_img).save(buffer, format='JPEG')

    # Reset the buffer's position to the start
    buffer.seek(0)

    # Return the watermarked image as a response
    return send_file(buffer, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)