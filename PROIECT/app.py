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
    # Get the watermarked image and watermark files
    watermarked_file = request.files['watermarked']
    watermark_file = request.files['watermark']

    # Load the watermarked image
    watermarked_img = Image.open(watermarked_file)

    # Load the original watermark image
    watermark_img = Image.open(watermark_file)

    # Convert both images to grayscale
    watermarked_gray = watermarked_img.convert('L')
    watermark_gray = watermark_img.convert('L')

    # Convert the grayscale images to NumPy arrays
    watermarked_arr = np.array(watermarked_gray)
    watermark_arr = np.array(watermark_gray)

    # Apply the 2D discrete Fourier transform to the watermarked image
    watermarked_dft = fftshift(fft2(watermarked_arr))

    # Apply the 2D discrete Fourier transform to the watermark image
    watermark_dft = fftshift(fft2(watermark_arr))

    # Compute the magnitude and phase spectra of the watermarked and watermark images
    watermarked_magnitude = np.abs(watermarked_dft)
    watermarked_phase = np.angle(watermarked_dft)
    watermark_magnitude = np.abs(watermark_dft)
    watermark_phase = np.angle(watermark_dft)

    # Retrieve the watermark by combining the magnitude spectrum of the watermark image
    # with the phase spectrum of the watermarked image
    retrieved_magnitude = watermark_magnitude
    retrieved_phase = watermarked_phase

    # Construct the complex Fourier spectrum using the retrieved magnitude and phase
    retrieved_dft = retrieved_magnitude * np.exp(1j * retrieved_phase)

    # Apply the inverse Fourier transform to retrieve the watermark
    retrieved_watermark_arr = ifft2(ifftshift(retrieved_dft)).real

    # Normalize the retrieved watermark to the range [0, 255]
    retrieved_watermark_norm = (retrieved_watermark_arr - np.min(retrieved_watermark_arr)) / np.ptp(retrieved_watermark_arr) * 255.0

    # Convert the retrieved watermark to an 8-bit unsigned integer array
    retrieved_watermark = np.uint8(retrieved_watermark_norm)

    # Save the retrieved watermark image to a buffer
    buffer = BytesIO()
    Image.fromarray(retrieved_watermark).save(buffer, format='JPEG')

    # Reset the buffer's position to the start
    buffer.seek(0)

    # Return the retrieved watermark image as a response
    return send_file(buffer, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=8000)