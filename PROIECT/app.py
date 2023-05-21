from io import BytesIO
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from flask import Flask, render_template, request, send_file

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
    watermarked_img = Image.open(request.files['watermarked_image'])
    original_watermark = np.array(Image.open(request.files['watermark']).resize(watermarked_img.size).convert('1')).astype(int)

    # Convert the watermarked image to a grayscale NumPy array
    watermarked_arr = np.array(watermarked_img.convert('L'))

    # Apply the 2D discrete Fourier transform to the watermarked image
    shifted_dft = np.fft.fftshift(np.fft.fft2(watermarked_arr))

    # Extract the watermark from the watermarked image by multiplying the shifted DFT of
    # the watermarked image by the complex conjugate of the shifted DFT of the original
    # watermark, and then applying the inverse Fourier transform to the result
    extracted_dft = shifted_dft * np.conj(np.fft.fft2(original_watermark, s=watermarked_arr.shape))
    extracted_img = np.fft.ifft2(np.fft.ifftshift(extracted_dft)).real

    # Convert the extracted watermark to a binary NumPy array by applying a threshold
    extracted_watermark = np.uint8(extracted_img > np.mean(extracted_img))[::-1]

    # Compare the extracted watermark with the original watermark to see if they match
    if np.array_equal(original_watermark, extracted_watermark):
        return 'Watermark detected!'
    else:
        return 'Watermark not detected.'

if __name__ == '__main__':
    app.run(debug=True, port=8000)