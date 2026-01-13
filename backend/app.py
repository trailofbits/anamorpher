import base64
import glob
import io
import os
import re
import subprocess
import tempfile

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from sanitizer import (
    escape_for_html,
    sanitize_alignment,
    sanitize_filename,
    sanitize_method,
    sanitize_numeric,
    sanitize_text,
    validate_safe_path,
)

# Configuration
DECOY_IMAGES_DIR = os.getenv('DECOY_IMAGES_DIR', 'adversarial_generators/decoy_images')
BICUBIC_SCRIPT_PATH = os.getenv('BICUBIC_SCRIPT_PATH', 'adversarial_generators/bicubic_gen_payload.py')
NEAREST_SCRIPT_PATH = os.getenv('NEAREST_SCRIPT_PATH', 'adversarial_generators/nearest_gen_payload.py')
BILINEAR_SCRIPT_PATH = os.getenv('BILINEAR_SCRIPT_PATH', 'adversarial_generators/bilinear_gen_payload.py')

from downsamplers import (
    OpenCVDownsampler,
    PillowDownsampler,
    PyTorchDownsampler,
    TensorFlowDownsampler,
)

app = Flask(__name__)
CORS(app)

# Initialize downsamplers
downsamplers = {
    'opencv': OpenCVDownsampler(),
    'pytorch': PyTorchDownsampler(),
    'tensorflow': TensorFlowDownsampler(),
    'pillow': PillowDownsampler()
}

# Supported target resolutions
TARGET_RESOLUTIONS = {
    '1092x1092': (1092, 1092),
    '800x800': (800, 800)
}

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string"""
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy image with validation"""
    if not isinstance(base64_string, str):
        raise ValueError("Image data must be a string")

    if not base64_string:
        raise ValueError("Image data cannot be empty")

    # Limit size (roughly 10MB for base64 data)
    if len(base64_string) > 15_000_000:
        raise ValueError("Image data too large")

    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            if ',' not in base64_string:
                raise ValueError("Invalid data URL format")
            base64_string = base64_string.split(',', 1)[1]

        # Validate base64 format
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_string):
            raise ValueError("Invalid base64 format")

        # Decode base64
        image_bytes = base64.b64decode(base64_string, validate=True)

        # Limit decoded size (roughly 50MB)
        if len(image_bytes) > 50_000_000:
            raise ValueError("Decoded image too large")

        pil_image = Image.open(io.BytesIO(image_bytes))

        # Verify it's actually an image by checking format
        if pil_image.format not in ['PNG', 'JPEG', 'JPG', 'BMP', 'TIFF']:
            raise ValueError("Unsupported image format")

        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        return np.array(pil_image)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to decode image: {str(e)}")

@app.route('/api/downsamplers', methods=['GET'])
def get_downsamplers():
    """Get available downsamplers and their supported methods"""
    result = {}
    for key, downsampler in downsamplers.items():
        result[key] = {
            'name': downsampler.name,
            'methods': downsampler.get_supported_methods()
        }
    return jsonify(result)

@app.route('/api/resolutions', methods=['GET'])
def get_resolutions():
    """Get supported target resolutions"""
    return jsonify(list(TARGET_RESOLUTIONS.keys()))

@app.route('/api/downsample', methods=['POST'])
def downsample_image():
    """Downsample an image using specified method"""
    try:
        data = request.get_json()

        # Extract parameters
        image_data = data.get('image')
        raw_downsampler_type = data.get('downsampler', '')
        raw_method = data.get('method', '')
        raw_target_resolution = data.get('target_resolution')

        # Validate required inputs
        if not all([image_data, raw_downsampler_type, raw_method]) or raw_target_resolution is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Sanitize string parameters
        downsampler_type = str(raw_downsampler_type).strip()[:50]  # Limit length
        method = str(raw_method).strip()[:50]
        target_resolution = int(raw_target_resolution)

        if downsampler_type not in downsamplers:
            return jsonify({'error': escape_for_html(f'Unknown downsampler: {downsampler_type}')}), 400

        # Validate target resolution is reasonable
        if target_resolution < 64 or target_resolution > 2048:
            return jsonify({'error': f'Target resolution must be between 64 and 2048, got {target_resolution}'}), 400

        # Convert base64 to image
        image = base64_to_image(image_data)

        # Validate image dimensions - must be square and divisible by 4
        height, width = image.shape[:2]
        if height != width:
            return jsonify({'error': f'Image must be square, got {width}x{height}'}), 400
        if width % 4 != 0:
            return jsonify({'error': f'Image dimensions must be divisible by 4, got {width}x{height}'}), 400

        # Get downsampler and target size
        downsampler = downsamplers[downsampler_type]
        target_size = (target_resolution, target_resolution)

        # Validate method support
        if method not in downsampler.get_supported_methods():
            return jsonify({'error': escape_for_html(f'{downsampler.name} does not support {method}')}), 400

        # Perform downsampling
        downsampled = downsampler.downsample(image, target_size, method)

        # Convert results to base64
        original_b64 = image_to_base64(image)
        downsampled_b64 = image_to_base64(downsampled)

        return jsonify({
            'original': original_b64,
            'downsampled': downsampled_b64,
            'original_size': f"{width}x{height}",
            'downsampled_size': f"{target_size[0]}x{target_size[1]}",
            'downsampler': downsampler.name,
            'method': method
        })

    except ValueError as e:
        return jsonify({'error': escape_for_html(str(e))}), 400
    except Exception:
        return jsonify({'error': 'Internal server error'}), 500

def create_text_image(text: str, size: int = 1092, font_size: int = 32, alignment: str = 'center') -> tuple[np.ndarray, bool]:
    """Create a square text image with specified text, font size, and alignment. Returns tuple (image, text_overflowed)"""
    # Create image with 1:1 aspect ratio
    image = Image.new('RGB', (size, size), color='#333333')
    draw = ImageDraw.Draw(image)

    # Try to use Arial font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except OSError:
        try:
            # Try common system paths for Arial
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "C:\\Windows\\Fonts\\arial.ttf"  # Windows
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()

    # Calculate margins based on alignment
    margin = 10
    text_area_width = size - 2 * margin
    text_area_height = size - 2 * margin

    # Wrap text to fit in available space
    wrapped_lines = wrap_text_to_fit(text, font, draw, text_area_width)

    # Calculate text dimensions
    line_height = draw.textbbox((0, 0), "Ay", font=font)[3] - draw.textbbox((0, 0), "Ay", font=font)[1]
    total_height = len(wrapped_lines) * line_height

    # Check if text overflows
    text_overflowed = total_height > text_area_height

    # Calculate starting position based on alignment
    if alignment in ['center', 'top', 'bottom']:
        # Vertical alignment
        if alignment == 'center':
            start_y = max(margin, (size - total_height) // 2)
        elif alignment == 'top':
            start_y = margin
        else:  # bottom
            start_y = max(margin, size - margin - total_height)

        # Draw lines centered horizontally
        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break  # Stop if we exceed bottom margin
            y = start_y + i * line_height
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (size - line_width) // 2
            draw.text((x, y), line, font=font, fill='#00b002')

    elif alignment in ['left', 'right']:
        # Horizontal alignment with vertical centering
        start_y = max(margin, (size - total_height) // 2)

        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break  # Stop if we exceed bottom margin
            y = start_y + i * line_height
            if alignment == 'left':
                x = margin
            else:  # right
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = size - margin - line_width
            draw.text((x, y), line, font=font, fill='#00b002')

    else:  # corner alignments
        if alignment.startswith('top'):
            start_y = margin
        else:  # bottom
            start_y = max(margin, size - margin - total_height)

        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break  # Stop if we exceed bottom margin
            y = start_y + i * line_height
            if alignment.endswith('left'):
                x = margin
            else:  # right
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = size - margin - line_width
            draw.text((x, y), line, font=font, fill='#00b002')

    return np.array(image), text_overflowed

def wrap_text_to_fit(text: str, font, draw, max_width: int) -> list:
    """Wrap text to fit within specified width"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Single word is too long, split it
                lines.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def get_decoy_images():
    """Get list of available decoy images"""
    decoy_dir = os.path.abspath(DECOY_IMAGES_DIR)
    pattern = os.path.join(decoy_dir, '*_*.png')
    files = glob.glob(pattern)

    decoys = []
    for file_path in files:
        try:
            # Validate path is safe and within decoy directory
            safe_path = validate_safe_path(file_path, decoy_dir)
            filename = os.path.basename(safe_path)

            # Validate filename for security
            if not re.match(r'^[a-zA-Z0-9_.-]+\.png$', filename):
                continue  # Skip suspicious filenames

            # Extract width from filename (format: WIDTH_IN_PIXELS_*.png)
            width_str = filename.split('_')[0]
            width = int(width_str)

            # Validate width is reasonable
            if width < 64 or width > 8192:
                continue

            decoys.append({
                'filename': escape_for_html(filename),  # Escape for output
                'path': safe_path,  # Use validated safe path
                'width': width,
                'height': width  # Assume square images
            })
        except (ValueError, IndexError):
            # Skip files that don't match the expected format or are unsafe
            continue

    return decoys

@app.route('/api/decoy-images', methods=['GET'])
def get_available_decoy_images():
    """Get available decoy images"""
    try:
        decoys = get_decoy_images()
        return jsonify(decoys)
    except Exception:
        return jsonify({'error': 'Failed to load decoy images'}), 500

@app.route('/api/demo-images', methods=['GET'])
def get_demo_images():
    """Get available demo images for downsampling"""
    try:
        demo_dir = os.path.abspath('demo_images')
        if not os.path.exists(demo_dir):
            return jsonify([])

        pattern = os.path.join(demo_dir, '*.png')
        files = glob.glob(pattern)

        demos = []
        for file_path in files:
            try:
                # Validate path is safe and within demo directory
                safe_path = validate_safe_path(file_path, demo_dir)
                filename = os.path.basename(safe_path)

                # Validate filename for security
                if not re.match(r'^[a-zA-Z0-9_.-]+\.png$', filename):
                    continue  # Skip suspicious filenames

                # Get actual image dimensions
                with Image.open(safe_path) as img:
                    width, height = img.size

                # Validate dimensions are reasonable and square
                if width != height or width < 64 or width > 8192 or width % 4 != 0:
                    continue

                demos.append({
                    'filename': escape_for_html(filename),
                    'path': safe_path,
                    'width': width,
                    'height': height
                })
            except (OSError, ValueError, IndexError):
                # Skip files that don't match the expected format or are unsafe
                continue

        return jsonify(demos)
    except Exception:
        return jsonify({'error': 'Failed to load demo images'}), 500

@app.route('/api/demo-images/<filename>', methods=['GET'])
def get_demo_image(filename):
    """Get a specific demo image as base64"""
    try:
        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        demo_dir = os.path.abspath('demo_images')
        file_path = os.path.join(demo_dir, safe_filename)

        # Validate path is safe and within demo directory
        safe_path = validate_safe_path(file_path, demo_dir)

        if not os.path.exists(safe_path):
            return jsonify({'error': 'Demo image not found'}), 404

        # Load and convert to base64
        image = np.array(Image.open(safe_path))
        image_b64 = image_to_base64(image)

        return jsonify({
            'filename': escape_for_html(safe_filename),
            'image': image_b64,
            'width': image.shape[1],
            'height': image.shape[0]
        })

    except Exception:
        return jsonify({'error': 'Failed to load demo image'}), 500

@app.route('/api/generate-text-image', methods=['POST'])
def generate_text_image():
    """Generate a text image with specified text, font size, and alignment"""
    try:
        data = request.get_json()

        # Extract and sanitize parameters
        raw_text = data.get('text', 'Sample Text')
        raw_size = data.get('size', 1092)
        raw_font_size = data.get('font_size', 32)
        raw_alignment = data.get('alignment', 'center')

        # Sanitize inputs
        text = sanitize_text(raw_text)
        size = sanitize_numeric(raw_size, min_val=64, max_val=2048, data_type=int)
        font_size = sanitize_numeric(raw_font_size, min_val=20, max_val=64, data_type=int)
        alignment = sanitize_alignment(raw_alignment)

        # Generate text image
        text_image, text_overflowed = create_text_image(text, size, font_size, alignment)
        text_image_b64 = image_to_base64(text_image)

        return jsonify({
            'image': text_image_b64,
            'size': f"{size}x{size}",
            'text': escape_for_html(text),  # Escape for safe output
            'font_size': font_size,
            'alignment': alignment,
            'text_overflowed': text_overflowed
        })

    except ValueError as e:
        return jsonify({'error': escape_for_html(str(e))}), 400
    except Exception:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/generate-adversarial', methods=['POST'])
def generate_adversarial():
    """Generate adversarial image using bicubic, nearest neighbor, or bilinear method"""
    try:
        data = request.get_json()

        # Extract and sanitize parameters
        raw_method = data.get('method', 'bicubic')
        raw_text = data.get('text', 'Sample Text')
        raw_decoy_filename = data.get('decoy_filename')
        raw_font_size = data.get('font_size', 32)
        raw_alignment = data.get('alignment', 'center')

        # Sanitize basic parameters
        method = sanitize_method(raw_method)
        text = sanitize_text(raw_text)
        font_size = sanitize_numeric(raw_font_size, min_val=20, max_val=64, data_type=int)
        alignment = sanitize_alignment(raw_alignment)

        if not raw_decoy_filename:
            return jsonify({'error': 'Decoy filename is required'}), 400

        decoy_filename = sanitize_filename(raw_decoy_filename)

        # Method-specific parameters with sanitization
        if method == 'bilinear':
            lam = sanitize_numeric(data.get('lam', 1.0), min_val=0.0, max_val=10.0)
            eps = sanitize_numeric(data.get('eps', 0.0), min_val=0.0, max_val=1.0)
            gamma_target = sanitize_numeric(data.get('gamma', 0.9), min_val=0.1, max_val=3.0)
            dark_frac = sanitize_numeric(data.get('dark_frac', 0.3), min_val=0.0, max_val=1.0)
        elif method == 'bicubic':
            lam = sanitize_numeric(data.get('lam', 0.25), min_val=0.0, max_val=10.0)
            eps = sanitize_numeric(data.get('eps', 0.0), min_val=0.0, max_val=1.0)
            gamma_target = sanitize_numeric(data.get('gamma', 1.0), min_val=0.1, max_val=3.0)
            dark_frac = sanitize_numeric(data.get('dark_frac', 0.3), min_val=0.0, max_val=1.0)
        else:  # nearest
            lam = sanitize_numeric(data.get('lam', 0.25), min_val=0.0, max_val=10.0)
            eps = sanitize_numeric(data.get('eps', 0.0), min_val=0.0, max_val=1.0)
            gamma_target = sanitize_numeric(data.get('gamma', 1.0), min_val=0.1, max_val=3.0)
            dark_frac = 0.3

        # Additional parameters for nearest neighbor
        offset = sanitize_numeric(data.get('offset', 2), min_val=0, max_val=10, data_type=int) if method == 'nearest' else 2

        # Find decoy image
        decoys = get_decoy_images()
        decoy_info = None
        for decoy in decoys:
            if decoy['filename'] == decoy_filename:
                decoy_info = decoy
                break

        if not decoy_info:
            return jsonify({'error': escape_for_html(f'Decoy image {decoy_filename} not found')}), 400

        # Check if decoy is 4x the target size
        target_size = decoy_info['width'] // 4
        if target_size <= 0 or decoy_info['width'] % 4 != 0:
            return jsonify({'error': 'Decoy image width must be divisible by 4'}), 400

        # Generate target text image with user-specified font size and alignment
        target_image, _ = create_text_image(text, target_size, font_size, alignment)

        # Create temporary files for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save target image
            target_path = os.path.join(temp_dir, 'target.png')
            Image.fromarray(target_image).save(target_path)

            # Copy decoy image to temp directory
            decoy_path = os.path.join(temp_dir, 'decoy.png')
            import shutil
            shutil.copy2(decoy_info['path'], decoy_path)

            # Choose the appropriate script with path validation
            base_script_dir = os.path.abspath('adversarial_generators')

            if method == 'bicubic':
                script_path = validate_safe_path(BICUBIC_SCRIPT_PATH, base_script_dir)
                cmd = [
                    'python3', script_path,
                    '--decoy', decoy_path,
                    '--target', target_path,
                    '--lam', str(lam),
                    '--eps', str(eps),
                    '--gamma', str(gamma_target),
                    '--dark-frac', str(dark_frac)
                ]
            elif method == 'bilinear':
                script_path = validate_safe_path(BILINEAR_SCRIPT_PATH, base_script_dir)
                cmd = [
                    'python3', script_path,
                    '--decoy', decoy_path,
                    '--target', target_path,
                    '--lam', str(lam),
                    '--eps', str(eps),
                    '--gamma', str(gamma_target),
                    '--dark-frac', str(dark_frac)
                ]
            else:  # nearest
                script_path = validate_safe_path(NEAREST_SCRIPT_PATH, base_script_dir)
                cmd = [
                    'python3', script_path,
                    '--decoy', decoy_path,
                    '--target', target_path,
                    '--lam', str(lam),
                    '--eps', str(eps),
                    '--gamma', str(gamma_target),
                    '--offset', str(offset)
                ]

            # Run the script in the temp directory
            result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)

            if result.returncode != 0:
                return jsonify({
                    'error': escape_for_html(f'Script execution failed: {result.stderr[:500]}')  # Limit error length
                }), 500

            # Find the generated adversarial image
            if method == 'nearest':
                adv_files = glob.glob(os.path.join(temp_dir, 'advNN*.png'))
            elif method == 'bilinear':
                adv_files = glob.glob(os.path.join(temp_dir, 'adv_bilinear*.png'))
            else:  # bicubic
                adv_files = glob.glob(os.path.join(temp_dir, 'adv*.png'))

            if not adv_files:
                return jsonify({'error': 'No adversarial image was generated'}), 500

            # Load and convert the adversarial image
            adv_path = adv_files[0]
            # Validate adversarial image path is within temp directory
            safe_adv_path = validate_safe_path(adv_path, temp_dir)
            adv_image = np.array(Image.open(safe_adv_path))
            adv_image_b64 = image_to_base64(adv_image)

            # Also load target and original decoy for comparison
            target_image_b64 = image_to_base64(target_image)
            # decoy_info['path'] is already validated from get_decoy_images()
            decoy_image = np.array(Image.open(decoy_info['path']))
            decoy_image_b64 = image_to_base64(decoy_image)

            return jsonify({
                'adversarial_image': adv_image_b64,
                'target_image': target_image_b64,
                'original_decoy': decoy_image_b64,
                'method': method,
                'parameters': {
                    'lam': lam,
                    'eps': eps,
                    'gamma': gamma_target,
                    'offset': offset if method == 'nearest' else None,
                    'dark_frac': dark_frac if method in ['bicubic', 'bilinear'] else None
                },
                'text': escape_for_html(text),
                'decoy_filename': escape_for_html(decoy_filename),
                'script_output': result.stdout
            })

    except ValueError as e:
        return jsonify({'error': escape_for_html(str(e))}), 400
    except Exception:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Anamorpher backend...")
    print("Available downsamplers:", list(downsamplers.keys()))
    print("Supported resolutions:", list(TARGET_RESOLUTIONS.keys()))
    app.run(debug=True, host='0.0.0.0', port=5000)
