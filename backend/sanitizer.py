import os
import re

import bleach
from markupsafe import escape

# Allowed characters for text input
ALLOWED_TEXT_CHARS = re.compile(r'^[a-zA-Z0-9\s.,!?;:()"\'-]+$')

# Maximum lengths for various inputs
MAX_TEXT_LENGTH = 1000
MAX_FILENAME_LENGTH = 255
MAX_ALIGNMENT_LENGTH = 20

def sanitize_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Sanitize text input by removing/escaping dangerous characters
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
    Returns:
        Sanitized text string
    Raises:
        ValueError: If text is invalid or too long
    """
    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    # Limit length
    if len(text) > max_length:
        raise ValueError(f"Text too long. Maximum {max_length} characters allowed.")

    # Remove null bytes and control characters except common whitespace
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\t\n\r')

    # Use bleach to clean HTML/XSS
    text = bleach.clean(text, tags=[], attributes={}, strip=True)

    # Further restrict to safe characters only
    if not ALLOWED_TEXT_CHARS.match(text):
        # Remove any remaining dangerous characters
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)

    return text.strip()

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and other attacks
    Args:
        filename: Input filename
    Returns:
        Sanitized filename
    Raises:
        ValueError: If filename is invalid
    """
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"Filename too long. Maximum {MAX_FILENAME_LENGTH} characters allowed.")

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Get just the basename to prevent directory traversal
    filename = os.path.basename(filename)

    # Remove potentially dangerous characters
    filename = re.sub(r'[<>:"|?*\\]', '', filename)

    # Ensure it doesn't start with . or -
    filename = filename.lstrip('.-')

    if not filename or filename in ['', '.', '..']:
        raise ValueError("Invalid filename")

    # Must be alphanumeric with limited special chars
    if not re.match(r'^[a-zA-Z0-9_.-]+$', filename):
        raise ValueError("Filename contains invalid characters")

    return filename

def sanitize_alignment(alignment: str) -> str:
    """
    Sanitize alignment parameter
    Args:
        alignment: Input alignment string
    Returns:
        Sanitized alignment string
    Raises:
        ValueError: If alignment is invalid
    """
    if not isinstance(alignment, str):
        raise ValueError("Alignment must be a string")

    if len(alignment) > MAX_ALIGNMENT_LENGTH:
        raise ValueError("Alignment parameter too long")

    # Whitelist valid alignments
    valid_alignments = {
        'center', 'top', 'bottom', 'left', 'right',
        'top-left', 'top-right', 'bottom-left', 'bottom-right'
    }

    alignment = alignment.strip().lower()

    if alignment not in valid_alignments:
        raise ValueError(f"Invalid alignment: {alignment}")

    return alignment

def sanitize_numeric(value: str | int | float,
                    min_val: float | None = None,
                    max_val: float | None = None,
                    data_type: type = float) -> int | float:
    """
    Sanitize numeric input
    Args:
        value: Input value to sanitize
        min_val: Minimum allowed value
        max_val: Maximum allowed value  
        data_type: Target data type (int or float)
    Returns:
        Sanitized numeric value
    Raises:
        ValueError: If value is invalid or out of range
    """
    try:
        if isinstance(value, str):
            value = value.strip()

        # Convert to target type
        if data_type == int:
            result = int(float(value))  # Convert via float to handle "1.0" strings
        else:
            result = float(value)

        # Check bounds
        if min_val is not None and result < min_val:
            raise ValueError(f"Value {result} is below minimum {min_val}")

        if max_val is not None and result > max_val:
            raise ValueError(f"Value {result} is above maximum {max_val}")

        return result

    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric value: {value}") from e

def sanitize_method(method: str) -> str:
    """
    Sanitize method parameter
    Args:
        method: Input method string
    Returns:
        Sanitized method string
    Raises:
        ValueError: If method is invalid
    """
    if not isinstance(method, str):
        raise ValueError("Method must be a string")

    # Whitelist valid methods
    valid_methods = {'bicubic', 'bilinear', 'nearest'}

    method = method.strip().lower()

    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}")

    return method

def escape_for_html(text: str) -> str:
    """
    Escape text for safe HTML output
    Args:
        text: Input text
    Returns:
        HTML-escaped text
    """
    return escape(text)

def validate_safe_path(file_path: str, allowed_base_dir: str) -> str:
    """
    Validate that a file path is safe and within the allowed directory
    Args:
        file_path: File path to validate
        allowed_base_dir: Base directory that file must be within
    Returns:
        Validated absolute path
    Raises:
        ValueError: If path is unsafe
    """
    if not isinstance(file_path, str):
        raise ValueError("Path must be a string")

    if not file_path:
        raise ValueError("Path cannot be empty")

    # Convert to absolute paths
    abs_file_path = os.path.abspath(file_path)
    abs_base_dir = os.path.abspath(allowed_base_dir)

    # Resolve any symlinks to prevent symlink attacks
    try:
        real_file_path = os.path.realpath(abs_file_path)
        real_base_dir = os.path.realpath(abs_base_dir)
    except OSError:
        raise ValueError("Invalid path")

    # Check if the real file path is within the real base directory
    if not real_file_path.startswith(real_base_dir + os.sep):
        raise ValueError("Path outside allowed directory")

    # Additional checks
    if '..' in file_path:
        raise ValueError("Path traversal not allowed")

    # Check for null bytes
    if '\x00' in file_path:
        raise ValueError("Null bytes not allowed in path")

    # Check if it's actually a file (not a directory or special file)
    if os.path.exists(real_file_path):
        if not os.path.isfile(real_file_path):
            raise ValueError("Path must be a regular file")

        # Check if it's a symlink (additional security)
        if os.path.islink(abs_file_path):
            raise ValueError("Symlinks not allowed")

    return real_file_path
