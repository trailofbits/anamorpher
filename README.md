# Anamorpher

Anamorpher (named after [anamorphosis](https://en.wikipedia.org/wiki/Anamorphosis)) is a tool for crafting and visualizing image scaling attacks against multi-modal AI systems. It provides a frontend interface and Python API for generating  images that only reveal multi-modal prompt injections when downscaled. 

## Demonstration

![Anamorpher Demo](gemini-cli-PoC.gif)

This demonstration shows a prompt injection stealthily delivered by an Anamorpher-generated image on the Gemini CLI to exfiltrate user data. Note that many systems like the one shown do not show the user a preview of the downscaled image, making this attack particularly impactful. 

## Features
- Generate payloads for systems using select implementations of the bicubic, bilinear, and nearest neighbor downscaling algorithms
- Compare payload effectiveness through a frontend interface that includes implementations from OpenCV, PyTorch, TensorFlow, and Pillow
- Include your own custom image downscaling implementation using the modular design of the Python API

Anamorpher is currently a prototype. We welcome any suggestions and feedback you may have! 

## Requirements

- **Python 3.11** 
- Virtual environment recommended

## Setup

1. Create and activate virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run backend:
```bash
cd backend
python3 app.py
```

4. Open frontend:
```bash
cd frontend
# Open frontend/index.html in web browser
```
This project can also be run with `uv`.

## Warnings and Known Limitations

- Due to the probabilistic nature of multi-modal AI systems, results may vary. For consistent evaluation, run each example 5 times.
- Not all payloads will work against each implementation and parameter set of the bicubic and bilinear downscaling algorithms as a result of varying robustness of anti-aliasing across implementations, different default parameters in scaling libraries, and implementation-specific optimizations that affect interpolation behavior.
- This also holds true of production AI systems more broadly as system scaling behavior is subject to change.

## Maintainers
- [Kikimora Morozova](https://github.com/kiki-morozova)
- [Suha Sabi Hussain](https://github.com/suhacker1)

## References
- [**Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning**](https://www.usenix.org/conference/usenixsecurity20/presentation/quiring)
- [**Seeing is Not Believing: Camouflage Attacks on Image Scaling Algorithms**](https://www.usenix.org/conference/usenixsecurity19/presentation/xiao)
