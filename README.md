# Anamorpher

A proof-of-concept tool for exploring image downscaling algorithms and generating adversarial payloads that exploit image scaling vulnerabilities in multimodal AI systems. Anamorpher demonstrates how carefully crafted images can survive downscaling operations and inject malicious content into AI model inputs.

## Background

Image scaling attacks exploit low sampling rates in mathematical interpolation algorithms to embed adversarial content that becomes visible after an image is downscaled. This technique has been demonstrated in several academic works:

- [**Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning**](https://www.usenix.org/conference/usenixsecurity20/presentation/quiring) (USENIX Security 2020)
- [**Seeing is Not Believing: Camouflage Attacks on Image Scaling Algorithms**](https://www.usenix.org/conference/usenixsecurity19/presentation/xiao) (USENIX Security 2019) 
- [**Adversarial Examples for Semantic Image Segmentation and Object Detection**](https://arxiv.org/abs/2003.08633) (arXiv 2020)

These attacks are particularly concerning in the context of [multimodal prompt injection](https://developer.nvidia.com/blog/how-hackers-exploit-ais-problem-solving-instincts/), where adversarial images can manipulate AI system behavior. Traditionally, such attacks were used for model backdoors, evasion, and poisoning, but by embedding text with Anamorpher, we show that they are viable vectors for prompt injection.

## Demonstration

![Anamorpher Demo](gemini-cli-PoC.gif)

The above demonstration shows how scaling exploits can use prompt injection to achieve data exfiltration on production systems, like gemini CLI. Note that AI systems which do not show the user a preview of the inputted image, are particularly vulnerable because the user cannot see what the model sees.

## Technical Capabilities

Anamorpher includes:
- Payload generators for certain implementations of the bicubic, bilinear, and nearest neighbor downscaling algorithms
- Web-based testing interface for comparing payload effectiveness across implementations, supporting target frameworks OpenCV, PyTorch, TensorFlow, and Pillow
- Modular design such that when Anamorpher is run locally, it may be extended to any image downscaling implementation and algorithm

## Requirements

- **Python 3.11** (required - torch==2.1.0 does not support Python 3.13)
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
This project can also be run with the `uv` python package and project manager.

## Limitations
The bicubic and bilinear payload generators are not effective against every implementation and parameter set. This limitation exists due to:

- Varying robustness of anti-aliasing across implementations
- Different default parameters in scaling libraries
- Implementation-specific optimizations that affect interpolation behavior

While we intend to develop payload generators for additional implementations, scaling algorithms change frequently. We cannot guarantee working proof-of-concepts for every popular implementation at all times.