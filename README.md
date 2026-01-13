# Anamorpher

Anamorpher (named after [anamorphosis](https://en.wikipedia.org/wiki/Anamorphosis)) is a tool for crafting and visualizing image scaling attacks against multi-modal AI systems. It provides a frontend interface and Python API for generating images that only reveal multi-modal prompt injections when downscaled. Refer to ["Weaponizing image scaling against production AI systems"](https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems/) for more information on this attack vector.

Anamorpher is in active beta development. We welcome feedback and contributions!

## Features

- Generate payloads for systems using select implementations of the bicubic, bilinear, and nearest neighbor downscaling algorithms
- Compare payload effectiveness through a frontend interface that includes implementations from OpenCV, PyTorch, TensorFlow, and Pillow
- Include your own custom image downscaling implementation using the modular design of the Python API


<div align="center">
<table>
  <tr>
    <td><img src="image_scaling_figure.png" alt="Comparison showing hidden prompt" width="400"></td>
    <td><img src="gemini-cli-PoC.gif" alt="Demo of attack in action" width="400"></td>
  </tr>
  <tr>
    <td align="center">Frontend Demo</td>
    <td align="center">Gemini CLI Attack Demo</td>
  </tr>
</table>
</div>

## Requirements

- **Python 3.11+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Run backend:
```bash
uv run python backend/app.py
```

3. Open `frontend/index.html` in a web browser

> Windows: Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) due to TensorFlow dependencies.

## Warnings and Known Limitations

-  Due to the probabilistic nature of these systems, results may vary. For consistent evaluation, run each example 5 times.
-  Additional image transformations may interfere with the effectiveness of the injections.
-  Not all payloads will work against each implementation and parameter set of the bicubic and bilinear downscaling algorithms as a result of varying robustness of anti-aliasing across implementations, different default parameters in scaling libraries, and implementation-specific optimizations that affect interpolation behavior.
-  This also holds true of production AI systems more broadly as system scaling behavior is subject to change.


## Maintainers
- [Kikimora Morozova](https://github.com/kiki-morozova)
- [Suha Sabi Hussain](https://github.com/suhacker1)

## References
- [**Weaponizing image scaling against production AI systems**](https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems/)
- [**Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning**](https://www.usenix.org/conference/usenixsecurity20/presentation/quiring)
- [**Seeing is Not Believing: Camouflage Attacks on Image Scaling Algorithms**](https://www.usenix.org/conference/usenixsecurity19/presentation/xiao)
