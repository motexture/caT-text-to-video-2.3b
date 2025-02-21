---
license: apache-2.0
---
# caT text to video

Conditionally augmented text-to-video model. Uses pre-trained weights from modelscope text-to-video model, augmented with temporal conditioning transformers to extend generated clips and create a smooth transition between them.

This project was trained at home as a hobby.

## Installation

### Clone the Repository

```bash
git clone https://github.com/motexture/caT-text-to-video-2.3b/
cd caT-text-to-video-2.3b
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python run.py
```

Visit the provided URL in your browser to interact with the interface and start generating videos.
