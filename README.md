<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://imgur.com/ORrR7Ci.png">
    <source media="(prefers-color-scheme: light)" srcset="https://imgur.com/a59Qpu8.png">
    <img src="" width="750px" style="height: auto;"></img>
  </picture>
</p>

<div align="center">
  
  <a href="https://www.python.org/">![Static Badge](https://img.shields.io/badge/python-3.12-orange)</a>
  <a href="https://github.com/dross20/babybert/blob/main/LICENSE">![GitHub license](https://img.shields.io/badge/license-MIT-yellow.svg)</a>
  <a href="https://pytorch.org/">![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch)</a>
  <a href="https://github.com/astral-sh/ruff">![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)</a>
  
</div>

---

Minimal implementation of the [BERT architecture proposed by Devlin et al.](https://arxiv.org/pdf/1810.04805) using the PyTorch library. This implementation focuses on simplicity and readability, so the model code is not optimized for inference or training efficiency. BabyBERT can be fine-tuned for downstream tasks such as named-entity recognition (NER), sentiment classification, or question answering (QA).

See the [roadmap](#%EF%B8%8F-roadmap) below for my future plans for this library!

## 📦 Installation

```bash
pip install git+https://github.com/dross20/babybert
```

## 🗺️ Roadmap

### Model Implementation
- [x] Build initial model implementation
- [ ] Write trainer class
- [ ] Create custom WordPiece tokenizer
- [ ] Introduce more parameter configurations
- [ ] Set up pretrained model checkpoints

### Usage Examples
- [ ] Sentiment classification
- [ ] Named entity recognition
- [ ] Question answering







