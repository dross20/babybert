<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/4565f1bd-942e-48ce-b31d-df127c1ff04a">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/484d38b7-9f15-4d76-9c51-460df00deab9">
    <img src="" width="750px" style="height: auto;"></img>
  </picture>
</p>

---
Minimal implementation of the [BERT architecture proposed by Devlin et al.](https://arxiv.org/pdf/1810.04805) using the PyTorch library. This implementation focuses on simplicity and readability, so the model code is not optimized for inference or training efficiency. BabyBERT can be fine-tuned for downstream tasks such as named-entity recognition (NER), sentiment classification, or question answering (QA).

See the [roadmap](#%EF%B8%8F-roadmap) below for my future plans for this library!

## 📦 Installation

```bash
pip install git+https://github.com/dross20/babybert
```

## 🗺️ Roadmap

### Model Implementation
- [ ] Build initial model implementation
- [ ] Write trainer class
- [ ] Create custom WordPiece tokenizer
- [ ] Introduce more parameter configurations
- [ ] Set up pretrained model checkpoints

### Usage Examples
- [ ] Sentiment classification
- [ ] Named entity recognition
- [ ] Question answering
