<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github-production-user-asset-6210df.s3.amazonaws.com/73395516/457617114-4565f1bd-942e-48ce-b31d-df127c1ff04a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250730%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250730T150418Z&X-Amz-Expires=300&X-Amz-Signature=517a2f5a399d31633eb100321f8f1938c2a39ef21bc1be7d7af94ccfba41fee6&X-Amz-SignedHeaders=host">
    <source media="(prefers-color-scheme: light)" srcset="https://github-production-user-asset-6210df.s3.amazonaws.com/73395516/457617051-484d38b7-9f15-4d76-9c51-460df00deab9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250730%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250730T150321Z&X-Amz-Expires=300&X-Amz-Signature=e8288dd394089a9fa2687ef99b23dde14530572ae5736d1a7695e5d0d1f9c166&X-Amz-SignedHeaders=host">
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
- [x] Build initial model implementation
- [ ] Write trainer class
- [ ] Create custom WordPiece tokenizer
- [ ] Introduce more parameter configurations
- [ ] Set up pretrained model checkpoints

### Usage Examples
- [ ] Sentiment classification
- [ ] Named entity recognition
- [ ] Question answering
