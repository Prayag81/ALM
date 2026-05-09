# 📚 Recommended Research Papers for ALM

This list contains the core research papers relevant to the development and understanding of the Audio Language Model (ALM) project, covering CNN design, Transformer integration, and multi-modal learning.

---

### 1. Model Architecture (Backbones)
*   **EnvNet**: *"Deep Design for Environmental Sound Classification"* (2016)
    *   **Context**: The specific "Double Conv" block structure in `utils/audio_encoder.py` is inspired by this architecture.
    *   [arXiv:1608.04363](https://arxiv.org/abs/1608.04363)

*   **PANNs**: *"PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Tagging"* (2020)
    *   **Context**: A foundational paper for the current state of audio classification/tagging using CNNs and independent Sigmoid outputs (Binary Relevance).
    *   [arXiv:1912.10211](https://arxiv.org/abs/1912.10211)

---

### 2. Multi-Label Theory & Loss Functions
*   **Zhang & Zhou**: *"Multilabel neural networks with applications to functional genomics and text categorization"* (2006)
    *   **Context**: The foundational paper for using independent Sigmoid outputs for multi-label tasks instead of a single Softmax.
    *   [IEEE Link](https://ieeexplore.ieee.org/document/1700142)

*   **BCEWithLogitsLoss**: *"The Log-Sum-Exp Trick for Numerical Stability"*
    *   **Context**: Explains the mathematical reason why your project uses the "WithLogits" version of the loss to prevent gradient explosion.

---

### 3. Modern Transformers for Audio
*   **AST**: *"AST: Audio Spectrogram Transformer"* (2021)
    *   **Context**: Essential reading for understanding your multi-label model's Transformer stage and self-attention on audio features.
    *   [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)

---

### 4. Multi-Modal & Embedding Design
*   **CLIP**: *"Learning Transferable Visual Models From Natural Language Supervision"* (2021)
    *   **Context**: Explains the "Projection Head" and L2-Normalization logic used to create unit-sphere audio embeddings.
    *   [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

*   **CLAP**: *"CLAP: Learning Audio Concepts from Natural Language Supervision"* (2023)
    *   **Context**: The ultimate framework for ALM—connecting audio encoders to text encoders.
    *   [arXiv:2206.04769](https://arxiv.org/abs/2206.04769)

---

### 5. Visualization & Classification Metrics
*   **Van der Maaten & Hinton**: *"Visualizing Data using t-SNE"* (2008)
    *   **Context**: The primary paper behind the cluster-maps you generate to see if "Dogs" and "Sirens" are grouping together properly in 2D space.
    *   [JMLR Link](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

*   **F1-Score Micro/Macro**: *"A systematic analysis of multi-label evaluation measures"*
    *   **Context**: Explains the difference between instance-based (Micro) and class-based (Macro) evaluation used in your training logs.

---

### 6. Training Augmentation & Robustness
*   **Mixup**: *"mixup: Beyond Empirical Risk Minimization"* (2018)
    *   **Context**: Explains why blending two audio clips (as you do in `audio_dataset.py`) makes the model significantly more robust to noise and "overlap" scenarios.
    *   [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)

*   **SpecAugment**: *"SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"* (2019)
    *   **Context**: The mathematical basis for the frequency and time masking used in your `audio_dataset.py` to prevent overfitting.
    *   [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)
