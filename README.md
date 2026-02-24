# AI Learning Bootstrap

This repository is an index of AI/ML learning projects.

Each item links to an independent repository.

Projects must be completed sequentially.

## Technologies Used
<table>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" alt="Python" width="48" style="display:inline-block;"/>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/github/explore/main/topics/numpy/numpy.png" alt="NumPy" width="48" style="display:inline-block;"/>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/github/explore/main/topics/jupyter-notebook/jupyter-notebook.png" alt="Jupyter" width="48" style="display:inline-block;"/>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/github/explore/main/topics/pytorch/pytorch.png" alt="PyTorch" width="48" style="display:inline-block;"/>
    </td>
  </tr>
</table>

---

## Project List

| ID  | Project                                                                                                                                                                                                                                                    | Repo | Status   |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | -------- |
| P01 | **Linear Regression (NumPy)** – Predict continuous targets (e.g., house prices) using gradient descent and visualize the best-fit line/plane; implement cost function, parameter updates, and convergence behavior manually                                | https://github.com/coderanger428/ai-linear-regression | complete |
| P02 | **Logistic Regression (NumPy)** – Predict binary outcomes (e.g., customer churn yes/no) using sigmoid activation and gradient descent; implement classification boundary, probability outputs, and training visualization from scratch                     | https://github.com/coderanger428/ai-logistic-regression | complete |
| P03 | **MLP From Scratch (NumPy)** – Predict student exam scores using multiple mixed features (categorical + numerical); implement fully connected layers, forward/backward propagation, SGD training loop, and training/validation loss visualization manually | https://github.com/coderanger428/ai-mlp-from-scratch | complete |
| P04 | **MNIST Classifier (NumPy)** – Classify handwritten digits (0–9) using a fully connected MLP implemented purely in NumPy; include data normalization, batching, training loop, evaluation, and accuracy/loss visualization                                 | https://github.com/coderanger428/ai-mlp-classifier-numpy | complete |
| P05 | **MNIST Classifier (PyTorch MLP)** – Reimplement MNIST digit classification using PyTorch; focus on tensor operations, automatic differentiation, model definition, training loop, and evaluation pipeline                                                 | https://github.com/coderanger428/ai-mnist-mlp-pytorch | complete |
| P06 | **MNIST CNN (PyTorch)** – Classify handwritten digits using a Convolutional Neural Network; implement convolution layers, pooling, feature extraction, training, evaluation, and performance visualization                                                 | https://github.com/coderanger428/ai-mnist-cnn-pytorch | complete |
| P07 | **Tabular ML Benchmark Project** – Train and compare Linear Regression, Random Forest, and MLP models on the same structured dataset; analyze performance, generalization, and model suitability for tabular data                                          | https://github.com/coderanger428/ai-benchmark-lr-rf-mlp | complete |
| P08 | **Simple Image Classifier** – Classify images (e.g., dogs vs cats) using a CNN in PyTorch; implement dataset loading, augmentation, training, evaluation, and prediction visualization                                                                     | https://github.com/coderanger428/ai-simple-image-classifier | complete |
| P09 | **Tiny Object Detector** – Train a lightweight object detection model to detect simple objects (e.g., cups, books, bottles); implement bounding box prediction, classification heads, training loop, and visualization of detections                       | https://github.com/coderanger428/ai-object-detector-simple | complete |
| P10 | **Object Counter System** – Count objects in images/video using detection + tracking; combine a detector with object tracking to produce real-time counting and annotated video output                                                                     |      | in progress |
| P11 | **Face Recognition System** – Build a face recognition pipeline using embedding models; implement face detection, embedding extraction, similarity matching, identity prediction, and real-time webcam recognition                                         |      |          |
| P12 | **OCR Pipeline** – Extract text from printed document images using preprocessing, segmentation, detection, and recognition stages; visualize detected text regions and reconstructed text output                                                           |      |          |
| P13 | **Tokenizer Project** – Implement word-level and subword-level tokenizers for text corpora; handle vocabulary construction, encoding/decoding, and token frequency analysis                                                                                |      |          |
| P14 | **Text Vectorization System** – Convert text into numerical representations using Bag-of-Words, TF-IDF, and embedding-based representations; visualize feature spaces and similarity structures                                                            |      |          |
| P15 | **Text Classifier** – Build a sentiment/intent classifier using vectorized text representations; implement preprocessing, training, evaluation, and classification visualization                                                                           |      |          |
| P16 | **Embedding Model** – Train simple word embeddings from text corpora; visualize embedding spaces and semantic similarity relationships                                                                                                                     |      |          |
| P17 | **Attention Model** – Implement an attention-only sequence model for text processing; visualize attention weights and sequence dependencies                                                                                                                |      |          |
| P18 | **Mini Transformer** – Implement a small transformer encoder model for text transformation tasks; include tokenization, positional encoding, multi-head attention, feed-forward layers, and evaluation                                                     |      |          |
| P19 | **Tiny GPT** – Implement a small GPT-style generative model for simple dialogue generation; include tokenization, embeddings, positional encoding, multi-head attention, decoder blocks, training, and text generation                                     |      |          |
| P20 | **Real-time AI Authentication System** – Build a face-based login system using webcam input; perform real-time face detection, recognition, identity verification, and optional anti-spoofing logic                                                        |      |          |
| P21 | **Gesture Dialogue System** – Build a multi-modal AI system combining gesture recognition from webcam input with a generative language model; enable gesture-based interaction and AI-generated dialogue responses                                         |      |          |

---

## Conventions

Project repositories use:
```text
ai-project-name
```
Example:
```text
ai-linear-regression
```
---

## Rules

- One active project at a time
- Each project has its own repo
- Each repo must contain a README
