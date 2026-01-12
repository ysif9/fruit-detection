# Fruit Detection & Classification

A comprehensive fruit recognition system that explores and compares multiple machine learning approaches for image
classification. This project implements both classical machine learning algorithms and deep learning models to classify
10 different types of fruits, providing insights into the strengths and trade-offs of each approach.

## 🎯 Project Overview

This project tackles multi-class fruit image classification using computer vision techniques. It serves as both a
practical classification system and an educational comparison of different modeling approaches, from traditional machine
learning to modern deep learning architectures.

### Key Features

- **10 Fruit Classes**: Apple, Banana, Orange, Mango, Grapes, Pineapple, Watermelon, Pomegranate, Strawberry, Lemon
- **Multiple Model Architectures**: 7 different approaches ranging from classical ML to transfer learning
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrices, and error analysis for each model
- **Production-Ready Pipeline**: Efficient data preprocessing, feature extraction, and model evaluation utilities
- **Hugging Face Integration**: Seamless dataset loading from the cloud

### Use Cases

- **Educational**: Compare classical ML vs. deep learning approaches on the same dataset
- **Benchmarking**: Evaluate trade-offs between model complexity, training time, and accuracy
- **Deployment Planning**: Choose the right model based on your constraints (accuracy, speed, resources)
- **Research**: Baseline implementations for fruit classification or similar image recognition tasks

## 📊 Dataset

- **Source**: [Hugging Face - ysif9/fruit-recognition](https://huggingface.co/datasets/ysif9/fruit-recognition)
- **Image Preprocessing**:
    - Resized to 224×224 pixels for deep learning models
    - Resized to 96×96 pixels for feature extraction (classical ML)
    - RGB color space
    - Normalized to [0, 1] range for neural networks
- **Splits**: Train, Validation, and Test sets
- **Classes**: 10 balanced fruit categories

## 🚀 Setup & Installation

### Prerequisites

- **Python**: 3.10 or higher
- **uv**: Fast Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ysif9/fruit-detection
   cd fruit-detection
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Verify Installation**
   ```bash
   uv run main.py
   ```

### Development Setup (PyCharm)

Follow the official [PyCharm uv integration guide](https://www.jetbrains.com/help/pycharm/uv.html) to configure your
IDE.

### Key Dependencies

- **Deep Learning**: TensorFlow 2.20+, Keras
- **Classical ML**: scikit-learn 1.7+, XGBoost 3.1+
- **Data Processing**: datasets (Hugging Face), NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Computer Vision**: OpenCV

### Using uv for Project Management

| Task                 | pip                                   | uv                          |
|----------------------|---------------------------------------|-----------------------------|
| Install dependencies | `pip install -r requirements.txt`     | `uv sync`                   |
| Add a package        | `pip install package_name`            | `uv add package_name`       |
| Add dev dependency   | `pip install --save-dev package_name` | `uv add --dev package_name` |
| Freeze dependencies  | `pip freeze > requirements.txt`       | `uv lock`                   |
| Run a script         | `python script.py`                    | `uv run script.py`          |

## 🧪 Running the Models

Each model is implemented in a separate Jupyter notebook for easy experimentation:

```bash
# Launch Jupyter Lab
uv run jupyter lab

# Or run individual notebooks
uv run jupyter notebook 1_SVM.ipynb
uv run jupyter notebook 2_KNN.ipynb
# ... etc
```

## 📈 Model Summaries & Comparisons

This project implements 7 different classification approaches, divided into two categories:

### Classical Machine Learning Models (Feature-Based)

These models use **MobileNetV2** (pre-trained on ImageNet) as a feature extractor, converting 96×96 images into
1280-dimensional feature vectors. The classifiers then learn from these high-level features.

#### 1. Support Vector Machine (SVM) - `1_SVM.ipynb`

**Architecture**:

- Feature extraction: MobileNetV2 (frozen, pre-trained on ImageNet)
- Dimensionality reduction: PCA (1280 → 100 dimensions) or LDA (1280 → 9 dimensions)
- Classifier: RBF kernel SVM with class balancing

**Performance**:

- **Accuracy**: 87.71% (PCA + RBF SVM variant)
- **Training Time**: ~2 minutes
- **F1-Score**: 0.87 (macro average)

**Strengths**:

- Highest accuracy among all models tested
- Excellent performance on distinctive fruits (Strawberry: 97% precision, Pomegranate: 93%)
- Robust to overfitting with proper regularization
- Works well with high-dimensional features

**Limitations**:

- Struggles with similar-colored fruits (Orange ↔ Lemon: 162 confusions)
- Slower inference compared to simpler models
- Requires careful hyperparameter tuning (C, gamma)

**Best For**: Production deployments where accuracy is paramount and inference time is acceptable

#### 2. K-Nearest Neighbors (KNN) - `2_KNN.ipynb`

**Architecture**:

- Feature extraction: MobileNetV2 (frozen)
- Classifier: KNN with k=5 neighbors, Euclidean distance

**Performance**:

- **Accuracy**: 80.74%
- **Training Time**: Instant (lazy learning)
- **F1-Score**: 0.81 (macro average)

**Strengths**:

- No training phase (lazy learning)
- Excellent on distinctive shapes (Watermelon: 94% precision, Banana: 93%)
- Simple and interpretable
- No assumptions about data distribution

**Limitations**:

- High confusion on similar fruits (Orange ↔ Lemon: 165 total errors)
- Slow inference (must compare to all training samples)
- Sensitive to feature scaling and distance metric choice
- Memory-intensive (stores all training data)

**Best For**: Quick prototyping, small datasets, or when interpretability is crucial

#### 3. Decision Tree - `3_DecisionTree.ipynb`

**Architecture**:

- Feature extraction: MobileNetV2 (frozen)
- Classifier: Single decision tree (default parameters)

**Performance**:

- **Accuracy**: 59.96%
- **Training Time**: Fast (~30 seconds)
- **F1-Score**: 0.59 (macro average)

**Strengths**:

- Fast training and inference
- Highly interpretable (can visualize decision rules)
- Handles non-linear relationships
- No feature scaling required

**Limitations**:

- Severe overfitting on high-dimensional features (1280 dimensions)
- Poor generalization (worst performer)
- All classes show weak performance (<75% precision)
- Creates overly specific splits that don't generalize

**Best For**: Educational purposes to demonstrate overfitting; not recommended for production

#### 4. Random Forest - `4_RandomForest.ipynb`

**Architecture**:

- Feature extraction: MobileNetV2 (frozen)
- Classifier: Ensemble of 200 decision trees
- Hyperparameter tuning: GridSearchCV (3-fold CV)

**Performance**:

- **Accuracy**: 82.02%
- **Training Time**: ~133 seconds (with GridSearch)
- **F1-Score**: 0.82 (weighted average)
- **Best Parameters**: n_estimators=200, max_depth=None, min_samples_split=2

**Strengths**:

- Strong improvement over single decision tree (82% vs 60%)
- Excellent on distinctive fruits (Strawberry: 93% precision, Watermelon: 91%)
- Robust to overfitting through ensemble averaging
- Provides feature importance rankings

**Limitations**:

- Class imbalance bias (Lemon → Orange: 139 errors, but reverse only 43)
- Lemon has worst recall (56%) due to bias toward Orange predictions
- Slower inference than single tree
- Less interpretable than single decision tree

**Best For**: Balanced accuracy and speed; good general-purpose classifier

#### 5. XGBoost - `5_XGBoost.ipynb`

**Architecture**:

- Feature extraction: MobileNetV2 (frozen)
- Classifier: Gradient boosted trees (100 estimators, max_depth=5)
- Optimization: Sequential boosting with regularization

**Performance**:

- **Accuracy**: 83.75%
- **Training Time**: Moderate (~2-3 minutes)
- **F1-Score**: 0.84 (weighted average)

**Strengths**:

- Second-best accuracy among classical ML models
- Balanced error patterns (symmetric Lemon ↔ Orange confusion)
- Consistent performance across classes (8/10 classes >75% F1)
- Better handling of class imbalance than Random Forest
- Built-in regularization prevents overfitting

**Limitations**:

- Still struggles with citrus fruits (Lemon ↔ Orange: 161 total errors)
- Requires careful hyperparameter tuning
- Longer training time than Random Forest
- More complex to interpret

**Best For**: When you need strong performance with balanced predictions across classes

### Deep Learning Models (End-to-End)

These models learn features directly from raw pixels using convolutional neural networks.

#### 6. Simple CNN - `simple_cnn.ipynb`

**Architecture**:

- Input: 250×250×3 RGB images
- Conv2D(20 filters, 3×3) → MaxPooling(2×2)
- Conv2D(40 filters, 3×3) → Flatten
- Dense(100, ReLU) → Dense(10, Softmax)
- Total parameters: ~2.5M

**Performance**:

- **Accuracy**: ~65-70% (estimated from architecture)
- **Training Time**: Fast (5 epochs with early stopping)
- **Epochs**: 5 with early stopping

**Strengths**:

- Learns features end-to-end (no pre-trained model needed)
- Lightweight architecture (fast training and inference)
- Good starting point for understanding CNNs
- Low memory footprint

**Limitations**:

- Limited depth (only 2 conv layers)
- Lower accuracy than transfer learning approaches
- Requires more data to train from scratch
- May underfit complex patterns

**Best For**: Educational purposes, resource-constrained environments, or when transfer learning isn't applicable

#### 7. MobileNetV2 Transfer Learning - `FruitClassifierCNN2.ipynb`

**Architecture**:

- Input: 224×224×3 RGB images with rescaling (1/255)
- Data augmentation: Random horizontal flip, rotation (±10%)
- Base: MobileNetV2 (frozen, pre-trained on ImageNet, pooling='avg')
- Custom head: BatchNorm → Dropout(0.35) → Dense(220, ReLU) → Dense(60, ReLU) → Dense(10, Softmax)
- Total parameters: ~3.5M (only head is trainable)

**Performance**:

- **Accuracy**: ~92-95% (estimated from architecture and typical transfer learning results)
- **Training Time**: Moderate (up to 100 epochs with early stopping, patience=10)
- **Batch Size**: 64

**Strengths**:

- Leverages ImageNet pre-training (1000 classes, 1.2M images)
- Data augmentation improves generalization
- Batch normalization stabilizes training
- Dropout prevents overfitting
- Expected to be the best performer (if trained properly)

**Limitations**:

- Requires more computational resources (GPU recommended)
- Longer training time than classical ML
- Less interpretable than classical models
- Needs careful tuning of augmentation and regularization

**Best For**: Maximum accuracy when computational resources are available; production systems with GPU inference

## 📊 Performance Comparison Table

| Model                      | Accuracy   | Training Time | Inference Speed | Best Use Case              |
|----------------------------|------------|---------------|-----------------|----------------------------|
| **SVM (PCA+RBF)**          | **87.71%** | ~2 min        | Medium          | Production (best accuracy) |
| **XGBoost**                | 83.75%     | ~2-3 min      | Medium          | Balanced performance       |
| **Random Forest**          | 82.02%     | ~2 min        | Medium          | General purpose            |
| **KNN**                    | 80.74%     | Instant       | Slow            | Prototyping                |
| **Decision Tree**          | 59.96%     | ~30 sec       | Fast            | Educational only           |
| **Simple CNN**             | ~65-70%*   | Fast          | Fast            | Learning/lightweight       |
| **MobileNetV2 (Transfer)** | ~92-95%*   | Moderate      | Medium          | Maximum accuracy           |

*Estimated based on architecture; actual results may vary

## 🔍 Key Insights

### Common Challenges Across All Models

1. **Citrus Confusion**: Orange ↔ Lemon is the most common error across all models (150-200 confusions)
2. **Red Fruit Similarity**: Apple ↔ Pomegranate confusion (100-150 errors) due to similar color
3. **Color Dominance**: Models rely heavily on color features, struggling with same-color fruits

### Best Performing Classes (Across Models)

- **Strawberry**: 93-97% precision (distinctive red color + texture)
- **Watermelon**: 90-94% precision (unique green exterior + size)
- **Banana**: 85-93% precision (distinctive yellow curve)

### Most Challenging Classes

- **Lemon**: 56-77% recall (frequently confused with Orange)
- **Mango**: 60-78% precision (confused with Orange and other yellow fruits)
- **Apple**: 69-85% recall (confused with Pomegranate)

## 🛠️ Technical Details

### Caching Strategy

The project uses NumPy caching to avoid re-extracting features:

- Features cached in `features_cache/` directory
- Speeds up experimentation with classical ML models
- Automatic cache invalidation on data changes

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Hyperparameter optimization for deep learning models
- Additional architectures (ResNet, EfficientNet, Vision Transformers)
- Class balancing techniques to reduce citrus confusion
- Model deployment examples (Flask API, TensorFlow Lite)
- Additional datasets or fruit classes

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
