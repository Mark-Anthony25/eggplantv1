# Eggplant Leaf Disease Detection v1

Transfer Learning with MobileNetV2 + Focal Loss for classifying eggplant leaf diseases into 7 categories:
**Healthy**, **Insect Pest**, **Leaf Spot**, **Mosaic Virus**, **Small Leaf**, **White Mold**, **Wilt**

## Project Structure

```
eggplantv1/
├── eggplant (4).ipynb       # Training notebook (Kaggle)
├── predict.py               # Standalone inference script
├── saved_models/             # Saved Keras models (after training)
│   └── eggplant_final.keras
└── README.md
```

## Critical: Image Preprocessing for Inference

The model has a **built-in `Rescaling` layer** that converts pixel values from `[0, 255]` to `[-1, 1]` (required by MobileNetV2). You must feed **raw pixel values in the `[0, 255]` range** — do **NOT** normalize to `[0, 1]`.

```python
# ❌ WRONG — causes near-random, low-confidence predictions
img = tf.image.resize(img, (256, 256))
img = img / 255.0              # BAD: normalizes to [0, 1]
model.predict(img)             # Model rescales to [-1, -0.99] ≈ all black!

# ✅ CORRECT — high-confidence, accurate predictions
img = tf.image.resize(img, (256, 256))
img = tf.cast(img, tf.float32) # Keep in [0, 255]
model.predict(img)             # Model rescales to [-1, 1] ✓
```

## Running Predictions (VS Code / Terminal)

```bash
# Single image
python predict.py --image path/to/leaf.jpg

# With Test-Time Augmentation for more robust results
python predict.py --image path/to/leaf.jpg --tta

# Predict on an entire folder
python predict.py --image-dir path/to/images/

# Use a custom model path
python predict.py --image path/to/leaf.jpg --model saved_models/eggplant_final.keras
```

## Loading the Model in Your Own Code

```python
import tensorflow as tf
import keras
import numpy as np

# 1. Register custom classes (required for model loading)
@keras.saving.register_keras_serializable(package="Custom")
class SoftCategoricalFocalLoss(keras.losses.Loss):
    # ... (see predict.py for full implementation)
    pass

@keras.saving.register_keras_serializable(package="Custom")
class SoftAccuracy(tf.keras.metrics.Metric):
    # ... (see predict.py for full implementation)
    pass

@keras.saving.register_keras_serializable(package="Custom")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # ... (see predict.py for full implementation)
    pass

# 2. Load the model
model = tf.keras.models.load_model(
    "saved_models/eggplant_final.keras",
    custom_objects={
        "SoftCategoricalFocalLoss": SoftCategoricalFocalLoss,
        "SoftAccuracy": SoftAccuracy,
        "WarmupCosineDecay": WarmupCosineDecay,
    },
)

# 3. Preprocess and predict (keep pixels in [0, 255]!)
img = tf.io.read_file("path/to/leaf.jpg")
img = tf.image.decode_image(img, channels=3, expand_animations=False)
img.set_shape([None, None, 3])
img = tf.image.resize(img, (256, 256))
img = tf.cast(img, tf.float32)          # [0, 255] — do NOT divide by 255
img = tf.expand_dims(img, axis=0)       # (1, 256, 256, 3)

probs = model.predict(img, verbose=0)[0]
class_names = ["Healthy", "Insect Pest", "Leaf Spot", "Mosaic Virus",
               "Small Leaf", "White Mold", "Wilt"]
pred_idx = np.argmax(probs)
print(f"{class_names[pred_idx]}: {probs[pred_idx]:.2%}")
```

## Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Input size**: 256 × 256 × 3
- **Built-in preprocessing**: `Rescaling(1/127.5, offset=-1)` maps [0, 255] → [-1, 1]
- **Pooling**: Dual (GlobalAveragePooling + GlobalMaxPooling) → 2560-dim
- **Head**: Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.4) → Dense(7, softmax)
- **Loss**: Focal Loss (γ=2.0) with label smoothing (0.1)
- **Training**: 2-phase (frozen backbone → fine-tune last 50 layers)