"""
Eggplant Leaf Disease Prediction Script
========================================
Standalone inference script for the eggplant leaf disease detection model.

CRITICAL: The model's built-in Rescaling layer expects raw pixel values
in the [0, 255] range. Do NOT normalize images to [0, 1] before feeding
them to the model — the model handles preprocessing internally.

Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --image path/to/leaf.jpg --model saved_models/eggplant_final.keras
    python predict.py --image-dir path/to/folder/
"""

import argparse
import pathlib
import sys

import numpy as np
import tensorflow as tf
import keras


# ── Custom classes required for model loading ──

@keras.saving.register_keras_serializable(package="Custom")
class SoftCategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.0, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        num_classes = tf.shape(y_pred)[-1]
        if y_true.shape.rank == 1 or (
            y_true.shape.rank == 2 and y_true.shape[-1] == 1
        ):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_true_soft = tf.one_hot(y_true, depth=num_classes)
        else:
            y_true_soft = tf.cast(y_true, tf.float32)
        if self.label_smoothing > 0:
            y_true_soft = y_true_soft * (
                1.0 - self.label_smoothing
            ) + self.label_smoothing / tf.cast(num_classes, tf.float32)
        cross_entropy = -y_true_soft * tf.math.log(y_pred)
        focal_weight = y_true_soft * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "label_smoothing": self.label_smoothing,
                "epsilon": self.epsilon,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="Custom")
class SoftAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_variable(
            shape=(), initializer="zeros", name="correct"
        )
        self.total = self.add_variable(shape=(), initializer="zeros", name="total")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_class = tf.argmax(y_pred, axis=-1)
        if y_true.shape.rank == 1 or (
            y_true.shape.rank == 2 and y_true.shape[-1] == 1
        ):
            true_class = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
        else:
            true_class = tf.argmax(y_true, axis=-1)
        matches = tf.cast(tf.equal(pred_class, true_class), tf.float32)
        self.correct.assign(self.correct + tf.reduce_sum(matches))
        self.total.assign(self.total + tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.correct / (self.total + 1e-7)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


@keras.saving.register_keras_serializable(package="Custom")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, alpha=1e-7, **kwargs):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))
        decay_step = step - warmup_steps
        decay_total = total_steps - warmup_steps
        cosine_frac = 0.5 * (
            1.0
            + tf.math.cos(
                np.pi
                * tf.minimum(decay_step / tf.maximum(decay_total, 1.0), 1.0)
            )
        )
        cosine_lr = self.alpha + (self.initial_lr - self.alpha) * cosine_frac
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
        }


# ── Constants ──

IMG_SIZE = 256

SUPPORTED_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]

CLASS_NAMES = [
    "Healthy",
    "Insect Pest",
    "Leaf Spot",
    "Mosaic Virus",
    "Small Leaf",
    "White Mold",
    "Wilt",
]


def load_model(model_path: str) -> tf.keras.Model:
    """Load the saved Keras model with all custom objects registered."""
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "SoftCategoricalFocalLoss": SoftCategoricalFocalLoss,
            "SoftAccuracy": SoftAccuracy,
            "WarmupCosineDecay": WarmupCosineDecay,
        },
    )
    print("Model loaded successfully.")
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.

    IMPORTANT: The model has a built-in Rescaling layer that converts
    pixel values from [0, 255] to [-1, 1]. Therefore, we must keep
    pixel values in the [0, 255] range — do NOT divide by 255.

    Returns:
        np.ndarray of shape (1, 256, 256, 3) with pixel values in [0, 255].
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)  # [0, 255] float32
    # Do NOT normalize to [0, 1] — the model handles this internally
    img = tf.expand_dims(img, axis=0)  # Add batch dimension: (1, 256, 256, 3)
    return img.numpy()


def predict_single(model: tf.keras.Model, image_path: str) -> dict:
    """
    Run prediction on a single image.

    Returns:
        dict with 'predicted_class', 'confidence', and 'all_confidences'.
    """
    img = preprocess_image(image_path)
    predictions = model.predict(img, verbose=0)
    probs = predictions[0]

    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_confidences": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }


def predict_with_tta(
    model: tf.keras.Model, image_path: str, runs: int = 7
) -> dict:
    """
    Run prediction with Test-Time Augmentation (TTA) for more robust results.

    Applies mild geometric augmentations and averages predictions across
    multiple runs.
    """
    tta_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom((-0.04, 0.04)),
            tf.keras.layers.RandomTranslation(0.03, 0.03),
        ],
        name="tta_augmentor",
    )

    img = preprocess_image(image_path)
    all_preds = []

    # First run: use the original (unaugmented) image
    all_preds.append(model.predict(img, verbose=0))

    # Remaining runs: apply TTA augmentation
    for _ in range(runs - 1):
        aug_img = tta_aug(img, training=True)
        all_preds.append(model.predict(aug_img, verbose=0))

    avg_probs = np.mean(all_preds, axis=0)[0]
    pred_idx = int(np.argmax(avg_probs))

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(avg_probs[pred_idx]),
        "all_confidences": {
            CLASS_NAMES[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eggplant Leaf Disease Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image leaf.jpg
  python predict.py --image leaf.jpg --model saved_models/eggplant_final.keras
  python predict.py --image-dir dataset/Healthy/ --tta
        """,
    )
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument(
        "--image-dir", type=str, help="Path to a directory of images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/eggplant_final.keras",
        help="Path to the saved Keras model (default: saved_models/eggplant_final.keras)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use Test-Time Augmentation for more robust predictions",
    )
    parser.add_argument(
        "--tta-runs",
        type=int,
        default=7,
        help="Number of TTA runs (default: 7)",
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image-dir")

    model = load_model(args.model)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        img_dir = pathlib.Path(args.image_dir)
        for ext in SUPPORTED_EXTENSIONS:
            image_paths.extend(sorted(str(p) for p in img_dir.glob(ext)))

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    print("=" * 70)

    for path in image_paths:
        if not pathlib.Path(path).exists():
            print(f"  [SKIP] File not found: {path}")
            continue

        if args.tta:
            result = predict_with_tta(model, path, runs=args.tta_runs)
            mode = f"TTA ({args.tta_runs} runs)"
        else:
            result = predict_single(model, path)
            mode = "Single-pass"

        print(f"Image: {path}")
        print(f"Mode:  {mode}")
        print(
            f"Result: {result['predicted_class']} "
            f"({result['confidence']:.2%} confidence)"
        )
        print("All class confidences:")
        for cls, conf in result["all_confidences"].items():
            bar = "█" * int(conf * 40)
            print(f"  {cls:<15} {conf:>7.2%} {bar}")
        print("=" * 70)


if __name__ == "__main__":
    main()
