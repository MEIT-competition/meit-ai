"""
YAMNet transfer learning for horn / siren / crash / normal classification.

Extracts embeddings, trains a Dense head on top of frozen YAMNet, evaluates,
and exports the deployable SavedModel + head_model.keras + logits/labels for
the other scripts (calibration, threshold_search, error_analysis).

fold 0-2 -> train, fold 3 -> val, fold 4 -> test.
"""
import numpy as np
import tensorflow as tf

from common import (
    BASE_DIR, CLASSES, CLASS_WEIGHTS, SAVED_MODEL_PATH, YAMNET_HANDLE,
    extract_embeddings_for_rows, load_manifest, load_yamnet, pool_clip_logits,
)


def main():
    rows = load_manifest()
    if not rows:
        print("manifest.csv is empty - run prepare_data.py first (with raw/ data populated)")
        return

    yamnet_model = load_yamnet()

    train_rows = [r for r in rows if r["fold"] in (0, 1, 2)]
    val_rows = [r for r in rows if r["fold"] == 3]
    test_rows = [r for r in rows if r["fold"] == 4]
    print(f"clips: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    print("Extracting embeddings (cached after first run)...")
    train_emb, train_labels, _, _ = extract_embeddings_for_rows(train_rows, yamnet_model, "train")
    val_emb, val_labels, _, val_clip_id = extract_embeddings_for_rows(val_rows, yamnet_model, "val")
    test_emb, test_labels, test_fp, test_clip_id = extract_embeddings_for_rows(test_rows, yamnet_model, "test")

    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name="input_embedding"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), name="logits"),
    ], name="danger_sound_head")

    head.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    history = head.fit(
        train_emb, train_labels,
        validation_data=(val_emb, val_labels),
        epochs=50,
        batch_size=32,
        shuffle=True,
        class_weight=CLASS_WEIGHTS,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True)],
        verbose=2,
    )

    test_logits = head.predict(test_emb, verbose=0)
    test_loss, test_acc = head.evaluate(test_emb, test_labels, verbose=0)
    print(f"test loss (frame-level)={test_loss:.4f} acc (frame-level)={test_acc:.4f}")

    # clip-level report - this is what actually matters, since the deployed
    # model pools a clip's frames into one decision before the threshold is applied
    from sklearn.metrics import classification_report
    pooled_logits, pooled_labels, _, _ = pool_clip_logits(test_logits, test_labels, test_clip_id)
    pooled_pred = np.argmax(pooled_logits, axis=1)
    print(f"test acc (clip-level) = {(pooled_pred == pooled_labels).mean():.4f}")
    print(classification_report(pooled_labels, pooled_pred, target_names=CLASSES, digits=3))

    val_logits = head.predict(val_emb, verbose=0)
    np.save(BASE_DIR / "val_logits.npy", val_logits)
    np.save(BASE_DIR / "val_labels.npy", val_labels)
    np.save(BASE_DIR / "val_clip_id.npy", val_clip_id)
    np.save(BASE_DIR / "test_logits.npy", test_logits)
    np.save(BASE_DIR / "test_labels.npy", test_labels)
    np.save(BASE_DIR / "test_filepaths.npy", test_fp)
    np.save(BASE_DIR / "test_clip_id.npy", test_clip_id)

    head.save(BASE_DIR / "head_model.keras")
    print(f"Saved head model to {BASE_DIR / 'head_model.keras'}")

    # Export deployable SavedModel: raw waveform in, class scores out.
    # Plain tf.Module instead of the Keras Functional API - hub.KerasLayer
    # doesn't accept a KerasTensor under Keras 3, so wiring this up as a
    # tf.keras.Model(Input(...), ...) breaks.
    class ServingModel(tf.Module):
        def __init__(self, yamnet_model, head_model):
            super().__init__()
            self.yamnet_model = yamnet_model
            self.head_model = head_model

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def __call__(self, audio):
            _, embeddings, _ = self.yamnet_model(audio)
            logits = self.head_model(embeddings)
            # take the frame with the single highest peak logit, not the mean -
            # short impulsive sounds (crash) only light up 1-2 of ~8 frames,
            # so averaging with the mostly-silent rest washes the signal out
            best_frame = tf.argmax(tf.reduce_max(logits, axis=1))
            return logits[best_frame]

    serving_model = ServingModel(yamnet_model, head)
    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(serving_model, str(SAVED_MODEL_PATH))
    print(f"Saved deployable model to {SAVED_MODEL_PATH}")
    print("\nNext: run calibration.py, then threshold_search.py, then error_analysis.py")


if __name__ == "__main__":
    main()
