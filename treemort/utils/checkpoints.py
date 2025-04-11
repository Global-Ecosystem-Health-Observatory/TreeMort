import os


def get_checkpoint(output_dir, model_name="flair_unet", model_file_name="best.weights.pth", model_weights="best"):
    if model_weights == "best":
        checkpoint = os.path.join(output_dir, model_name, model_file_name)
        if not os.path.exists(checkpoint):
            checkpoint = None

    elif model_weights == "latest":
        checkpoint_dir = os.path.join(output_dir, model_name)
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints,
                key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
            )
            checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        else:
            checkpoint = None

    return checkpoint