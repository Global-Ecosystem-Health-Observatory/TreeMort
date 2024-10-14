import os


def get_checkpoint(output_dir, model_weights="best", model_type="student", teacher_model_name=None):
    checkpoint = None
    
    if model_type == "teacher":
        if teacher_model_name:
            checkpoint = os.path.join(output_dir, teacher_model_name)
            if not os.path.exists(checkpoint):
                checkpoint = None
        return checkpoint

    if model_weights == "best":
        checkpoint = os.path.join(output_dir, "best.weights.pth")
        if not os.path.exists(checkpoint):
            checkpoint = None

    elif model_weights == "latest":
        checkpoints = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints,
                key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
            )
            checkpoint = os.path.join(output_dir, latest_checkpoint)
        else:
            checkpoint = None

    return checkpoint