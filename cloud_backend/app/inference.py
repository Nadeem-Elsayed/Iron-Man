import os
from typing import List, Dict, Any, Union
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Internal model imports
try:
    from models.vision_3d import JesterSequenceLSTM
    from models.vision_2d import HaGRIDResNet
except ImportError:
    from cloud_backend.app.models.vision_3d import JesterSequenceLSTM
    from cloud_backend.app.models.vision_2d import HaGRIDResNet


# Standard ImageNet normalization matching training
transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_inference_device():
    """
    Selects the best available accelerator:
    1. DirectML (Windows discrete GPU)
    2. CUDA (Linux/AWS GPU)
    3. CPU fallback
    """
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            target_idx = 0
            for i in range(torch_directml.device_count()):
                name = torch_directml.device_name(i)
                if "7800" in name or "XT" in name:
                    target_idx = i
            return torch_directml.device(target_idx)
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def find_weights_path(candidates: List[str]) -> str:
    """Finds the first existing weights file from a list of candidate paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
        # Check relative to base_dir
        rel_base = os.path.abspath(os.path.join(base_dir, path))
        if os.path.exists(rel_base):
            return rel_base
        # Check relative to repo_root
        rel_root = os.path.abspath(os.path.join(repo_root, path))
        if os.path.exists(rel_root):
            return rel_root
    return ""


class GestureInferenceEngine:
    """
    Production-ready gesture inference engine.
    Supports both 3D dynamic sequences (Jester) and 2D static poses (HaGRID).
    Includes confidence thresholding to prevent false triggers.
    """

    # Jester 4-class mapping
    LABELS_3D = [
        "swipe_left",         # 0
        "swipe_right",        # 1
        "no_gesture",         # 2 (Control - Still)
        "doing_other_things"  # 3 (Control - Moving)
    ]

    # HaGRID 5-class mapping (Alphabetical order from datasets.ImageFolder: hand_heart, mute, no_gesture, take_picture, thumb_index2)
    LABELS_2D = [
        "heart",              # 0 (hand_heart)
        "mute",               # 1 (mute)
        "no_gesture",         # 2 (no_gesture)
        "take_photo",         # 3 (take_picture)
        "thumb_index2"        # 4 (thumb_index2)
    ]

    def __init__(
        self,
        weights_3d_path: str = None,
        weights_2d_path: str = None,
        device = None,
        sequence_length: int = 15,
        default_confidence_threshold_3d: float = 0.70,
        default_confidence_threshold_2d: float = 0.85
    ):
        self.device = device or get_inference_device()
        self.sequence_length = sequence_length
        self.confidence_threshold_3d = default_confidence_threshold_3d
        self.confidence_threshold_2d = default_confidence_threshold_2d

        # 1. Load 3D Dynamic Model
        self.model_3d = None
        target_3d_weights = weights_3d_path or find_weights_path([
            "../weights/jester_lstm_spatial_tuned.pth",
            "cloud_backend/weights/jester_lstm_spatial_tuned.pth",
            "../weights/jester_lstm_spatial.pth",
            "cloud_backend/weights/jester_lstm_spatial.pth",
            "../weights/jester_lstm_finetuned.pth",
            "cloud_backend/weights/jester_lstm_finetuned.pth",
            "../weights/jester_lstm_epoch3.pth",
            "cloud_backend/weights/jester_lstm_epoch3.pth",
            "../weights/jester_lstm.pth",
            "cloud_backend/weights/jester_lstm.pth"
        ])

        if target_3d_weights and os.path.exists(target_3d_weights):
            print(f"[InferenceEngine] Loading 3D weights: {target_3d_weights}")
            state = torch.load(target_3d_weights, map_location=self.device, weights_only=False)
            in_channels = state['lstm.weight'].shape[1] if 'lstm.weight' in state else 2048
            spatial_zones = in_channels // 2048 if in_channels >= 2048 else 1
            print(f"[InferenceEngine] Detected spatial zones: {spatial_zones} (in_channels={in_channels})")
            self.model_3d = JesterSequenceLSTM(
                num_classes=4,
                pool_mode='flatten',
                sequence_length=self.sequence_length,
                dropout=0.0,
                spatial_zones=spatial_zones
            ).to(self.device)
            self.model_3d.load_state_dict(state, strict=True)
            self.model_3d.eval()
            print(f"[InferenceEngine] 3D Model ready on {self.device}.")
        else:
            print("[InferenceEngine] Warning: 3D weights not found. 3D inference disabled.")

        # 2. Load 2D Static Model
        self.model_2d = None
        target_2d_weights = weights_2d_path or find_weights_path([
            "weights/hagrid_resnet50_finetuned.pth",
            "./weights/hagrid_resnet50_finetuned.pth",
            "../weights/hagrid_resnet50_finetuned.pth",
            "cloud_backend/weights/hagrid_resnet50_finetuned.pth",
            "../weights/hagrid_resnet50.pth",
            "cloud_backend/weights/hagrid_resnet50.pth"
        ])

        if target_2d_weights and os.path.exists(target_2d_weights):
            print(f"[InferenceEngine] Loading 2D weights: {target_2d_weights}")
            self.model_2d = HaGRIDResNet(num_classes=5).to(self.device)
            state = torch.load(target_2d_weights, map_location=self.device, weights_only=False)
            self.model_2d.load_state_dict(state, strict=True)
            self.model_2d.eval()
            print(f"[InferenceEngine] 2D Model ready on {self.device}.")

    def _to_pil(self, frame: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Converts BGR numpy frame or RGB image to PIL Image."""
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            # Check if frame is BGR (OpenCV) or RGB
            if frame.ndim == 3 and frame.shape[2] == 3:
                # Default to OpenCV BGR -> RGB
                return Image.fromarray(frame[:, :, ::-1])
            return Image.fromarray(frame)
        raise ValueError("Unsupported frame format. Must be PIL.Image or np.ndarray.")

    def predict_3d(
        self,
        frames: List[Union[np.ndarray, Image.Image]],
        confidence_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Runs inference on a 15-frame rolling buffer.
        Returns predicted gesture, confidence, class probabilities, and an actionable flag.
        """
        if self.model_3d is None:
            raise RuntimeError("3D model is not loaded. Provide valid weights_3d_path.")

        if len(frames) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {len(frames)}.")

        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold_3d

        # Preprocess each frame
        tensor_frames = [transform_2d(self._to_pil(f)) for f in frames]
        # Stack into [T, C, H, W] -> permute to [C, T, H, W] -> add batch dim [1, C, T, H, W]
        sequence_tensor = torch.stack(tensor_frames).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model_3d(sequence_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        pred_idx = int(pred_idx.item())
        confidence = float(conf.item())
        gesture_name = self.LABELS_3D[pred_idx]

        # Actionable filter: only fire when confidence passes threshold AND is an active swipe
        is_actionable = (gesture_name in ["swipe_left", "swipe_right"]) and (confidence >= threshold)

        return {
            "gesture": gesture_name,
            "confidence": round(confidence, 4),
            "is_actionable": is_actionable,
            "probabilities": {name: round(float(probs[i].item()), 4) for i, name in enumerate(self.LABELS_3D)}
        }

    def predict_2d(
        self,
        frame: Union[np.ndarray, Image.Image],
        confidence_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Runs inference on a single static frame (HaGRID).
        """
        if self.model_2d is None:
            raise RuntimeError("2D model is not loaded. Provide valid weights_2d_path.")

        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold_2d

        img_tensor = transform_2d(self._to_pil(frame)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model_2d(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        pred_idx = int(pred_idx.item())
        confidence = float(conf.item())
        gesture_name = self.LABELS_2D[pred_idx]

        is_actionable = (gesture_name != "no_gesture") and (confidence >= threshold)

        return {
            "gesture": gesture_name,
            "confidence": round(confidence, 4),
            "is_actionable": is_actionable,
            "probabilities": {name: round(float(probs[i].item()), 4) for i, name in enumerate(self.LABELS_2D)}
        }

