from .packages import *
from .emotion_model import EmotionResNet18_SE

# label_to_idx = {
#     'angry': 0, 
#     'disgust': 1, 
#     'fear': 2, 
#     'happy': 3, 
#     'neutral': 4, 
#     'sad': 5, 
#     'surprise': 6, 
#     'contempt': 7
# }

def load_model(path_emotion_model = "./model/best_emotion_resnet18_se.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(path_emotion_model):
        checkpoint = torch.load(path_emotion_model, map_location=device)
        label_to_idx = checkpoint["label_to_idx"]
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        emotion_model = EmotionResNet18_SE(num_classes=len(idx_to_label))

        # REMOVE "module." prefix if trained with DataParallel
        new_state_dict = {}
        state_dict = checkpoint["model_state"]
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k.replace("module.", "")] = v
            else:
                new_state_dict[k] = v

        emotion_model.load_state_dict(new_state_dict, strict=True)
        emotion_model.to(device)
        emotion_model.float()
        emotion_model.eval()

        # Freeze BatchNorm hoàn toàn để chắc chắn
        # for m in emotion_model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.track_running_stats = False
        #         m.running_mean = None
        #         m.running_var = None

        print(f"[INFO] Emotion model loaded on {device}")
        return emotion_model, idx_to_label

    print("[ERROR] Model checkpoint not found.")
    return None, None
