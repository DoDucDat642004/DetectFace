from .packages import *
from .emotion_model import EmotionResNet

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

def load_model(path_emotion_model = "./model/best_emotion.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path_emotion_model, map_location=device)
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    if os.path.exists("./emotion/model/model_emotion.pt") == False:
        if os.path.exists(path_emotion_model):

            emotion_model = EmotionResNet(num_classes=len(idx_to_label))

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

            # CPU
            scripted_model = torch.jit.script(emotion_model)
            scripted_model.save("./emotion/model/model_emotion.pt")
        else :
            print("[ERROR] Model checkpoint not found.")
            return None, None
    emotion_model = torch.jit.load("./emotion/model/model_emotion.pt", map_location=device)
    emotion_model.to(device)
    emotion_model.eval()
        
    print(f"[INFO] Emotion model loaded on {device}")
    return emotion_model, idx_to_label

