from emotion.packages import *
from emotion.device import *
from emotion.emotion_model import EmotionEfficientNet

def load_model(path_emotion_model="./emotion/model/best_emotion.pth"):

    # Cấu hình thiết bị
    device = get_device()
    
    # Kiểm tra file trọng số
    if not os.path.exists(path_emotion_model):
        print(f"Không tìm thấy file checkpoint tại: {path_emotion_model}")
        return None, None

    # Load checkpoint lấy thông tin nhãn (labels)
    checkpoint = torch.load(path_emotion_model, map_location=device)
    
    # Khôi phục dictionary map từ index sang tên nhãn (VD: 0 -> 'angry')
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Đường dẫn file model (JIT/TorchScript)
    jit_model_path = "./emotion/model/model_emotion.pt"

    if not os.path.exists(jit_model_path):
        
        emotion_model = EmotionEfficientNet(num_classes=len(idx_to_label))


        state_dict = checkpoint["model_state"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k.replace("module.", "")] = v
            else:
                new_state_dict[k] = v

        # Load trọng số
        emotion_model.load_state_dict(new_state_dict, strict=True)
        emotion_model.to(device)
        emotion_model.float()
        emotion_model.eval()

        # Freeze
        # for m in emotion_model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False

        try:
            # Tạo dummy input trace model
            # Input size transforms (1 channel, 112x112)
            dummy_input = torch.randn(1, 1, 112, 112).to(device)
            scripted_model = torch.jit.trace(emotion_model, dummy_input)
            
            # Lưu model đã tối ưu
            os.makedirs(os.path.dirname(jit_model_path), exist_ok=True)
            scripted_model.save(jit_model_path)
            print(f"Đã lưu model : {jit_model_path}")
        except Exception as e:
            print(f"Không thể chuyển sang JIT, sử dụng model thường. Lỗi: {e}")
            return emotion_model, idx_to_label

    # Load model JIT
    emotion_model = torch.jit.load(jit_model_path, map_location=device)
    emotion_model.to(device)
    emotion_model.eval()
        
    print(f"Emotion model loaded on {device}")
    return emotion_model, idx_to_label