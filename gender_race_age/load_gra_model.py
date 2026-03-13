from gender_race_age.packages import *
from gender_race_age.device import *
from gender_race_age.gra_model import MultiTaskEfficientNetB0

def load_model(path_gra_model="./gender_race_age/model/best_model.pt"):
    """
    Load model GRA (Gender-Race-Age).
    """
    device = get_device()
    
    # Đường dẫn file model (JIT/TorchScript)
    script_path = "./gender_race_age/model/model_gra.pt"

    if not os.path.exists(script_path):
        if os.path.exists(path_gra_model):
            print(f"Converting checkpoint from {path_gra_model}...")
            
            # Khởi tạo kiến trúc model
            gra_model = MultiTaskEfficientNetB0()

            # Load checkpoint
            checkpoint = torch.load(path_gra_model, map_location=device)
            state_dict = checkpoint["model_state_dict"]


            ignore_prefixes = (
                "criterion_", # Các tham số trong hàm Loss
                "log_var_"    # Tham số Uncertainty Weighting
            )

            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Xử lý lỗi DataParallel
                # Nếu train trên nhiều GPU, tên layer sẽ có dạng "module.features..."
                # Cần xóa "module." để chạy được trên 1 CPU/GPU.
                k_clean = k.replace("module.", "")
                
                # Bỏ qua các key nằm trong danh sách
                if any(k_clean.startswith(prefix) for prefix in ignore_prefixes):
                    continue
                
                filtered_state_dict[k_clean] = v
            
            # LOAD STATE DICT
            gra_model.load_state_dict(filtered_state_dict, strict=False)
            
            gra_model.float()
            gra_model.eval()
            gra_model.to(device)

            os.makedirs("./gender_race_age/model/", exist_ok=True)
            
            # torch.jit.script
            scripted_model = torch.jit.script(gra_model)
            scripted_model.save(script_path)
            print(f"Đã lưu model : {script_path}")
        else :
            print(f"Không tìm thấy file checkpoint {path_gra_model}")
            return None
    
    # LOAD MODEL
    try:
        gra_model = torch.jit.load(script_path, map_location=device)
        gra_model.to(device)
        gra_model.eval()
        print(f"GRA model loaded on {device}")
        return gra_model
    except Exception as e:
        print(f"Failed to load scripted model: {e}")
        if os.path.exists(script_path):
            os.remove(script_path)
        return None