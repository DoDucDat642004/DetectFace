from .packages import *
from .gra_model import MultiTaskFaceModel

def load_model(path_gra_model = "./model/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists("./gender_race_age/model/model_gra.pt") == False:
        if os.path.exists(path_gra_model):

            gra_model = MultiTaskFaceModel()

            checkpoint = torch.load(path_gra_model, map_location=device)
            gra_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            gra_model.float()
            gra_model.eval()
            gra_model.to(device)

            # CPU
            scripted_model = torch.jit.script(gra_model)
            scripted_model.save("./gender_race_age/model/model_gra.pt")
        else :
            print("[ERROR] Model checkpoint not found.")
            return None, None
    
    gra_model = torch.jit.load("./gender_race_age/model/model_gra.pt", map_location=device)
    gra_model.to(device)
    gra_model.eval()

    print(f"[INFO] Emotion model loaded on {device}")
    return gra_model

