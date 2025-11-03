from .packages import *
from .gra_model import MultiTaskFaceModel

def load_model(path_gra_model = "./model/multitask_simple.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gra_model = MultiTaskFaceModel(backbone_type="resnet34").to(device)

    if not os.path.exists(path_gra_model):
        print("No checkpoint found.")
        return gra_model
    
    checkpoint = torch.load(path_gra_model, map_location=device)
    gra_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    gra_model.to(device)
    gra_model.float()
    gra_model.eval()

    return gra_model