from ultralytics import YOLO
from emotion.load_emotion_model import load_model as load_emotion_model
from gender_race_age.load_gra_model import load_model as load_gra_model

def load_models(cfg, device):
    # Load model
    yolo = YOLO(cfg.YOLO_MODEL)

    emo_model, idx_to_label = load_emotion_model(cfg.EMOTION_MODEL)
    emo_model.to(device).eval()

    gra_model = load_gra_model(cfg.GRA_MODEL)
    gra_model.to(device).eval()

    return yolo, emo_model, gra_model, idx_to_label
