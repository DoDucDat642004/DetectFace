from collections import Counter
from pathlib import Path
import pandas as pd

def export_people_excel(summary, output_dir="output"):
    """
    Xuất báo cáo thống kê file Excel (.xlsx).

    1. Các sheet phân loại (Emotion, Gender...): Bảng đếm số lượng và % tỉ lệ.
    2. Sheet Confidence: Điểm số độ tin cậy.
    3. Sheet Timeline: Dữ liệu theo thời gian.

    Args:
        summary (dict): Dictionary chứa dữ liệu tổng hợp.
        output_dir (str): Đường dẫn thư mục lưu file.
    """
    
    # Thiết lập đường dẫn và thư mục
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    excel_path = output_dir / "people_summary.xlsx"

    # context manager (with) tự động lưu và đóng file sau khi ghi xong
    # engine="xlsxwriter": Engine ghi Excel.
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:

        # XỬ LÝ DỮ LIỆU PHÂN LOẠI
        # Duyệt qua các trường thông tin
        categories = ["emotion", "gender", "race", "age"]
        
        for key in categories:
            # Lấy danh sách giá trị
            values = summary.get(key, [])
            
            if not values: continue

            # Đếm tần suất xuất hiện của từng giá trị ({'Male': 10, 'Female': 5})
            counter = Counter(values)
            
            # Chuyển đổi sang DataFrame
            df = pd.DataFrame(
                counter.items(), 
                columns=[key.capitalize(), "Count"]
            ).sort_values("Count", ascending=False)
            
            # Tính phần trăm tỉ lệ (%)
            # (Số lượng / Tổng số lượng) * 100
            df["Percent (%)"] = (df["Count"] / df["Count"].sum()) * 100
            
            df["Percent (%)"] = df["Percent (%)"].round(2)

            # index=False: Không ghi cột số thứ tự (0, 1, 2...) vào file
            df.to_excel(writer, sheet_name=key.capitalize(), index=False)

        # XỬ LÝ DỮ LIỆU SỐ (CONFIDENCE SCORES)
        if summary.get("emotion_conf"):
            df_conf = pd.DataFrame({
                "Emotion_confidence": summary["emotion_conf"]
            })
            # Ghi dữ liệu
            df_conf.to_excel(
                writer, 
                sheet_name="Emotion_confidence", 
                index=False
            )

        ''' Danh sách các dictionary ([{
                "frame": 0,
                "time": 0.0,
                "person_id": pid,
                "emotion": emo,
                "confidence": emo_conf,
            }, ...])
        '''
        if summary.get("timeline"):
            df_timeline = pd.DataFrame(summary["timeline"])
            df_timeline.to_excel(
                writer, 
                sheet_name="Timeline", 
                index=False
            )

    print(f"Excel saved: {excel_path.resolve()}")
    return excel_path