from pathlib import Path
import csv

def export_people_csv(summary, output_dir="output"):
    """
    Xuất dữ liệu tóm tắt (summary) file CSV.

    Args:
        summary (dict): Dictionary chứa dữ liệu dạng cột. 
                        Ví dụ: {'Name': ['A', 'B'], 'Age': [20, 25]}
        output_dir (str): Tên thư mục đầu ra (mặc định là "output").

    Returns:
        Path: Đường dẫn tuyệt đối đến file CSV vừa tạo.
    """
    
    # Đường dẫn thư mục
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Định nghĩa đường dẫn cho file CSV
    csv_path = output_dir / "people_summary.csv"

    # Lấy danh sách các keys để làm tiêu đề cột (Header)
    keys = summary.keys()
    
    rows = zip(*summary.values())

    # Ghi file CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Ghi hàng tiêu đề
        writer.writerow(keys)
        
        # Ghi nội dung dữ liệu
        for row in rows:
            writer.writerow(row)

    print(f"CSV saved: {csv_path.resolve()}")
    
    return csv_path