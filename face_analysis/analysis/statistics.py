from collections import Counter
from pathlib import Path
import json
import pandas as pd

from face_analysis.analysis.export_csv import export_people_csv
from face_analysis.analysis.export_excel import export_people_excel


# Danh sách nhãn
EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
GENDER_LABELS = ["Male", "Female"]
RACE_LABELS = ["White", "Black", "Latino", "Asian", "Indian"]
AGE_LABELS = [
    "0-9", 
    "10-19", 
    "20-29", 
    "30-39",
    "40-59", 
    "60+"
]
CATEGORY_MAP = {
    "Emotion": ("emotion", EMOTION_LABELS),
    "Gender": ("gender", GENDER_LABELS),
    "Race": ("race", RACE_LABELS),
    "Age": ("age", AGE_LABELS),
}

def _majority(values):
    """
    Tìm giá trị xuất hiện nhiều nhất trong danh sách (Voting).
    Ví dụ: ['happy', 'neutral', 'happy'] -> 'happy'
    """
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]

def _setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


def analyze_people(people_data, output_dir="output"):
    """
    Phân tích dữ liệu từ các ID (Tracking ID).
    
    1. Tổng hợp dữ liệu (Emotion, Gender...) cho từng ID.
    2. Vẽ biểu đồ.
    3. Xuất file (JSON, CSV, Excel).
    """

    if not people_data:
        print("Không có dữ liệu để phân tích.")
        return None

    # Đường dẫn thư mục output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tổng hợp dữ liệu
    summary = {
        "emotion": [],
        "emotion_conf": [],
        "gender": [],
        "race": [],
        "age": [],
        "timeline": [],
    }


    # Duyệt qua từng người (Track ID)
    for person_id, data in people_data.items():
        # Lấy nhãn phổ biến nhất (Majority Voting)
        # Ví dụ: Xuất hiện trong 10 frame, có 2 cảm xúc xuất hiện nhưng 7 frame có biểu hiện là 'happy' -> Người này là 'happy'
        summary["emotion"].append(_majority(data.get("emotion", [])))
        summary["gender"].append(_majority(data.get("gender", [])))
        summary["race"].append(_majority(data.get("race", [])))
        summary["age"].append(_majority(data.get("age", [])))

        # Gom toàn bộ lịch sử độ tin cậy và timeline
        if data["emotion_conf"] != []:
            summary["emotion_conf"].extend(data["emotion_conf"])
        else: summary.pop("emotion_conf", None)
        
        if data["timeline"] != []:
            summary["timeline"].extend(data["timeline"])
        else: summary.pop("timeline", None)

    # Lọc bỏ giá trị None
    for k in ["emotion", "gender", "race", "age"]:
        summary[k] = [v for v in summary[k] if v is not None]

    # Visualization
    try:
        _plot_bar(summary, output_dir / "statistics_counts.png")
        _plot_pie(summary, output_dir / "statistics_percent.png")
        plot_emotion_gender(summary, output_dir / "emotion_gender.png")
        plot_emotion_gender_heatmap(summary, output_dir / "emotion_gender_heatmap.png")
        plot_emotion_age_heatmap(summary, output_dir / "emotion_age_heatmap.png")
        plot_emotion_race_heatmap(summary, output_dir / "emotion_race_heatmap.png")

        if summary.get("timeline"):
            plot_emotion_timeline(summary["timeline"], output_dir / "emotion_timeline.png")

        if summary.get("emotion_conf"):
            plot_confidence_distribution(
                summary["emotion_conf"], 
                output_dir / "emotion_confidence.png"
            )
    except Exception as e:
        print(f"Lỗi {e}")
        import traceback
        traceback.print_exc()

    
    # JSON Summary
    with open(output_dir / "people_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV & Excel
    try:
        export_people_csv(summary, output_dir)
        export_people_excel(summary, output_dir)
    except Exception as e:
        print(f"Không thể xuất Excel/CSV: {e}")
    
    print(f"Dữ liệu lưu : {output_dir.resolve()}")
    return summary

def _plot_bar(summary, save_path):
    """Vẽ 4 biểu đồ cột (Bar Chart) cho Emotion, Gender, Race, Age"""
    plt, sns = _setup_plotting()
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Thống kê số lượng người theo nhóm", fontsize=16, fontweight="bold")

    # Duyệt qua 4 loại thuộc tính
    for ax, (title, (key, labels)) in zip(axes.flatten(), CATEGORY_MAP.items()):
        values = summary.get(key, [])
        counter = Counter(values)
        counts = [counter.get(lbl, 0) for lbl in labels]
        total = max(sum(counts), 1)

        palette = sns.color_palette("viridis", len(labels))
        bars = ax.bar(labels, counts, color=palette, edgecolor="black", linewidth=0.7)

        # Thêm nhãn số lượng trên đỉnh cột
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{c}\n({c/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9
                )

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Số lượng")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_pie(summary, save_path):
    """Vẽ 4 biểu đồ tròn (Pie Chart)"""
    plt, sns = _setup_plotting()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Tỷ lệ phần trăm phân bố", fontsize=16, fontweight="bold")

    for ax, (title, (key, labels)) in zip(axes.flatten(), CATEGORY_MAP.items()):
        counter = Counter(summary.get(key, []))
        
        # Vẽ những label có dữ liệu (>0) để biểu đồ đẹp hơn
        labels_f = [lbl for lbl in labels if counter.get(lbl, 0) > 0]
        counts_f = [counter[lbl] for lbl in labels_f]

        if not counts_f:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.axis("off")
            continue

        palette = sns.color_palette("Set3", len(labels_f))
        ax.pie(
            counts_f, labels=labels_f,
            autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
            startangle=90, colors=palette,
            wedgeprops={"edgecolor": "white", "linewidth": 1}
        )
        ax.set_title(title, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_emotion_gender(summary, save_path):
    """Cảm xúc theo Giới tính (Grouped Bar Chart)"""
    plt, sns = _setup_plotting()
    
    df = pd.DataFrame({
        "Emotion": summary["emotion"],
        "Gender": summary["gender"],
    })
    if df.empty: return

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Emotion", hue="Gender", palette="Set2")

    plt.title("Phân bố cảm xúc theo giới tính", fontweight="bold")
    plt.ylabel("Số lượng")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_emotion_gender_heatmap(summary, save_path):
    """
    Heatmap: Cảm xúc vs Giới tính
    """
    plt, sns = _setup_plotting()
    df = pd.DataFrame({"Emotion": summary["emotion"], "Gender": summary["gender"]})
    if df.empty: return

    # Tạo bảng chéo
    pivot = pd.crosstab(df["Emotion"], df["Gender"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Purples", linewidths=.5, cbar=False)
    plt.title("Emotion x Gender Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_emotion_age_heatmap(summary, save_path):
    """Heatmap: Cảm xúc vs Độ tuổi"""
    plt, sns = _setup_plotting()
    df = pd.DataFrame({"Emotion": summary["emotion"], "Age": summary["age"]})
    if df.empty: return

    pivot = pd.crosstab(df["Emotion"], df["Age"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5)
    plt.title("Emotion x Age Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_emotion_race_heatmap(summary, save_path):
    """Heatmap: Cảm xúc vs Chủng tộc"""
    plt, sns = _setup_plotting()
    df = pd.DataFrame({"Emotion": summary["emotion"], "Race": summary["race"]})
    if df.empty: return

    pivot = pd.crosstab(df["Emotion"], df["Race"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="GnBu", linewidths=.5)
    plt.title("Emotion x Race Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    plt, sns = _setup_plotting()

    df = pd.DataFrame({"Emotion": summary["emotion"], "Gender": summary["gender"]})
    if df.empty: return

    pivot = pd.crosstab(df["Emotion"], df["Gender"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Greens", linewidths=.5)

    plt.title("Tương quan Cảm xúc - Giới tính", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_emotion_timeline(timeline, save_path):
    """Biểu đồ đường (Line Chart): Diễn biến cảm xúc theo thời gian"""
    plt, sns = _setup_plotting()

    df = pd.DataFrame(timeline)
    # Lấy những frame có độ tin cậy >= 0.5
    df = df[df["confidence"] >= 0.5]
    if df.empty: return

    # Đếm số lượng người có cảm xúc X tại thời điểm T
    # groupby: Gom theo giây (time) và loại cảm xúc -> Đếm số ID unique
    emotion_counts = (
        df.groupby(["time", "emotion"])["person_id"]
        .nunique()
        .reset_index(name="count")
    )

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=emotion_counts,
        x="time", y="count", hue="emotion",
        linewidth=2, marker="o"
    )

    plt.title("Diễn biến cảm xúc theo thời gian", fontweight="bold")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Số lượng người")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confidence_distribution(confidences, save_path):
    """Biểu đồ phân phối độ tin cậy (Histogram)"""
    plt, sns = _setup_plotting()
    if not confidences: return

    plt.figure(figsize=(8, 4))
    sns.histplot(confidences, bins=20, kde=True, color="steelblue")

    plt.title("Phân phối độ tin cậy mô hình", fontweight="bold")
    plt.xlabel("Độ tin cậy (Confidence Score)")
    plt.ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()