import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

# =============================
# BẢNG NHÃN
# =============================
# label_to_idx = {
#     'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
#     'neutral': 4, 'sad': 5, 'surprise': 6, 'contempt': 7
# }

label_to_idx = {
    'angry': 0, 'fear': 1, 'happy': 2,
    'neutral': 3, 'sad': 4, 'surprise': 5,
}

idx_to_gender = {0: "Male", 1: "Female"}
idx_to_race = {
    0: "White", 1: "Black", 2: "Latino_Hispanic",
    3: "East Asian", 4: "Southeast Asian",
    5: "Indian", 6: "Middle Eastern"
}
idx_to_age = {
    0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29",
    4: "30-39", 5: "40-49", 6: "50-59",
    7: "60-69", 8: "70+"
}



# =============================
# HÀM PHÂN TÍCH & VẼ BIỂU ĐỒ
# =============================
def analyze_people(people_data):
    print("\n==== THỐNG KÊ THEO NGƯỜI ====")

    if not people_data:
        print("⚠️ Không có người nào được ghi nhận.")
        return

    # Tạo tóm tắt dữ liệu
    summary = {"Emotion": [], "Gender": [], "Race": [], "Age": []}

    for pid, data in people_data.items():
        summary["Emotion"].append(Counter(data["emotion"]).most_common(1)[0][0])
        summary["Gender"].append(Counter(data["gender"]).most_common(1)[0][0])
        summary["Race"].append(Counter(data["race"]).most_common(1)[0][0])
        summary["Age"].append(Counter(data["age"]).most_common(1)[0][0])

    total_people = len(people_data)
    print(f"👥 Tổng số người phát hiện: {total_people}")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9
    })

    categories = {
        "Emotion": list(label_to_idx.keys()),
        "Gender": list(idx_to_gender.values()),
        "Race": list(idx_to_race.values()),
        "Age": list(idx_to_age.values())
    }

    # =============================
    # 1️⃣ BIỂU ĐỒ CỘT — SỐ LƯỢNG NGƯỜI
    # =============================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
    fig1.suptitle("Thống kê số lượng người theo nhãn", fontsize=16, fontweight='bold')

    for ax, (key, labels) in zip(axes1.flatten(), categories.items()):
        counter = Counter(summary[key])
        counts = [counter.get(lbl, 0) for lbl in labels]
        total = sum(counts) if sum(counts) > 0 else 1

        palette = sns.color_palette("viridis", len(labels))
        bars = ax.bar(labels, counts, color=palette, edgecolor="black", linewidth=0.7)

        for bar, c in zip(bars, counts):
            if c > 0:
                percent = (c / total) * 100
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{c}\n({percent:.1f}%)",
                        ha='center', va='bottom', fontsize=9, color='black', fontweight='medium')

        ax.set_title(f"{key}", fontweight="bold")
        ax.set_ylabel("Số lượng người")
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./output/statistics_counts_pretty.png", bbox_inches="tight", dpi=300)
    print("Đã lưu biểu đồ số lượng: ./output/statistics_counts_pretty.png")

    # =============================
    # 2️⃣ BIỂU ĐỒ TRÒN — TỶ LỆ %
    # =============================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle("Tỷ lệ phần trăm theo nhãn", fontsize=16, fontweight='bold')

    for ax, (key, labels) in zip(axes2.flatten(), categories.items()):
        counter = Counter(summary[key])
        counts = [counter.get(lbl, 0) for lbl in labels]

        # --- Lọc bỏ các nhãn có giá trị = 0 ---
        filtered_labels = [lbl for lbl, c in zip(labels, counts) if c > 0]
        filtered_counts = [c for c in counts if c > 0]

        if len(filtered_counts) == 0:
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue

        palette = sns.color_palette("Set3", len(filtered_labels))
        wedges, texts, autotexts = ax.pie(
            filtered_counts,
            labels=filtered_labels,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
            colors=palette,
            startangle=90,
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.7}
        )

        # Căn giữa chữ % cho đẹp
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('medium')

        ax.set_title(f"{key}", fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./output/statistics_percent_pretty.png", bbox_inches="tight", dpi=300)
    print("Đã lưu biểu đồ tỷ lệ: ./output/statistics_percent_pretty.png")

    # =============================
    # 3️⃣ LƯU DỮ LIỆU JSON
    # =============================
    with open("people_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Dữ liệu thống kê đã lưu tại: people_summary.json")
