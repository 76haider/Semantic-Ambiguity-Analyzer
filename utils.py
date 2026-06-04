def create_ambiguity_chart(results_dict):
    """Create bar chart for ambiguity analysis results"""
    functions = list(results_dict.keys())
    counts = list(results_dict.values())

    # Clean colors - formal blues + extras for 10 functions
    colors = ["#2E86AB", "#345995", "#348AA7", "#4CB944",
              "#F4A261", "#E76F51", "#6C91C2", "#8FA6CB",
              "#E74C3C", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(functions, counts, color=colors[:len(functions)])

    ax.set_xlabel("Count", fontsize=11, color="#333333")
    ax.set_title("Semantic Analysis Results", fontsize=14,
                 fontweight="bold", color="#2C3E50", pad=15)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10, color="#333333")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#666666')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')

    plt.tight_layout()
    return fig
