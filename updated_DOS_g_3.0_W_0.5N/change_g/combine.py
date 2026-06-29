


from PIL import Image
import matplotlib.pyplot as plt

# Load images
paths = [
    "ap_circular_heatmap_g_1.png",
    "ap_circular_heatmap_g_3.png",
    "ap_circular_heatmap_g_5.png",
    "ap_circular_heatmap_g_7.png",
    "ap_circular_heatmap_g_9.png",

]

images = [Image.open(p) for p in paths]

# Create combined figure
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for ax, img, title in zip(
    axes,
    images,
    ["g = 1", "g = 3", "g = 5", "g = 7", "g = 9"]
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
out_path = "combined_corona_ring_g.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()