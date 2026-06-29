from PIL import Image
import matplotlib.pyplot as plt

paths = [
    "ap_circular_heatmap_attempt321.png",
    "ap_circular_heatmap_attempt382.png",
    "ap_circular_heatmap_attempt7.png",
    "ap_circular_heatmap_attempt56.png",
    "ap_circular_heatmap_attempt27.png",
    "ap_circular_heatmap_attempt14.png"
    
]

images = [Image.open(p) for p in paths]

fig, axes = plt.subplots(1, 6, figsize=(15, 5))

for ax, img, title in zip(
    axes,
    images,
    ["Sharpness = 0.0013", "Sharpness = 0.013", "Sharpness = 0.13", "Sharpness = 6.25","Sharpness = 62.5", "Sharpness = 625"]
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

out_path = "combined_corona_ring_sharpness.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
