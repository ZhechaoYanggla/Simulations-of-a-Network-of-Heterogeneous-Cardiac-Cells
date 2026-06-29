


from PIL import Image
import matplotlib.pyplot as plt

# Load images
paths = [
    "combined_corona_running_avg_0.9.png",
    "combined_corona_running_avg_0.89.png",
    "combined_corona_running_avg_0.88.png",
    "combined_corona_running_avg_0.87.png",
    "combined_corona_running_avg_0.85.png",
    "combined_corona_running_avg_0.83.png",
]

images = [Image.open(p) for p in paths]

# Create combined figure
fig, axes = plt.subplots(6, 1, figsize=(5, 15))

for ax, img, title in zip(
    axes,
    images,
    ["Amplitude = 0.90", "Amplitude = 0.89", "Amplitude = 0.88", "Amplitude = 0.87", "Amplitude = 0.85", "Amplitude = 0.83"]
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
out_path = "combined_corona_ring_amplitudes.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()