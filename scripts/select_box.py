import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

box = []

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    box.clear()
    box.extend([x, y, w, h])

    print(f"Selected box: {box}")

img_path = input("Enter image path: ").strip()
img = Image.open(img_path)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
ax.set_title("Drag to select box, then close window")

selector = RectangleSelector(
    ax,
    onselect,
    useblit=True,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True
)

plt.show()

if box:
    print("\nFinal box for --box input:")
    print(*box)
else:
    print("No box selected.")