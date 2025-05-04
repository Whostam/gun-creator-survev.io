import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# -- Utility Functions --
def extract_palette(image: Image.Image, n_colors: int = 5):
    """
    Extracts a dominant color palette from the given PIL image using KMeans clustering.
    """
    # Resize for faster clustering
    small = image.resize((64, 64))
    arr = np.array(small).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]


def generate_gun_image(palette: list, resolution: tuple = (64, 64)) -> Image.Image:
    """
    Generates a simple pixel-art gun-like image by randomly assigning colors from the palette.
    This stub can be replaced by more advanced shape/template-based algorithms.
    """
    img = Image.new('RGB', resolution, color=palette[0])
    pixels = img.load()
    for x in range(resolution[0]):
        for y in range(resolution[1]):
            pixels[x, y] = palette[np.random.randint(len(palette))]
    return img

# -- Streamlit App --

def main():
    st.title("Survev.io-Style Gun Generator")
    st.markdown(
        "Upload Swrv.io gun images to extract visual style and generate new pixel-art guns."
    )

    # Image upload
    uploaded_files = st.file_uploader(
        "Upload Survev.io gun images (PNG, JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        palettes = []
        st.header("Extracted Color Palettes")
        for file in uploaded_files:
            img = Image.open(file).convert('RGB')
            pal = extract_palette(img)
            palettes.append(pal)
            # Display swatches
            swatch = np.zeros((50, 50 * len(pal), 3), dtype=np.uint8)
            for i, col in enumerate(pal):
                swatch[:, i * 50:(i + 1) * 50] = col
            st.image(swatch, caption=f"Palette from {file.name}")

        # Selection and generation controls
        idx = st.selectbox(
            "Select base image for style", options=list(range(len(palettes))),
            format_func=lambda i: uploaded_files[i].name
        )
        chosen_palette = palettes[idx]

        st.subheader("Generate New Gun")
        res_option = st.selectbox("Resolution", options=["32x32", "64x64", "128x128"], index=1)
        res = tuple(map(int, res_option.split('x')))

        if st.button("Generate Gun"):
            new_gun = generate_gun_image(chosen_palette, resolution=res)
            st.image(new_gun, caption="Custom Survev.io-Style Gun")
            # Example stats
            stats = {
                "damage": int(np.random.uniform(10, 50)),
                "fire_rate": round(np.random.uniform(0.1, 1.0), 2),
                "reload_time": round(np.random.uniform(1.0, 3.0), 2)
            }
            st.json(stats)

    st.sidebar.header("About")
    st.sidebar.info(
        "This prototype uses KMeans color extraction and random pixel patterns."
    )

if __name__ == '__main__':
    main()
