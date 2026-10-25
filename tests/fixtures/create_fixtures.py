import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def generate_fixtures(output_dir: Path):
    """
    Generates synthetic controlled forensic test fixtures:
    1. Pristine baseline image
    2. Exact Copy-Move forged image
    3. Resampled / stretched manipulated image
    4. Spliced image with artificial sharp boundary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base image with distinct textures
    w, h = 400, 400
    base = Image.new("RGB", (w, h), color=(220, 225, 230))
    draw = ImageDraw.Draw(base)

    # Add geometric shapes and gradients to create local keypoints
    for i in range(10, 380, 40):
        draw.line([(i, 10), (i, 380)], fill=(180, 190, 200), width=2)
        draw.line([(10, i), (380, i)], fill=(180, 190, 200), width=2)

    # Draw a unique textured object at (50, 50)
    draw.rectangle([50, 50, 130, 130], fill=(40, 80, 180), outline=(20, 40, 100), width=3)
    draw.ellipse([65, 65, 115, 115], fill=(240, 160, 40))
    draw.polygon([(70, 70), (90, 110), (110, 70)], fill=(200, 30, 30))

    pristine_path = output_dir / "pristine_sample.jpg"
    base.save(pristine_path, "JPEG", quality=95)

    # 2. Copy-Move Forgery: Duplicate the object from (50, 50) to (240, 240)
    copymove = base.copy()
    crop_obj = base.crop((48, 48, 132, 132))
    copymove.paste(crop_obj, (240, 240))
    copymove_path = output_dir / "copymove_sample.jpg"
    copymove.save(copymove_path, "JPEG", quality=95)

    # 3. Resampled image: Rescale patch with interpolation
    resampled = base.copy()
    scaled_patch = crop_obj.resize((120, 120), Image.Resampling.BILINEAR)
    resampled.paste(scaled_patch, (220, 220))
    resampled_path = output_dir / "resampled_sample.jpg"
    resampled.save(resampled_path, "JPEG", quality=95)

    # 4. Spliced image with sharp boundary & high noise
    spliced = base.copy()
    noise_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)
    spliced.paste(noise_img, (200, 50))
    spliced_path = output_dir / "spliced_sample.jpg"
    spliced.save(spliced_path, "JPEG", quality=95)

    # 5. Clean PNG fixture
    png_path = output_dir / "pristine_sample.png"
    base.save(png_path, "PNG")

    return {
        "pristine": pristine_path,
        "copymove": copymove_path,
        "resampled": resampled_path,
        "spliced": spliced_path,
        "png": png_path
    }

if __name__ == "__main__":
    out = Path(__file__).parent
    generate_fixtures(out)
    print("Test fixtures generated successfully.")
