"""Render PDF pages and a contact sheet for visual inspection, not just page-count checks."""
import argparse
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    args = parser.parse_args()
    source = Path(args.pdf)
    output = source.parent / "rendered"
    output.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(source)
    thumb_width, thumb_height = 360, 530
    columns = 3
    rows = (len(document) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "#d3d7dc")
    draw = ImageDraw.Draw(sheet)
    for i, page in enumerate(document):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        path = output / f"page_{i+1:02d}.png"
        pix.save(path)
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width-12, thumb_height-24))
        x, y = (i % columns)*thumb_width, (i // columns)*thumb_height
        sheet.paste(image, (x+6, y+18))
        draw.text((x+6,y+3), f"Page {i+1}", fill="black")
    sheet.save(output / "contact_sheet.png")
    print(f"Rendered {len(document)} pages to {output}")


if __name__ == "__main__":
    main()
