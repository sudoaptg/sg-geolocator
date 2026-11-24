import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sg_geolocator.clues import block_number_variants, glyph_boxes, glyph_stacks, stack_strip, stacked_numbers


def _wall_with_stack(tmp_path, text="596B"):
    """A pale wall with a block number painted down a column, plus a column of identical windows."""
    img = Image.new("RGB", (400, 600), (225, 200, 170))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    for k, ch in enumerate(text):
        d.text((100, 150 + k * 26), ch, fill=(70, 70, 70), font=font)
    for k in range(5):  # windows: same shape every floor
        d.rectangle((300, 120 + k * 40, 312, 140 + k * 40), fill=(40, 40, 50))
    path = tmp_path / "wall.png"
    img.save(path)
    return path


def test_stacked_digits_found_and_laid_out(tmp_path):
    path = _wall_with_stack(tmp_path)
    gray = np.asarray(Image.open(path).convert("L"))
    stacks = glyph_stacks(glyph_boxes(gray), gray)
    assert len(stacks) == 1 and len(stacks[0]) == 4  # the digits; the identical windows are dropped
    assert all(95 <= b[0] <= 115 for b in stacks[0])
    strip = stack_strip(gray, stacks[0])
    assert strip.shape[1] > 3 * strip.shape[0]  # characters side by side
    assert strip[:, :, 0].min() < 80 and np.median(strip) > 240  # dark text on white


def test_stacked_numbers_reads_with_lookalike_fix(tmp_path):
    path = _wall_with_stack(tmp_path)
    fake = lambda strips: ([("5968", 0.7)] * len(strips), 0.0)
    out = stacked_numbers(path, recognizer=fake)
    assert out and out[0][0] == "5968" and abs(out[0][2] - 0.27) < 0.03
    assert block_number_variants("5968") == ["5968", "596B"]
    assert block_number_variants("221") == ["221"]
    assert block_number_variants("A1") == []
