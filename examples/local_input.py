"""One request from examples/local_inputs/ x four seeds with one cell of the grid.

    python examples/local_input.py examples/local_inputs/keypoints_03.txt [weight] [respond] [class id]

Give a class id to ask for the same mask or pose on another class.
"""
import sys
from pathlib import Path
from PIL import Image
from lism import load_task, mask_from_text, pose_from_text, preset, sample, to_pil
from lism.backbones import load_rae

# The generator's class condition for each request.
CLASSES = {"foreground_01": 281, "foreground_02": 269, "foreground_03": 129, "foreground_04": 3,      # cat, wolf, spoonbill, shark
           "foreground_05": 898, "foreground_06": 874, "foreground_07": 404, "foreground_08": 671,    # bottle, trolleybus, airliner, bike
           "keypoints_01": 340, "keypoints_02": 292, "keypoints_03": 292, "keypoints_04": 339,        # zebra, tiger, tiger, horse
           "keypoints_05": 339, "keypoints_06": 291, "keypoints_07": 386}                             # horse, lion, elephant


def main(path, weight="fit", respond="path", class_id=None):
    path, text = Path(path), Path(path).read_text()
    kind = path.stem.split("_")[0]                                  # foreground or keypoints
    task = load_task(kind)
    wanted = mask_from_text(text) if kind == "foreground" else pose_from_text(text)
    class_id = CLASSES[path.stem] if class_id is None else int(class_id)
    model = load_rae(class_id=class_id)
    request, guide = task.request(wanted), preset(weight, respond, task)
    output = Path("outputs") / f"{path.stem}_{class_id}_{weight}_{respond}"
    output.mkdir(parents=True, exist_ok=True)
    overview = Image.new("RGB", (1024, 256))
    for col, seed in enumerate(range(52900, 52904)):
        image = to_pil(sample(model, request, guide, seed=seed))
        image.save(output / f"{seed}.png")
        overview.paste(image, (256 * col, 0))
    overview.save(output / "overview.png")
    print(f"Wrote {output}/overview.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
