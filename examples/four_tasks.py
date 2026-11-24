"""Four tasks x four seeds with one cell of the grid: 16 images and an overview.

    python examples/four_tasks.py [weight] [respond]        default: fit path
"""
import sys
from pathlib import Path
from PIL import Image
from lism import load_task, mask_from_text, pose_from_text, preset, sample, to_pil
from lism.backbones import load_rae

INPUTS = Path(__file__).parent / "local_inputs"


def main(weight="fit", respond="path"):
    output = Path("outputs") / f"four_tasks_{weight}_{respond}"
    output.mkdir(parents=True, exist_ok=True)
    model = load_rae()
    # (name, generator class condition, task, what is asked). Class 1000 is the null class:
    # the class request is then the only signal.
    attributes = load_task("attributes", condition=609)            # the covariance fitted on class 609 (jeep)
    cases = [("class", 1000, load_task("class"), 207),                                    # golden retriever
             ("attributes", 609, attributes, {"still water": 1, "snow": 1}),
             ("foreground", 281, load_task("foreground"), mask_from_text((INPUTS / "foreground_01.txt").read_text())),     # tabby cat
             ("keypoints", 340, load_task("keypoints"), pose_from_text((INPUTS / "keypoints_01.txt").read_text()))]  # zebra
    overview = Image.new("RGB", (1024, 1024))
    for row, (name, class_id, task, wanted) in enumerate(cases):
        model.class_id = class_id
        request, guide = task.request(wanted), preset(weight, respond, task)
        for col, seed in enumerate(range(52900, 52904)):
            image = to_pil(sample(model, request, guide, seed=seed))
            image.save(output / f"{name}_{seed}.png")
            overview.paste(image, (256 * col, 256 * row))
    overview.save(output / "overview.png")
    print(f"Wrote {output}/overview.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
