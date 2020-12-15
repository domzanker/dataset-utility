import cv2
import argparse
from pathlib import Path
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor, as_completed

FILES_PATTERN = {"front": "cam_front"}


def write_scene(scene_path, args):
    # for every sample_* dir in SCENE_PATH
    output_path = args.out_path / scene_path.name
    writer = cv2.VideoWriter(
        f"{output_path}.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        15,
        (args.width, args.height),
    )

    frames = []
    # print(args.scene_path.glob("sample"))
    for sample in trange(len(list(scene_path.glob("sample_*"))), unit="frame"):
        sample_path = scene_path / f"sample_{sample}"
        # load every image file
        images = {}
        for key, perspective in FILES_PATTERN.items():
            # file = entry.glob(f"{perspective}.png")

            file = sample_path / f"{perspective}.png"
            img = cv2.imread(str(file))
            if img is not None:
                """
                cv2.imshow("frame", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                """
                writer.write(img)

    writer.release()


def write_all_scenes(args):
    executor = ProcessPoolExecutor(max_workers=10)
    scenes = list(args.path.glob("scene_*"))
    with tqdm(total=len(scenes), leave=False, smoothing=0, unit="sample") as pbar:
        """
        for i in range(number_of_samples):
            sample_pipeline(dataset.scene_dir)
            pbar.update(1)
        """
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(write_scene, scene_path=s, args=args): s for s in scenes
            }

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, default="/tmp/carla/videos")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--preview", type=bool, default=False)

    args = parser.parse_args()
    args.out_path.mkdir(parents=True, exist_ok=True)
    write_all_scenes(args)
