import os
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_camera_predictor,
)

# sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
checkpoint = "../checkpoints/efficienttam_s_512x512.pt"
model_cfg = "configs/efficienttam/efficienttam_s_512x512.yaml"

predictor = build_efficienttam_camera_predictor(model_cfg, checkpoint)

# cap = cv2.VideoCapture(<your video or camera >)

if_init = False
points = np.array([[285, 207]], dtype=np.float32)
points2 = np.array([[270, 189]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

from PIL import Image
import numpy as np

image = Image.open("./videos/talos/00000.jpg")  # Load image
image = image.convert("RGB")  # Ensure 3-channel image
image_array = np.array(image)
frame =image_array

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # while True:
    #     # ret, frame = cap.read()
    #     # if not ret:
    #     #     break
    #     width, height = frame.shape[:2][::-1]
    i = 1
    while True:

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=2, points=points, labels=labels)
            # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=3, points=points2, labels=labels)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(frame_idx=0, obj_id=3, bbox=(260, 179, 280, 199), points=points2, labels=labels)
            start_time = time.time()
        else:
            image = Image.open(f"./videos/talos/{i:05d}.jpg")  # Load image
            image = image.convert("RGB")  # Ensure 3-channel image
            image_array = np.array(image)
            frame = image_array
            out_obj_ids, out_mask_logits = predictor.track(frame)

            i += 1
            if i == 200:
                break

# Record the end time and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.4f} seconds")


# show the results on the current (interacted) frame on all objects
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

video_dir = "./videos/talos"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
prompts = {}
prompts[2] = points, labels
plt.figure(figsize=(9, 6))
plt.title(f"frame 0")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[0])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    # show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.show()