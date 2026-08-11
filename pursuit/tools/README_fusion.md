# Fine-tuning the round-7 fusion model for pursuit

Measured on live Rivermark sequences (`pursuit.tools.compare_detectors`), the
project's own 25M RGB+motion model and the simulator-trained 2.9M nano fail in
opposite directions:

| span px | nano acquire@0.20 | fusion acquire@0.20 |
|---------|------------------|---------------------|
| 4-8     | 0.055            | **0.255**           |
| 8-14    | 0.117            | **0.247**           |
| 14-25   | **0.569**        | 0.083               |
| 25-50   | **0.781**        | 0.027               |
| 50+     | **0.880**        | 0.000               |

Long range is where a lock has to *start*, so the fusion model's advantage is
the one worth keeping; its close-range collapse is a domain gap against a
renderer it has never seen. Hence fine-tune rather than choose.

    python -m pursuit.tools.make_sim_fusion          # 4ch tiles from work/simdata
    python tools/train_yolo.py \
      --data work/simdata_fusion/data.yaml \
      --model configs/yolov8m-p2-ch4.yaml \
      --weights work/runs/combined-fusion-m-p2-2/weights/best.pt \
      --imgsz 640 --epochs 30 --batch 6 --name sim-fusion-m-p2 \
      --mc --nwd --hsv 0 0 0 --lr0 0.002

`--hsv 0 0 0` is mandatory: HSV augmentation BGR->HSV converts and assumes three
channels. `--lr0 0.002` keeps the fine-tune from walking away from the real-data
weights that motivate using this model at all.

Then run the loop with `--detector fusion --weights <new best.pt>`.
