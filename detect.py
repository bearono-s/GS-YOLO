import argparse
import warnings

from ultralytics import YOLO

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='GS-YOLO Inference/Detection Script')
    parser.add_argument('--model', type=str, default='runs/detect/gs-yolo-train/weights/best.pt',
                        help='trained model weights path or model name')
    parser.add_argument('--source', type=str, default='data/images',
                        help='source image/video path or camera index (0,1) or stream URL')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='device id or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save detection results to labels')
    parser.add_argument('--save-conf', action='store_true', help='save confidence score in text labels')
    parser.add_argument('--project', type=str, default='runs/detect', help='save results project path')
    parser.add_argument('--name', type=str, default='gs-yolo-detect', help='save results subfolder name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--verbose', action='store_true', help='print debug logs')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print('[GS-YOLO] detect args:', args)

    model = YOLO(args.model)

    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    print(f"[GS-YOLO] detection completed. results saved in {args.project}/{args.name}")


if __name__ == '__main__':
    main()
