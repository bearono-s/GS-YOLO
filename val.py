import argparse
import warnings

from ultralytics import YOLO

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='GS-YOLO Validation Script')
    parser.add_argument('--model', type=str, default='runs/detect/gs-yolo-train/weights/best.pt',
                        help='trained YOLO model path or loaded name')
    parser.add_argument('--data', type=str, default='datasets/VisDrone/visDrone.yaml',
                        help='dataset yaml path for validation')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='device id or cpu')
    parser.add_argument('--verbose', action='store_true', help='print debug logs')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print('[GS-YOLO] validation args:', args)

    model = YOLO(args.model)

    results = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

    print('[GS-YOLO] validation results:')
    print(results)


if __name__ == '__main__':
    main()
