import argparse
import warnings

from ultralytics import YOLO

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='GS-YOLO Model Export Script')
    parser.add_argument('--model', type=str, default='runs/detect/gs-yolo-train/weights/best.pt',
                        help='trained model weights path')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'torchscript', 'coreml', 'tflite', 'pmml', 'openvino'],
                        help='export format')
    parser.add_argument('--imgsz', type=int, default=640, help='image size used for model export')
    parser.add_argument('--device', type=str, default='cpu', help='device used for export')
    parser.add_argument('--verbose', action='store_true', help='print debug logs')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print('[GS-YOLO] export args:', args)

    model = YOLO(args.model)

    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
    )

    print(f"[GS-YOLO] model exported to format={args.format} (see runs/)")


if __name__ == '__main__':
    main()
