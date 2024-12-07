import argparse
from insetGAN import INSETGAN

def main(args):
    # 인스턴스 생성
    inference_model = INSETGAN()
    # 이미지 평가 수행
    inference_model.evaluate(args.canvas_image_path, args.inset_image_path, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate canvas and inset images using ImageInference")
    parser.add_argument('--canvas_image_path', type=str, default='/workspace/results_config_2/0_24.png', help="Path to the canvas image")
    parser.add_argument('--inset_image_path', type=str, default='/workspace/results_config_2/0_face_input.png', help="Path to the inset image")
    parser.add_argument('--output_path', type=str, default='/workspace/results_config_2/result.png', help="Path to save the output image")

    args = parser.parse_args()
    main(args)