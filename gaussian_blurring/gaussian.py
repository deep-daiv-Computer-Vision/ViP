import cv2
import numpy as np
import os
import argparse
import glob

# 핑크컬러 마스크를 binary로 바꾸기
def mask_to_binary(input_folder, output_frame):
    # output_frame 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_frame):
        os.makedirs(output_frame)

    # 목표 색상 (BGR)
    target_color = np.array([182, 116, 245])  # BGR (OpenCV는 BGR 순서를 사용합니다)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # 이미지 파일만 처리
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
    
        image = cv2.imread(file_path)

        # 이미지 로드 실패 시 에러 메시지 출력
        if image is None:
            print(f'Error: Failed to load image from {file_path}')
            continue
        
        # 타겟 컬러만 마스크로 설정 (모든 픽셀에 대해 타겟 컬러와 동일한지 확인)
        mask = cv2.inRange(image, target_color, target_color)

        # 마스크 저장
        binary_mask = os.path.join(output_frame, f'mask_{filename}')
        cv2.imwrite(binary_mask, mask)

        print(f'Binary mask saved to {binary_mask}')

# 2. binary mask로 gaussian blur 
def gaussian_blur(image_folder, mask_folder, output_folder):
    # output_folder 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted(glob.glob(os.path.join(image_folder, "*.*")))
    mask_files = sorted(glob.glob(os.path.join(mask_folder, "*.*")))

    print(f'Found {len(image_files)} images and {len(mask_files)} masks.')

    # 이미지 파일 수와 마스크 파일 수가 일치하지 않으면 경고 메시지 출력
    if len(image_files) != len(mask_files):
        print("Warning: The number of images and masks do not match.")
    
    for idx, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
        print(f'Processing image {image_path} and mask {mask_path}')

        # 이미지와 마스크 파일만 처리
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')) or not mask_path.lower().endswith('.png'):
            print(f'Skipping non-image or non-mask file: {image_path}, {mask_path}')
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f'Error: Failed to load image from {image_path}')
            continue
        
        if mask is None:
            print(f'Error: Failed to load mask from {mask_path}')
            continue

        # Ensure the mask is 8-bit single channel
        if mask.dtype != np.uint8:  # 추가된 부분
            mask = mask.astype(np.uint8)  # 추가된 부분

        # Ensure the mask size matches the image size
        if mask.shape != image.shape[:2]:  # 추가된 부분
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)  # 추가된 부분

        # Proceed with bitwise_and operation
        try:
            original_section = cv2.bitwise_and(image, image, mask=mask)
        except cv2.error as e:  # 추가된 부분
            print(f'Error applying bitwise_and: {e}')  # 추가된 부분
            continue  # 추가된 부분

        original_section = cv2.bitwise_and(image, image, mask=mask)
        blurred_image = cv2.GaussianBlur(image, (45, 45), 0)    
        inverse_mask = cv2.bitwise_not(mask)    
        blurred_section = cv2.bitwise_and(blurred_image, blurred_image, mask=inverse_mask)
        result = cv2.add(original_section, blurred_section)

        output_filename = f'blur_{idx:05d}.png'
        out_frame = os.path.join(output_folder, output_filename)
        cv2.imwrite(out_frame, result)

        print(f'Result saved to {out_frame}')

# 3. 2번 결과를 비디오로 변환
def videoGen(output_folder, output_video_path):
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = sorted([img for img in os.listdir(output_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    fps = 60
    frame_array = []

    for img_name in images: 
        img_path = os.path.join(output_folder, img_name)
        img = cv2.imread(img_path)
        frame_array.append(img)

    if not frame_array:
        print("No frames to process.")
        return

    height, width, layers = frame_array[0].shape
    size = (width, height)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frame_array:
        out.write(frame)
    out.release()
    print(f"video saved to {output_video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images, apply Gaussian blur, and generate video.')
    parser.add_argument('--img-path', type=str, required=True, help='Path to the image folder')
    parser.add_argument('--color-mask', type=str, required=True, help='Path to the color mask folder')
    parser.add_argument('--output-frame', type=str, required=True, help='Path to the output folder for processed frames')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the output video file')
    parser.add_argument('--binary-mask', type=str, required=True, help='Path to the output folder for binary masks')

    args = parser.parse_args()

    # 1. 컬러 마스크 파일 -> binary로 
    mask_to_binary(args.color_mask, args.binary_mask)

    # 2. binary mask 로 gaussian blur
    gaussian_blur(args.img_path, args.binary_mask, args.output_frame)

    # 3. 결과 프레임을 비디오로 변환
    videoGen(args.output_frame, args.video_path)
