import cv2
import os

def apply_blur(input_file, output_folder, char_img):
    try:
        for i in range(1, 6):
            blur_strength = i * 4
            blur_strength = max(1, blur_strength)  # Ensure at least 1
            blur_strength = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength

            blur_folder = os.path.join(output_folder, 'image_blur')
            os.makedirs(blur_folder, exist_ok=True)
            char_img_name = os.path.splitext(char_img)[0]

            # Read the image
            img = cv2.imread(input_file)

            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

            # Specify the output file path
            output_path = os.path.join(blur_folder, f"{char_img_name}_blurred_{blur_strength}.png")

            # Save the blurred image
            cv2.imwrite(output_path, blurred_img)

            print(f"Blurred image saved: {output_path}")

    except Exception as e:
        print(f"Failed to process image {input_file}: {str(e)}")

if __name__ == "__main__":
    for i in range(1, 6):
        blur_strength = i * 4  # Adjust the blur strength as needed
        img_files = [file for file in os.listdir("./data")]
        for img_file in img_files:
            char_img_folders = [file for file in os.listdir(f"./data/{img_file}")]
            for char_img_folder in char_img_folders:
                char_imgs = [file for file in os.listdir(f"./data/{img_file}/{char_img_folder}")]
                for char_img in char_imgs:
                    print(char_img)
                    input_file = f"./data/{img_file}/{char_img_folder}/{char_img}"
                    output_folder = f"./data/{img_file}/{char_img_folder}"
                    apply_blur(input_file, output_folder, char_img)
