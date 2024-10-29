from PIL import Image
import os

def rotate_and_save(input_path, output_folder,char_img):
    # Load the image
    try:
        # Load the image
        original_image = Image.open(input_path)

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Rotate and save images in 22.5-degree increments (16 images in total)
        for angle in range(0, 360, 22):
            rotated_image = original_image.rotate(angle,fillcolor="white")
            rotate_folder = os.path.join(output_folder, 'rotate_img')
            os.makedirs(rotate_folder, exist_ok=True)
            char_img_name = os.path.splitext(char_img)[0]
            output_path = os.path.join(rotate_folder, f"{char_img_name}_rotated_{angle}.png")
            rotated_image.save(output_path)

            print(f"Rotated image saved: {output_path}")

    except Exception as e:
        print(f"Error processing image {input_path}: {str(e)}")
if __name__ == "__main__":
    # Specify the path to your input image
    data_folders = [data_letter for data_letter in os.listdir("./data")]
    for data_letter_folder in data_folders:
        img_files = [file for file in os.listdir(f"./data/{data_letter_folder}")]
        for img_file in img_files:
            # Extract font name from the file
            print(img_file)
            char_img_files = [file for file in os.listdir(f"./data/{data_letter_folder}/{img_file}")]
            for char_img in char_img_files:
                print(char_img)
                input_image_path = f"./data/{data_letter_folder}/{img_file}/{char_img}"

                # Specify the output folder
                output_folder = f"./data/{data_letter_folder}/{img_file}"

                # Call the function to rotate and save images
                rotate_and_save(input_image_path, output_folder, char_img)


