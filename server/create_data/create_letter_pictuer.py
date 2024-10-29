import os
from PIL import Image, ImageDraw, ImageFont




def save_letters_as_images(font_folder, output_folder, characters, image_size):
    # List all font files in the font folder
    font_files = [file for file in os.listdir(font_folder)]
    for font_file in font_files:
        # Extract font name from the file
        font_name = os.path.splitext(font_file)[0]
        os.makedirs(output_folder, exist_ok=True)

        # Load the font file
        font_path = os.path.join(font_folder, font_file)
        font = ImageFont.truetype(font_path, size=image_size[0])
        data_char = [char_folders for char_folders in os.listdir(output_folder)]

        for char, char_folder in zip(characters, data_char):

            # Create a blank image
            img = Image.new("RGB", image_size, color="white")
            draw = ImageDraw.Draw(img)
            text_bbox = font.getbbox(char)

            # Calculate the position to center the text
            x = (image_size[0] - (text_bbox[2] - text_bbox[0])) // 2 - text_bbox[0]
            y = (image_size[1] - (text_bbox[3] - text_bbox[1])) // 2 - text_bbox[1]

            draw.text((x, y), char, font=font, fill="black")
            char_output_folder = os.path.join(output_folder, char_folder)
            # Save the image
            img.save(os.path.join(char_output_folder, f"{char_folder}_{font_name}.png"))


if __name__ == "__main__":
    font_folder = "./fonts/english_font"
    output_folder = "./data/capital_english_letters"
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    image_size = (64, 64)
    save_letters_as_images(font_folder, output_folder, characters, image_size)

    font_folder_lower = "./fonts/english_font"
    output_folder_lower = "./data/english_lowercase_letters"
    characters_lower = "abcdefghijklmnopqrstuvwxyz"
    image_size = (64, 64)
    save_letters_as_images(font_folder_lower, output_folder_lower, characters_lower, image_size)

    font_folder_lower = "./fonts/hebrew_font_writing"
    output_folder_lower = "./data/hebrew_Spelling_and_writing_letters"
    characters_lower = "אבחדעגהךכקלםמןנףפרסשתטויצץז"
    image_size = (64, 64)
    save_letters_as_images(font_folder_lower, output_folder_lower, characters_lower, image_size)

    font_folder_typefaces = "./fonts/hebrow_font_typefaces"
    output_folder_typefaces = "./data/hebrew_typefaces"
    characters_typefaces = "אבחדעגהךכקלםמןנףפרסשתטויצץז"
    image_size = (64, 64)
    save_letters_as_images(font_folder_lower, output_folder_lower, characters_lower, image_size)

    font_files = [file for file in os.listdir('./fonts') ]
    for fonts in font_files:
        font_folder_numbers = f"./fonts/{fonts}"
        output_folder_numbers = "./data/numbers_and_signs"
        characters_numbers = "&'*(:,$8!54-#91)%+.?76/320"
        image_size = (64, 64)
        save_letters_as_images(font_folder_numbers, output_folder_numbers, characters_numbers, image_size)










