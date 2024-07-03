import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
import argparse

# 屏蔽所有警告
warnings.filterwarnings("ignore")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def process_images_in_folder(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)

    # 遍历文件夹中的每个文件
    for file_name in files:
        # 检查文件是否是PNG或JPG格式
        if file_name.lower().endswith(('.png', '.jpg')):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)

            # 执行您的图像处理操作
            process_image(file_path)
            print(file_name)


def process_image(image_path, processor=processor, model=model):
    # 打开图像
    raw_image = Image.open(image_path).convert('RGB')

    # 其余代码保持不变
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption for {image_path}: {caption}")


# 指定要处理的文件夹路径
folder_test = r"C:\BaiduNetdiskDownload\train_data"

# args settings
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument("-f",
                    "--folder",
                    type=str,
                    default=folder_test,
                    required=True,
                    help="Path to the folder containing images")
args = parser.parse_args()
folder_path = args.folder

# 处理文件夹中的所有图像
process_images_in_folder(folder_path)
