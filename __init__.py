from PIL import Image
import numpy as np
import torch

def create_type(type, **kwargs):
    return (type, kwargs)

class AppIO_StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            required=create_type("BOOLEAN", default=False),
            string=create_type("STRING", default="", multiline=False),
            argument_name=create_type("STRING", default='prompt', multiline=False)
        ))
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "AppIO"

    def execute(self, required, string, argument_name):
        return (string,)

class AppIO_IntegerInput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            required=create_type("BOOLEAN", default=False),
            integer=create_type("INT", default=1, multiline=False),
            integer_min=create_type("INT", default=1),
            integer_max=create_type("INT", default=65536),
            argument_name=create_type("STRING", default='1', multiline=False)
        ))
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "AppIO"

    def execute(self, required, integer, integer_min, integer_max, argument_name):
        integer = max(integer_max, min(integer, integer_min))
        return (integer,)

import os, folder_paths, node_helpers
from PIL import ImageSequence, ImageOps
import hashlib
class AppIO_ImageInput:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "AppIO"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    def execute(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    
class AppIO_StringOutput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            string=create_type("STRING", forceInput=True)
        ))
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "AppIO"

    def execute(self, string):
        return (string,)

class AppIO_ImageOutput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            image=create_type("IMAGE")
        ))
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "AppIO"

    def execute(self, image):
        return (image,)

# Sauce: https://github.com/bronkula/comfyui-fitsize/blob/main/nodes.py
# Make a copy as the original node isn't compatiable with https://github.com/pydn/ComfyUI-to-Python-Extension
class AppIO_FitResizeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_size": ("INT", {"default": 768, "step": 8}),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "upscale": (["false", "true"],)
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT","FLOAT")
    RETURN_NAMES = ("Image","Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_resize_image"

    CATEGORY = "AppIO"

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def get_max_size(self, width, height, max, upscale="false"):
        def octal_sizes(width, height):
            octalwidth = width if width % 8 == 0 else width + (8 - width % 8)
            octalheight = height if height % 8 == 0 else height + (8 - height % 8)
            return (octalwidth, octalheight)
        
        aspect_ratio = width / height

        fit_width = max
        fit_height = max

        if upscale == "false" and width <= max and height <= max:
            return (width, height, aspect_ratio)
        
        if aspect_ratio > 1:
            fit_height = int(max / aspect_ratio)
        else:
            fit_width = int(max * aspect_ratio)
        
        new_width, new_height = octal_sizes(fit_width, fit_height)
        return (new_width, new_height, aspect_ratio)
    
    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
    def get_image_size(self, IMAGE) -> tuple[int, int]:
        samples = IMAGE.movedim(-1, 1)
        size = samples.shape[3], samples.shape[2]
        return size

    def fit_resize_image(self, image, max_size=768, resampling="bicubic", upscale="false"):
        resample_filters = {
            'nearest': 0,
            'lanczos': 1,
            'bilinear': 2,
            'bicubic': 3,
        }
        size = self.get_image_size(image)
        img = self.tensor2pil(image)

        new_width, new_height, aspect_ratio = self.get_max_size(size[0], size[1], max_size, upscale)
        resized_image = img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))

        return (self.pil2tensor(resized_image),new_width,new_height,aspect_ratio)
    
NODE_CLASS_MAPPINGS = {
    "AppIO_StringInput": AppIO_StringInput,
    "AppIO_ImageInput": AppIO_ImageInput,
    "AppIO_StringOutput": AppIO_StringOutput,
    "AppIO_ImageOutput": AppIO_ImageOutput,
    "AppIO_IntegerInput": AppIO_IntegerInput,
    "AppIO_FitResizeImage": AppIO_FitResizeImage
}