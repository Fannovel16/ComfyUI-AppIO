from PIL import Image, ImageColor
import numpy as np
import torch
from collections import namedtuple
from einops import repeat, rearrange

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

class AppIO_ImageInputFromID:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            argument_name=create_type("STRING", multiline=False, default='')
        ))
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "AppIO"
    def execute(self, argument_name):
        raise NotImplementedError("This node only works when running on ComfyUI-backed-bot")
    
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

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

class AppIO_ResizeInstanceImageMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": dict(
                image=create_type("IMAGE"),
                mask=create_type("MASK"),
                expand_out_mask=create_type("INT", min=-1024, max=1024, step=1, default=0),
                move_x=create_type("INT", min=-1024, max=1024, step=1, default=0),
                move_y=create_type("INT", min=-1024, max=1024, step=1, default=0),

                max_size=create_type("INT", min=16, max=2048, default=512, step=16),
                resampling=create_type(["lanczos", "nearest", "bilinear", "bicubic"]),
                upscale=create_type(["false", "true"], default="true"),
            ),
            "optional": dict(
                opt_bg_image=create_type("IMAGE"),
                opt_bg_mask=create_type("MASK")
            )
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize"
    CATEGORY = "AppIO"

    def resize(self, image, mask, expand_out_mask: int, move_x: int, move_y: int, opt_bg_image=None, opt_bg_mask=None, **kwargs):
        from nodes import NODE_CLASS_MAPPINGS
        resizer = AppIO_FitResizeImage()
        mask_to_segs = NODE_CLASS_MAPPINGS["MaskToSEGS"]()
        grow_mask = NODE_CLASS_MAPPINGS["GrowMask"]()
        
        segs, *_ = mask_to_segs.doit(mask, combined=False, crop_factor=1, bbox_fill=False, drop_size=10, contour_fill=False)
        (frame_h, frame_w), *segs = segs
        if opt_bg_image is not None:
            image_canvas = opt_bg_image[:1,...,:3].clone()
        else:
            image_canvas = torch.zeros(1, frame_h, frame_w, 3)
        mask_canvas = torch.zeros_like(image_canvas)[0,:,:,0]

        if opt_bg_mask is not None:
            opt_segs, *_ = mask_to_segs.doit(opt_bg_mask, combined=False, crop_factor=1, bbox_fill=False, drop_size=10, contour_fill=False)
            _, *opt_segs = opt_segs
            opt_seg = opt_segs[0][0]
            place_x1, place_y1, place_x2, place_y2 = tuple(map(int, opt_seg.crop_region))

        for seg in segs[0]:
            x1, y1, x2, y2 = tuple(map(int, seg.crop_region))

            cropped_mask = repeat(torch.from_numpy(seg.cropped_mask).float(), "h w -> 1 h w 3")
            resized_mask = resizer.fit_resize_image(cropped_mask, **kwargs)[0].mean(-1).squeeze()
            resized_image = resizer.fit_resize_image(image[:, y1:y2, x1:x2, :], **kwargs)[0]
            resized_mask_nhwc = rearrange(resized_mask, "h w -> 1 h w 1")
            resized_image *= resized_mask_nhwc
            
            if opt_bg_mask is None:
                place_x1, place_y1, place_x2, place_y2 = x1, y1, x2, y2
            cx, cy = (place_x1 + place_x2) // 2, (place_y1 + place_y2) // 2
            cx += move_x; cy += move_y
            h, w = resized_mask.shape
            x1, y1, x2, y2 = cx-(w//2), cy-(h//2), cx+(w//2), cy+(h//2)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, frame_w), min(y2, frame_w)
            
            max_h, max_w = (y2-y1), (x2-x1)
            mask_canvas[y1:y2, x1:x2] = resized_mask[:max_h, :max_w]
            
            resized_mask_nhwc = resized_mask_nhwc[:, :max_h, :max_w, :]
            resized_image = resized_image[:, :max_h, :max_w, :3]
            image_canvas[:, y1:y2, x1:x2, :3] = (
                resized_image + 
                image_canvas[:, y1:y2, x1:x2, :3] * (1 - resized_mask_nhwc)
            )
        mask_canvas = grow_mask.expand_mask(mask_canvas, expand=expand_out_mask, tapered_corners=True)[0]
        return (image_canvas, mask_canvas)
    
NODE_CLASS_MAPPINGS = {
    "AppIO_StringInput": AppIO_StringInput,
    "AppIO_ImageInput": AppIO_ImageInput,
    "AppIO_StringOutput": AppIO_StringOutput,
    "AppIO_ImageOutput": AppIO_ImageOutput,
    "AppIO_IntegerInput": AppIO_IntegerInput,
    "AppIO_FitResizeImage": AppIO_FitResizeImage,
    "AppIO_ImageInputFromID": AppIO_ImageInputFromID,
    "AppIO_ResizeInstanceImageMask": AppIO_ResizeInstanceImageMask
}