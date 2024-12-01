from PIL import Image, ImageDraw
import numpy as np
import torch
from collections import namedtuple
from einops import repeat, rearrange
import torchvision.transforms.functional as TF
from dataclasses import dataclass

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
            },
            "optional" : { "opt_mask": ("MASK",)}
        }

    RETURN_TYPES = ("IMAGE","INT","INT","FLOAT", "MASK")
    RETURN_NAMES = ("Image","Fit Width", "Fit Height", "Aspect Ratio", "Mask")
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

    def fit_resize_image(self, image, max_size=768, resampling="bicubic", upscale="false", opt_mask=None):
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
        resized_image = self.pil2tensor(resized_image)

        resized_mask = torch.zeros_like(resized_image)[...,0]
        if opt_mask is not None:
            mask_img = self.tensor2pil(repeat(opt_mask, "n h w -> n h w c", c=3))
            resized_mask = mask_img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))
            resized_mask = self.pil2tensor(resized_mask).mean(-1)
        
        return (resized_image,new_width,new_height,aspect_ratio,resized_mask)

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = [int(x) for x in (x1, y1, x2, y2)]

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return self.x1 + self.w // 2

    @property
    def cy(self):
        return self.y1 + self.h // 2

    def update(self, dw, dh):
        """Expand or shrink the bounding box by dw and dh."""
        self.x1, self.x2 = max(self.x1 - dw, 0), self.x2 + dw
        self.y1, self.y2 = max(self.y1 - dh, 0), self.y2 + dh

    def adjust_to_center(self, cx, cy, new_width, new_height, max_w, max_h):
        """Recenter and constrain the bounding box within image dimensions."""
        self.x1, self.x2 = cx - (new_width // 2), cx + (new_width // 2)
        self.y1, self.y2 = cy - (new_height // 2), cy + (new_height // 2)
        self.x1, self.x2 = max(self.x1, 0), min(self.x2, max_w)
        self.y1, self.y2 = max(self.y1, 0), min(self.y2, max_h)

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
                opt_bg_mask=create_type("MASK"),
                opt_mask_shape=create_type(['none', 'circle', 'square', 'triangle'])
            )
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize"
    CATEGORY = "AppIO"

    #Ref: https://github.com/kijai/ComfyUI-KJNodes/blob/1dbb38d63dd9769b15a85c4c527391f955242568/nodes/mask_nodes.py#L800
    def create_shape_mask(self, shape, width, height):
        image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(image)
        color = 'white'

        location_x = width // 2
        location_y = height // 2

        if shape == 'circle' or shape == 'square':
            # Define the bounding box for the shape
            left_up_point = (location_x - width // 2, location_y - height // 2)
            right_down_point = (location_x + width // 2, location_y + height // 2)
            two_points = [left_up_point, right_down_point]

            if shape == 'circle':
                draw.ellipse(two_points, fill=color)
            elif shape == 'square':
                draw.rectangle(two_points, fill=color)
                
        elif shape == 'triangle':
            # Define the points for the triangle
            left_up_point = (location_x - width // 2, location_y + height // 2) # bottom left
            right_down_point = (location_x + width // 2, location_y + height // 2) # bottom right
            top_point = (location_x, location_y - height // 2) # top point
            draw.polygon([top_point, left_up_point, right_down_point], fill=color)

        return torch.from_numpy(np.array(image)).float().__div__(255.)[:, :, 0]

    def resize(self, image, mask, expand_out_mask: int, move_x: int, move_y: int, opt_bg_image=None, opt_bg_mask=None, opt_mask_shape='none', **kwargs):
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
            seg: SEG = seg
            x1, y1, x2, y2 = tuple(map(int, seg.crop_region))

            cropped_mask = repeat(torch.from_numpy(seg.cropped_mask).float(), "h w -> 1 h w 3")
            resized_mask = resizer.fit_resize_image(cropped_mask, **kwargs)[0].mean(-1).squeeze()
            resized_image = resizer.fit_resize_image(image[:, y1:y2, x1:x2, :], **kwargs)[0]
            resized_mask_nhwc = rearrange(resized_mask, "h w -> 1 h w 1")
            resized_image *= resized_mask_nhwc
            if opt_mask_shape != 'none':
                cropped_mask = repeat(self.create_shape_mask(opt_mask_shape, x2-x1, y2-y1), "h w -> 1 h w 3")
                resized_mask = resizer.fit_resize_image(cropped_mask, **kwargs)[0].mean(-1).squeeze()
            
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

class AppIO_ResizeInstanceAndPaste:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": dict(
                input_image=create_type("IMAGE"),
                input_segs=create_type("SEGS"),
                bg_image=create_type("IMAGE"),
                bg_image_segs=create_type("SEGS"),
                scale_factor=create_type("FLOAT", min=0.1, max=5, step=0.1, default=1.),
                move_x=create_type("INT", min=-1024, max=1024, step=1, default=0),
                move_y=create_type("INT", min=-1024, max=1024, step=1, default=0),
                
                pad_width=create_type("INT", min=0, max=1024, step=1, default=64),
                pad_height=create_type("INT", min=0, max=1024, step=1, default=64),
                
                expand_out_mask=create_type("INT", min=-1024, max=1024, step=1, default=0)
            )
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize_paste"
    CATEGORY = "AppIO"
    
    def resize_paste(self, input_image, input_segs, bg_image, bg_image_segs, 
                     scale_factor: int, move_x: int, move_y: int, pad_width: int, pad_height: int, expand_out_mask: int):
        from nodes import NODE_CLASS_MAPPINGS
        segs_to_mask = NODE_CLASS_MAPPINGS["SegsToCombinedMask"]()
        grow_mask = NODE_CLASS_MAPPINGS["GrowMask"]()

        dx, dy = move_x // 2, move_y // 2
        dw, dh = pad_width // 2, pad_height // 2

        # Extract primary segmentation masks
        _, *input_segs = input_segs
        _, *bg_segs = bg_image_segs
        input_seg = input_segs[0][0]
        bg_seg = bg_segs[0][0]

        # Initialize bounding boxes
        inp = BBox(*input_seg.crop_region)
        bg = BBox(*bg_seg.crop_region)

        # Compute scale factor and resize input_bbox
        scale_factor *= max(bg.w / inp.w, bg.h / inp.h)
        inp.update(dw, dh)
        new_height = int(inp.h * scale_factor // 2 * 2)
        new_width = int(inp.w * scale_factor // 2 * 2)

        # Process input image
        input_image = rearrange(input_image, "b h w c -> b c h w").clone()
        resized_crop = TF.resize(
            input_image[:, :, inp.y1:inp.y2, inp.x1:inp.x2],
            (new_height, new_width)
        )
        resized_crop = rearrange(resized_crop, "b c h w -> b h w c").clone()

        # Prepare result image
        result_image = bg_image.clone()
        max_h, max_w = result_image.shape[1:3]

        # Adjust bg_bbox and paste resized_crop
        bg.adjust_to_center(bg.cx + dx, bg.cy + dy, new_width, new_height, max_w, max_h)
        result_image[:, bg.y1:bg.y2, bg.x1:bg.x2, :] = resized_crop[
            :, :bg.y2 - bg.y1, :bg.x2 - bg.x1
        ]

        # Generate result mask
        result_mask = grow_mask.expand_mask(
            segs_to_mask.doit(bg_image_segs)[0],
            expand=expand_out_mask,
            tapered_corners=True
        )[0]

        return result_image, result_mask


NODE_CLASS_MAPPINGS = {
    "AppIO_StringInput": AppIO_StringInput,
    "AppIO_ImageInput": AppIO_ImageInput,
    "AppIO_StringOutput": AppIO_StringOutput,
    "AppIO_ImageOutput": AppIO_ImageOutput,
    "AppIO_IntegerInput": AppIO_IntegerInput,
    "AppIO_FitResizeImage": AppIO_FitResizeImage,
    "AppIO_ImageInputFromID": AppIO_ImageInputFromID,
    "AppIO_ResizeInstanceImageMask": AppIO_ResizeInstanceImageMask,
    "AppIO_ResizeInstanceAndPaste": AppIO_ResizeInstanceAndPaste
}