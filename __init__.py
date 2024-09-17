
def create_type(type, **kwargs):
    return (type, kwargs)

class AppIO_StringInput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            x=create_type("STRING", default='', multiline=True),
            argument_name=create_type("STRING", default='1', multiline=False)
        ))
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "identity"

    def identity(x):
        return (x,)
    
class AppIO_ImageInput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(x=create_type("IMAGE")))
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "identity"

    def identity(x):
        return (x,)

class AppIO_StringOutput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            x=create_type("STRING", default='', multiline=True)
        ))
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "identity"
    OUTPUT_NODE = True

    def identity(x):
        return (x,)

class AppIO_ImageOutput:
    @classmethod
    def INPUT_TYPES(s):
        return dict(required=dict(
            x=create_type("STRING", default='', multiline=True)
        ))
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "identity"
    OUTPUT_NODE = True

    def identity(x):
        return (x,)
    
NODE_CLASS_MAPPINGS = {
    "AppIO_StringInput": AppIO_StringInput,
    "AppIO_ImageInput": AppIO_ImageInput,
    "AppIO_StringOutput": AppIO_StringOutput,
    "AppIO_ImageOutput": AppIO_ImageOutput
}