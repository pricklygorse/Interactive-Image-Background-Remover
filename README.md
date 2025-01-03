# Interactive Image Background Remover

A (work in progress) user interface for several background remover models, currently supporting onnx versions of u2net, disnet, rmbg, BiRefNet and interactive editing using Segment Anything (not V2 yet). Similar idea to Photoroom where you can just run the background remover model, or adjust the finer details by adding/removing points/areas/manual paintbrush.

Tested on Linux and Windows, but not Mac

You can use the python script or download executables from [https://github.com/pricklygorse/interactive-image-background-remover/actions?query=is%3Asuccess](https://github.com/pricklygorse/interactive-image-background-remover/actions?query=is%3Asuccess). Choose the most recent build, and scroll down to Artefacts

![Screenshot of main window](Images/main_image.jpg)

Load your image, and either run one of the whole image models (u2net, disnet, rmbg, BiRefNet) or click/draw a box to run Segment Anything. Left click is a positive point, right is a negative (avoid this area) point.

The original image is displayed on the left, the current image you are working on is displayed to the right.

Press A to add the current mask to your image, Z to remove.

Scroll wheel to zoom, and middle click to pan around the image. The models will be applied only to the visible zoomed image, which enables much higher detail and working in finer detail than just running the models on the whole image

Use manual paintbrush mode to draw areas that you want to add/remove to the image without using a model.

Post process mask removes the partial transparency from outputs of whole-image models. 

Includes a built-in image editor and cropper. Loading this will reset your current working image. 

![Screenshot of main window](Images/image_editor.jpg)

Running the script from the command line with multiple images specified will load the images one after another for assisted batch usage.

# Usage

## Running

`pip install` any missing packages or set up a new python environment, copy the background removal models you want to use into the Models folder (see model links below) then run `python backgroundremoval.py`. Any images supplied as arguments will be opened sequentially so you can work on the images in a batch, or it will open a file picker dialog to open a file if no arguements provided.

## Mouse

Left Mouse click: Add coordinate point for segment anything models

Right Mouse click: Add negative coordinate (area for the model to avoid)

Left click and drag: Draw box for segment anything models

## Hotkeys:

a : Add current mask to working image

z : Remove current mask from working image

q: Undo last action

p : Manual paintbrush mode

c : Clear current mask (and coordinate points)

w : Reset the current working image

r : Reset everything (image, masks, coordinates)

v : Clear the visible area on the working image

s : Save image as....

j : Quick save JPG with white background

_Run whole-image models (if downloaded to Models folder)_

u : u2net

i : disnet

o : rmbg

b : BiRefNet-general-bb_swin_v1_tiny-epoch_232



# Models

Background removal models in onnx format can be downloaded from these locations:

- Segment Anything + mobile-sam: [https://huggingface.co/vietanhdev/segment-anything-onnx-models/tree/main](https://huggingface.co/vietanhdev/segment-anything-onnx-models/tree/main)
- rembg: [https://huggingface.co/briaai/RMBG-1.4/tree/main/onnx](https://huggingface.co/briaai/RMBG-1.4/tree/main/onnx)
- u2net, disnet, BiRefNet, Segment Anything, and more: [https://github.com/danielgatis/rembg/releases/tag/v0.0.0](https://github.com/danielgatis/rembg/releases/tag/v0.0.0)

Place the models (or symlinks if located elsewhere) in the Models folder. The script checks for models at each start up. 

If using quantised Segment Anything models, these require the .quant suffix before .encoder in the filename, which is the opposite of how they are downloaded from the links above.

I highly recommend starting with mobile-sam as it has almost instantaneous mask generation even on older cpu-only computers, then trying the larger segment anything models if you need a higher quality mask. Only using mobile-sam and just zooming in when I need more detail however has been very effective for me.

The following models are hardcoded, simply add to this section if you want to include a different model.

``` python
# will also match quantised versions .quant
sam_models = [
            "mobile_sam",
            "sam_vit_b_01ec64", 
            "sam_vit_h_4b8939",
            "sam_vit_l_0b3195",
            ]

whole_models = [
        "rmbg1_4",
        "rmbg1_4-quant",
        "isnet-general-use",
        "isnet-anime",
        "u2net",
        "BiRefNet", # matches all birefnet variations
]
```



# Support Me

Find this useful and want to support my work? [You can buy me a coffee (or whatever) here.](https://ko-fi.com/pricklygorse) :)

I'm fairly new to python and tkinter so any improvements to the code, features, and suggestions are welcome. There are likely bugs.


# Acknowledgements

This was originally inspired by the command line program [RemBG by Daniel Gatis](https://github.com/danielgatis/rembg), and some of the inference code is adapted from this. 

Huge thanks to Meta for Segment Anything and all the other model authors for releasing their models. 
