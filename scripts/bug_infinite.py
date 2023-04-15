# Author: @avantcontra
# https://github.com/avantcontra/sd-webui-bug-infinite


import time

import modules.scripts as scripts
import gradio as gr
import os

import PIL
from PIL import Image, ImageDraw

from modules import images, processing, script_callbacks, devices
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state

import numpy as np
import random
import cv2
from PIL import Image, ImageDraw

class Script(scripts.Script):
    # Extension title in menu UI
    def title(self):
        return "Bug Infinite"

    # Decide to show menu in txt2img or img2img
    # - in "txt2img" -> is_img2img is `False`
    # - in "img2img" -> is_img2img is `True`
    #
    # below code always show extension menu
    def show(self, is_img2img):
        return is_img2img

    # Setup menu ui detail
    def ui(self, is_img2img):
        if not is_img2img:
            return None
        
        with gr.Accordion('Bug Infinite', open=True):
            with gr.Row():
                zoom_mode = gr.Radio(choices=["in", "out"], value="in", label="Zoom Mode")
                fps = gr.Slider(label="Video FPS", minimum=1, maximum=240, step=1, value=30)
                
            with gr.Row():
                # enabled = gr.Checkbox(label='Enable', value=False)
                output_path = gr.Textbox(label="Video Output Folder", value="BugInfinite", lines=1)
                mask_width = gr.Slider(label='Mask Width', info="Mask area to inpainting", minimum=1, maximum=256, step=1, value=128)
              
            with gr.Row():
                num_outpainting_steps = gr.Slider(label='Outpainting keyframes num', info="Keyframes generated from inpainting", minimum=1, maximum=120, value=30, step=1)
                num_interpol_frames = gr.Slider(label="Interpolation num", info="Interpolation frames num between every 2 keyframes", minimum=1, maximum=120, value=30, step=1)
            # with gr.Row():
                
        return [num_interpol_frames, num_outpainting_steps, output_path, fps, zoom_mode, mask_width]

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def run(self, p, num_interpol_frames, num_outpainting_steps, output_path, fps, zoom_mode, mask_width):
        def write_video(file_path, frames, fps, reversed = True):
            print("write_video",file_path)
            """
            Writes frames to an mp4 video file
            :param file_path: Path to output video, must end with .mp4
            :param frames: List of PIL.Image objects
            :param fps: Desired frame rate
            :param reversed: if order of images to be reversed (default = True)
            """
            if reversed == True:
                frames.reverse()

            w, h = frames[0].size
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            #fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
            #fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

            for frame in frames:
                np_frame = np.array(frame.convert('RGB'))
                cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
                writer.write(cv_frame)

            writer.release() 


        def shrink_and_paste_on_blank(current_image, mask_width):
            # print("shrink_and_paste_on_blank", mask_width)
            """
            Decreases size of current_image by mask_width pixels from each side,
            then adds a mask_width width transparent frame, 
            so that the image the function returns is the same size as the input. 
            :param current_image: input image to transform
            :param mask_width: width in pixels to shrink from each side
            """

            height = current_image.height
            width = current_image.width
           

            #shrink down by mask_width
            mask_height = round(height/width * mask_width)
            # print("current_image h w,", height, width)
            # print("mask_height width,", mask_height, mask_width)
            
            prev_image = current_image.resize((width-2*mask_width, height-2*mask_height))
            prev_image = prev_image.convert("RGBA")
            # print("prev_imag size", prev_image.size)
            prev_image = np.array(prev_image)
            # print("prev_imag shape", prev_image.shape)

            #create blank non-transparent image
            blank_image = np.array(current_image.convert("RGBA"))*0
            blank_image[:,:,3] = 1
            # print("blank_image shape", blank_image.shape)

            #paste shrinked onto blank
            blank_image[mask_height:height-mask_height,mask_width:width-mask_width,:] = prev_image
            prev_image = Image.fromarray(blank_image)

            return prev_image

        #@markdown DOWNLOAD MODEL WEIGHTS AND SET UP DIFFUSION PIPELINE <br><br>
        #@markdown Pick your favourite inpainting model:
       
        # print(p.image_mask)
        # print(p.mask_blur)
        # print(p.inpainting_fill)
        # print(p.inpaint_full_res)

        
        def dummy(images, **kwargs):
            return images, False
        p.safety_checker = dummy
        # p.enable_attention_slicing() #This is useful to save some memory in exchange for a small speed decrease.

        #@markdown FIND A GOOD CONCEPT FOR YOUR VIDEO: <br>
        #@markdown (Image output of this block will be the last image of the video)

        # prompt = p.prompt[0]
        # negative_prompt = p.negative_prompt[0]

        #@markdown Number of initial example images to generate:
        num_init_images = 1#len(p.init_images) #@param

        #@markdown Height (and width) of the images in pixels (= resolution of the video generated in the next block, has to be divisible with 8):
        width = p.width#image_width
        height = p.height#width
        #@markdown Since the model was trained on 512 images increasing the resolution to e.g. 1024 will
        #@markdown drastically reduce its imagination, so the video will vary a lot less compared to 512

        # current_image = PIL.Image.new(mode="RGBA", size=(height, width))
        
        current_image = p.init_images[0]
        current_image = current_image.convert("RGBA")
        mask_image = np.array(current_image)[:,:,3] 
        mask_image = Image.fromarray(255-mask_image).convert("RGB")
        current_image = current_image.convert("RGB")
        
        
        # p.image_mask=mask_image
        # p.steps=num_inference_steps
        # processed = process_images(pipe)
        init_images = [current_image]#processed.images

        # print(p)
        # print(p.mask_blur)
        # print(p.inpainting_fill)
        # print(p.inpaint_full_res)

        # image_grid(init_images, rows=1, cols=num_init_images)

        """We shrink the init image from the previous block and outpaint its outer frame using the same concept defined above (e.g. prompt, negative prompt, inference steps) but with a different seed. To generate an "inifinte zoom" video this is repeated **num_outpainting_steps** times and then rendered in reversed order.  
        
        To keep the outpainted part coherent and full of new content its width has to be relatively large (e.g. **mask_width** = 128 pixels if resolution is 512*512). 
        
        This on the other hand means that the generated video would be too fast and aestetically unpleasant. To slow down and smoothen the video we generate **num_interpol_frames** additional images between outpainted images using simple "interpolation".    

        Notes:    

        - Length of the video is proportional to num_outpainting_steps * num_interpol_frames.   
        - The time to generate the video is proportional to num_outpainting_steps.  
        - On a T4 GPU it takes about ~7 minutes to generate the video of width = 512, num_inference_steps = 20, num_outpainting_steps = 100. With fps = 24 and num_interpol_frames = 24 the video will be about 1:40 minutes long.

        """


        current_image = init_images[0]

        timestring = str(int(time.time()))
        p.outpath_samples = p.outpath_samples + "/" + output_path
        images.save_image(current_image, p.outpath_samples, timestring + "-init")

        all_frames = []
        all_frames.append(current_image)

        for i in range(num_outpainting_steps):
            print('Generating Outpainting keyframe: ' + str(i+1) + ' / ' + str(num_outpainting_steps))

            prev_image_fix = current_image

            prev_image = shrink_and_paste_on_blank(current_image, mask_width)
            # images.save_image(current_image, "H:/downf/testttt/samples", "prev_image")

            current_image = prev_image

            #create mask (black image with white mask_width width edges)
            mask_image = np.array(current_image)[:,:,3] 
            mask_image = Image.fromarray(255-mask_image).convert("RGB")
            # images.save_image(current_image, "H:/downf/testttt/samples", "mask_image")

            #inpainting step
            current_image = current_image.convert("RGB")
            # images.save_image(current_image, "H:/downf/testttt/samples", "current_image")   
           
            p.init_images=[current_image]
            # p.prompt=["colorful psychedelic mushroom forest, atmospheric, hyper realistic, epic composition, cinematic, DeviantArt"]*num_init_images
            # p.negative_prompt=["blur"]*num_init_images
            p.image_mask=mask_image

            processed = process_images(p)
          
            # print(p.prompt)
            # print(p.seed)
            # print(p.inpainting_mask_invert)
            # print(p.inpainting_fill) #funny to try different options
            # print(p.inpaint_full_res)
            current_image = processed.images[0]
            # images.save_image(current_image, p.outpath_samples, timestring + "-processed")

            p.seed = processed.seed + 1

            current_image.paste(prev_image, mask=prev_image)

            #interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
            mask_height = round(height/width * mask_width)
            for j in range(num_interpol_frames - 1):
                interpol_image = current_image
                interpol_width = round(
                    (1- ( 1-2*mask_width/width )**( 1-(j+1)/num_interpol_frames ) )*width/2 
                    )
                interpol_height = round(
                    (1- ( 1-2*mask_height/height )**( 1-(j+1)/num_interpol_frames ) )*height/2 
                    )
                interpol_image = interpol_image.crop((interpol_width,
                                                    interpol_height,
                                                    width - interpol_width,
                                                    height - interpol_height))

                interpol_image = interpol_image.resize((width, height))

                #paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                interpol_width2 = round(
                    ( 1 - (width-2*mask_width) / (width-2*interpol_width) ) / 2*width
                    )
                prev_image_fix_crop = shrink_and_paste_on_blank(prev_image_fix, interpol_width2)
                interpol_image.paste(prev_image_fix_crop, mask = prev_image_fix_crop)
                all_frames.append(interpol_image)
                # images.save_image(interpol_image, p.outpath_samples, timestring + "-processed")
                

            all_frames.append(current_image)
   
        write_video(p.outpath_samples + "/" + timestring + ".mp4", all_frames, fps, zoom_mode == "in")
      
        processed = Processed(p, [], p.seed, processed.info)

        return processed

