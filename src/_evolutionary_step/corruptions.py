# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import random
from PIL import Image
import traceback
import albumentations as AUG
import cv2



def overly_img(img, foreground_image_path, alpha):

    img2 = Image.open(foreground_image_path).convert(img.mode)
    img2 = img2.resize(img.size)
    img = Image.blend(img, img2, alpha=alpha)

    return img

# /////////////// Distortions ///////////////

def ice(x, severity=1):
    x = Image.fromarray((x * 255).astype(np.uint8))
    foreground_image_path = ['img/ice2.png', 'img/ice2.png', 'img/ice3.png', 'img/ice3.png', 'img/ice4.png']
    alpha = [0.3, 0.4, 0.4, 0.5, 0.6]
    img = overly_img(x, foreground_image_path[severity-1], alpha[severity-1])
    return img


def rain_mask(x):
    x = Image.fromarray((x * 255).astype(np.uint8))
    foreground_image_path = 'img/rain.png'
    alpha = 0.15
    img = overly_img(x, foreground_image_path, alpha)
    return img


def pixel_trap(image, severity=1):
    levels = [int(image.shape[2] / 0.05), int(image.shape[2] / 0.03), int(image.shape[2] / 0.01)]
    
    indices = np.random.choice(image.shape[0], levels[severity-1], replace=False)
    image[indices] = 0

    return image * 255


def row_add_logic(image, severity=1):
    #levels = [int(image.shape[2] / 0.05), int(image.shape[2] / 0.04), int(image.shape[2] / 0.03), int(image.shape[2] / 0.02), int(image.shape[2] / 0.01)]
    levels = int(image.shape[2] / severity)
    ind = int(image.shape[0]/2)
    for i in range(1, levels+1):#levels[severity-1]+1):
        image[ind+i] = image[ind]
                
    return image * 255
        
        
def shifted_pixel(image, severity=1):
    #levels = [int(image.shape[2] / 0.5) , int(image.shape[2] / 0.35), int(image.shape[2] / 0.3), int(image.shape[2] / 0.25), int(image.shape[2] / 0.1)]
    levels = int(image.shape[2] / severity)
    max_shift = levels#levels[severity-1]
    m,n = image.shape[0], image.shape[1]
    col_start = np.random.randint(0, max_shift, image.shape[0])
    idx = np.mod(col_start[:,None] + np.arange(n), n)
    image = image[np.arange(m)[:,None], idx]

    return image * 255


def broken_lens(x, severity=1):
    x = Image.fromarray((x * 255).astype(np.uint8))
    foreground_image_path_list = {'img/broken1.png', 'img/broken2.png'}
    foreground_image_path = random.choice(tuple(foreground_image_path_list))

    alpha = severity#[0.4, 0.5, 0.6, 0.7, 0.8]
    img = overly_img(x, foreground_image_path, alpha)
    return img


def apply_novelty(RGBA_x, frame_num_novelty):
    scale_factor = [i for i in range(50, 0, -1)]
    scale_factor.sort(reverse=True)
    size_index = scale_factor[frame_num_novelty]

    def paste_img_rgba(bg, fg):
        text_img = Image.new('RGBA', (bg.width,bg.height), (0, 0, 0, 0))
        text_img.paste(bg,((text_img.width - bg.width) // 2, (text_img.height - bg.height) // 2))
        text_img.paste(fg, ((text_img.width - fg.width) // 2, (text_img.height - fg.height) // 2), mask=fg.split()[3])
        return text_img

    foreground_image_path = ['img/fallen_tree.png']# 'img/crashed_car.png'
    foreground_image = foreground_image_path[0]    

    #these 4 lines of code are just to avoid that the object approaches the camera too fast
    # slow_approach_lvl = [1, 2] # increase slow_approach_lvl to make the approach slower
    # for l in range(slow_approach_lvl[0]):
    #     scale_factor2 = [i for i in range(20, 0, -1)]
    #     scale_factor = scale_factor+scale_factor2

    #print(scale_factor)
    
    print(size_index)
    
    background_img = Image.fromarray((RGBA_x).astype(np.uint8))
    
    img2 = Image.open(foreground_image).convert('RGBA')
    
    basewidth = int(img2.size[0]/size_index)
    
    wpercent = (basewidth/float(img2.size[0]))
    hsize = int((float(img2.size[1])*float(wpercent)))
    img2 = img2.resize((basewidth,hsize), Image.ANTIALIAS)

    img_rgba = paste_img_rgba(background_img, img2)

    img_rgba = np.array(img_rgba)
    modified_image = img_rgba[:, :, :3]
    modified_image = modified_image[:, :, ::-1]

    return modified_image


def apply_anomaly(RGBA_x, frame_num_novelty):
    scale_factor = [i for i in range(50, 0, -1)]
    scale_factor.sort(reverse=True)
    size_index = scale_factor[frame_num_novelty]

    def paste_img_rgba(bg, fg):
        text_img = Image.new('RGBA', (bg.width,bg.height), (0, 0, 0, 0))
        text_img.paste(bg,((text_img.width - bg.width) // 2, (text_img.height - bg.height) // 2))
        text_img.paste(fg, ((text_img.width - fg.width) // 2, (text_img.height - fg.height) // 2), mask=fg.split()[3])
        return text_img

    foreground_image_path = ['img/crashed_car.png']# it's an array if you want to add more images
    foreground_image = foreground_image_path[0]    

    scale_factor = [i for i in range(50, 0, -1)]

    #these 4 lines of code are just to avoid that the object approaches the camera too fast
    slow_approach_lvl = [1, 2] # increase slow_approach_lvl to make the approach slower
    # for l in range(slow_approach_lvl[1]):
    #     scale_factor2 = [i for i in range(20, 0, -1)]
    #     scale_factor = scale_factor+scale_factor2

    scale_factor.sort(reverse=True)

    #print(scale_factor)
    #print(frame_num)
    size_index = scale_factor[frame_num]
    #print(size_index)
    
    background_img = Image.fromarray((RGBA_x).astype(np.uint8))
    
    img2 = Image.open(foreground_image).convert('RGBA')
    
    basewidth = int(img2.size[0]/size_index)
    
    wpercent = (basewidth/float(img2.size[0]))
    hsize = int((float(img2.size[1])*float(wpercent)))
    img2 = img2.resize((basewidth,hsize), Image.ANTIALIAS)

    img_rgba = paste_img_rgba(background_img, img2)

    img_rgba = np.array(img_rgba)
    modified_image = img_rgba[:, :, :3]
    modified_image = modified_image[:, :, ::-1]

    return modified_image


def apply_threats(img, aug_type, severity):

    transform = None

    if aug_type == 'shifted_pixel':
        image=shifted_pixel(img, severity)
        return np.array(image)

    elif aug_type == 'row_add_logic':
        image=row_add_logic(img, severity)
        return np.array(image)

    elif aug_type == 'broken_lens':
        severity = int(severity)
        image=broken_lens(img, severity)
        return np.array(image)

    elif aug_type == 'condensation':
        x = Image.fromarray((img * 255).astype(np.uint8))
        foreground_image_path = 'img/condensation1.png'
        alpha = severity#[0.2, 0.3, 0.4, 0.5, 0.6]
        img = overly_img(x, foreground_image_path, alpha)
        return img

    elif aug_type == 'dirty':
        x = Image.fromarray((img * 255).astype(np.uint8))
        foreground_image_path = 'img/dirty.png'
        alpha = severity#[0.4, 0.5, 0.6, 0.7, 0.8]
        img = overly_img(x, foreground_image_path, alpha)
        return img

    elif aug_type == 'sun_flare':
        flare_roi_list = {(0.25, 0.25, 0.55, 0.55), (0.55, 0.55, 0.75, 0.75), (0.70, 0.70, 0.90, 0.90)}
        flare_roi = random.choice(tuple(flare_roi_list))
        #flare_roi = (severity, severity, severity+0.25, severity+0.25)
        #[(0.25, 0.25, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75), (0.5, 0.5, 0.75, 0.75), (0.5, 0.5, 1, 1), (0.75, 0.75, 1, 1)]
        #src_radius_list = {400, 500, 600, 700, 800}
        src_radius = severity #random.choice(tuple(src_radius_list))

        try:
            transform = AUG.RandomSunFlare(flare_roi=flare_roi, 
                              angle_lower=0, angle_upper=1, 
                              num_flare_circles_lower=6, 
                              num_flare_circles_upper=10, 
                              src_radius=int(src_radius), src_color=(255, 255, 255),
                              always_apply=True)
        except Exception as e:
            # transform = AUG.RandomSunFlare(flare_roi=(0, 0, 0.1, 0.1), 
            #                   angle_lower=0, angle_upper=1, 
            #                   num_flare_circles_lower=6, 
            #                   num_flare_circles_upper=10, 
            #                   src_radius=400, src_color=(255, 255, 255),
            #                   always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'snow':
        severity = int(severity)
        brightness_coeff = severity#[1,2,3,4,5]
        try:
            transform = AUG.RandomSnow(snow_point_lower=0, 
                          snow_point_upper=0.5, 
                          brightness_coeff=brightness_coeff, always_apply=True)
        except Exception as e:
            # transform = AUG.RandomSnow(snow_point_lower=0, 
            #               snow_point_upper=0.5, 
            #               brightness_coeff=1, always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'smoke':
        alpha_coef = severity#[0.01, 0.05, 0.1, 0.2, 0.3]

        try:
            transform = AUG.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=alpha_coef, always_apply=True)
        except Exception as e:
            # transform = AUG.RandomFog(fog_coef_lower=0, fog_coef_upper=0, alpha_coef=0.0, always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'rain':
        mask=False
        rain_types = ['drizzle', 'heavy', 'torrential', 'heavy_rain_mask', 'torrential_rain_mask']
        severity = int(severity)
        try:
            if rain_types[severity-1] == 'heavy_rain_mask':
                mask=True
                rain_types[severity-1] = 'heavy'
                
            elif rain_types[severity-1] == 'torrential_rain_mask':
                mask=True
                rain_types[severity-1] = 'torrential'
                
            transform = AUG.RandomRain(slant_lower=-10, slant_upper=10, 
                          drop_length=20, drop_width=1, drop_color=(200, 200, 200), 
                          blur_value=7, brightness_coefficient=0.7, 
                          rain_type=rain_types[severity-1], always_apply=True)
            if mask:    
                image = transform(image=img)['image']
                #image *= 255
                image=rain_mask(image)
                return np.array(image)

        except Exception as e:
            # transform = AUG.RandomRain(slant_lower=-10, slant_upper=10, 
            #                   drop_length=20, drop_width=1, drop_color=(200, 200, 200), 
            #                   blur_value=7, brightness_coefficient=0.7, 
            #                   rain_type=None, always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'brightness':
        brightness_limit = (severity, severity)#[(0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8)]
        #print('brightness_limit', brightness_limit[severity-1])
        try:
            transform = AUG.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=(0.0,0.0), always_apply=True)
        except Exception as e:
            # transform = AUG.RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.0,0.0), always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'contrast':
        contrast_limit = (severity, severity)#[(0.5, 0.5), (0.6, 0.6), (0.7, 0.7), ]
        try:
            transform = AUG.RandomBrightnessContrast(brightness_limit=(0.0,0.0), contrast_limit=contrast_limit, always_apply=True)
        except Exception as e:
            # transform = AUG.RandomBrightnessContrast(brightness_limit=(0.0,0.0), contrast_limit=(0, 0), always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'channel_dropout':
        severity = int(severity)
        channel_drop_range = (severity, severity)#[(1,1), (1,1), (2,2), (2,2), (2,2)]
        #fill_values = {0, 255, 0, 255, 122}#[0, 255, 0, 255, 122]
        #fill_value = random.choice(tuple(fill_values))

        try:
            transform = AUG.ChannelDropout(channel_drop_range=channel_drop_range, #fill_value=fill_value, 
                always_apply=True)
        except Exception as e:
            # transform = AUG.ChannelDropout(always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'channel_shuffle':
        transform = AUG.ChannelShuffle(always_apply=True)

    elif aug_type == 'gaussian_blur':
        #blur_limit = [(3, 7), (8, 13), (14, 20), (21, 28), (29, 37)]
        severity = int(severity)
        blur_limit = (severity, severity+4)
        
        try:
            transform = AUG.GaussianBlur(blur_limit=blur_limit, always_apply=True)
        except Exception as e:
            # transform = AUG.GaussianBlur(blur_limit=(0, 0), always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'gaussian_noise':
        #var_limit = [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.7, 0.7)]
        var_limit = (severity, severity)
        try:
            transform = AUG.GaussNoise(var_limit=var_limit, mean=0, always_apply=True)
        except Exception as e:
            # transform = AUG.GaussNoise(var_limit=(0, 0), mean=0, always_apply=True)
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'coarse_dropout':
        # holes = [10, 30, 40, 50]
        # height = [10, 20, 25, 30]
        # width = [10, 20, 25, 30]
        severity = int(severity)
        holes = severity+5
        height = severity
        width = severity

        try:
            transform = AUG.CoarseDropout(max_holes=holes, max_height=height, max_width=width, 
                             min_holes=holes, min_height=height, min_width=width,
                             always_apply=True) 
        except Exception as e:
            # transform = AUG.CoarseDropout(always_apply=True) # default values
            print('error',e)
            traceback.print_exc()

    elif aug_type == 'grid_dropout':
        # u_min = [10, 25, 50, 66, 75]
        # u_max = [10, 25, 50, 66, 75]
        # h_x = [1, 3, 5, 7, 10]
        # h_y = [1, 3, 5, 7, 10]
        severity = int(severity) # [2, 5, 7]
        u_min = severity*10
        u_max = severity*10
        h_x, h_y = severity, severity

        try:
            transform = AUG.GridDropout(
                           unit_size_min=u_min,#[severity-1], 
                           unit_size_max=u_max,#[severity-1], 
                           holes_number_x=h_x,#[severity-1], 
                           holes_number_y=h_y,#[severity-1], 
                           always_apply=True
                           )
        except Exception as e:
            # transform = AUG.GridDropout(
            #                unit_size_min=1, 
            #                unit_size_max=1, 
            #                holes_number_x=1, 
            #                holes_number_y=1, 
            #                always_apply=True
            #                )
            print('error',e)
            traceback.print_exc()

    try:
        shapee = img.shape[2]
    except:
        print('no support for grey scale imgs at this time')
        return img * 255
    
    image = transform(image=img)['image']
    return image * 255


def apply_mult_threats(incoming_image, individual):
    
    for gene in individual:
        incoming_image = incoming_image / 255
        incoming_image = np.array(incoming_image, dtype=np.float32)
        aug_type, severity = gene[0], gene[1]
        #print('aug type and severity', aug_type, severity)
        
        #severity = 0 means "no transformation"
        if severity>0:
            incoming_image = apply_threats(incoming_image, aug_type, severity)

    return incoming_image


def parallelize_transformations(tasks, num_cpu=2):
    #from multiprocessing import Process
    s = int(len(tasks)/num_cpu) # we recommend to use divisible values