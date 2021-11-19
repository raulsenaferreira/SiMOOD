#!../venv/bin/python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).

from __future__ import print_function
from original_manual_control import (World, HUD, KeyboardControl, CameraManager, CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor)
from typing import *
from multiprocessing.connection import Client
import logging
import time
import pygame
import argparse
import json
import subprocess
import signal
import h5py
import numpy as np
import datetime
import shutil
import glob
import os
import sys
import cv2 as cv
# ================================ custom ==============================================
from bounding_box_extractor import get_2d_bounding_box, get_class_from_actor_type
from safety_monitors import baseline_safety_monitors as SM_base
import corruptions
import AEBS
import data_utils
import draw_utils
import ML_functions
import evaluation_module

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

SCENARIO_PROCESS = None
LOG_PERCEPTION_PATH = "src/log/perception.csv"
LOG_PERCEPTION_TXT = 'Exception during object image transformation for {} using {}: {} \n'
CARLA_EGG_PATH = '../carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg'

def setup_weather(world, args):
    weather = world.get_weather()

    if args.rain == 'none':
        weather.precipitation = 0
        weather.precipitation_deposits = 0
        weather.wetness = 0
    elif args.rain == 'light':
        weather.precipitation = 20
        weather.precipitation_deposits = 20
        weather.wetness = 20
    elif args.rain == 'medium':
        weather.precipitation = 50
        weather.precipitation_deposits = 50
        weather.wetness = 50
    elif args.rain == 'heavy':
        weather.precipitation = 100
        weather.precipitation_deposits = 100
        weather.wetness = 100

    # Note: Fog seems to be only working properly with vulkan renderer
    if args.fog == 'none':
        weather.fog_density = 0
        weather.fog_distance = 0
        weather.fog_falloff = 0
    elif args.fog == 'light':
        weather.fog_density = 20
        weather.fog_distance = 0
        weather.fog_falloff = 0.2
    elif args.fog == 'medium':
        weather.fog_density = 50
        weather.fog_distance = 0
        weather.fog_falloff = 0.5
    elif args.fog == 'heavy':
        weather.fog_density = 100
        weather.fog_distance = 0
        weather.fog_falloff = 1

    if args.clouds == 'none':
        weather.cloudiness = 0
    elif args.clouds == 'light':
        weather.cloudiness = 20
    elif args.clouds == 'medium':
        weather.cloudiness = 50
    elif args.clouds == 'heavy':
        weather.cloudiness = 100

    if args.time_of_day == 'day':
        weather.sun_altitude_angle = 60
        weather.sun_azimuth_angle = 10
    elif args.time_of_day == 'sunset':
        weather.sun_altitude_angle = 0.5
        weather.sun_azimuth_angle = 180
    elif args.time_of_day == 'night':
        weather.sun_altitude_angle = -90
        weather.sun_azimuth_angle = 0

    weather.wind_intensity = 0

    world.set_weather(weather)


def calculate_camera_calibration(image_width, image_height, fov):
    calibration = np.identity(3)
    calibration[0, 2] = image_width / 2.0
    calibration[1, 2] = image_height / 2.0
    calibration[0, 0] = calibration[1, 1] = image_width / (2.0 * np.tan(float(fov) * np.pi / 360.0))

    return calibration


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def numpy_rgba2rgb(array):
    """
    Receives a numpy array [w, h, 4] pixel (alpha)
    Returns a numpy array [w, h, 3] without alpha
    """
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    return array

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

# -- find carla module 
try:
    #sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob(CARLA_EGG_PATH)[0])
except IndexError:
    print("IndexError")
    pass

import carla

class WorldSR(World):

    restarted = False

    def restart(self):

        if self.restarted:
            return
        self.restarted = True

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    print("Ego vehicle found")
                    self.player = vehicle
                    break
        
        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    
    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)


    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False

        self.hud.tick(self, clock)
        return True

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    if not args.no_recording:
        # IKS: Create the directory to store all sensor and meta data
        if os.path.exists(args.output_dir):
            logging.error('Output directory already exists. Please specify another non-existing directory in the cli.')
            return
        else:
            os.makedirs(args.output_dir)

        # IKS: Store the configuration for this scenario
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as config_file:
            config = dict(vars(args))

            # Remove all non-scenario related arguments
            for irrelevant_key in ['debug', 'host', 'port', 'autopilot', 'res', 'rolename', 'filter']:
                config.pop(irrelevant_key, None)

            json.dump(config, config_file)

        # IKS: Copy the scenario in the output folder
        shutil.copy(args.scenario_path, os.path.join(args.output_dir, 'scenario.xosc'))

    try:
        global SCENARIO_PROCESS
        SCENARIO_PROCESS = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scenario_runner.py'),
             '--openscenario', args.scenario_path,
             '--reloadWorld',
             '--timeout', '2000'],
            env=os.environ.copy())

        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        # hud.toggle_info()  # Hide HUD
        # hud.notification = lambda *args, **kwargs: None  # Disable HUD notifications
        hud.render = lambda *args, **kwargs: None  # Disable HUD rendering
        world = WorldSR(client.get_world(), hud, args)
        # controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()

        # IKS: Setup weather
        time.sleep(2)  # To safely overwrite weather set by OpenScenario
        setup_weather(world.world, args)

        # IKS: Switch to synchronous mode to ensure synchrony between the world and the sensors
        settings = world.world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05  # 1 / fps
        world.world.apply_settings(settings)

        # IKS: Switch to first person camera
        world.camera_manager.toggle_camera()

        # IKS: Setup the DVS (RGB camera is initialized by default by the camera manager)
        if 'event_camera' in args.sensors:
            world.dvs = world.world.spawn_actor(world.camera_manager.sensors[7][-1],  # Blueprint of the DVS
                                                carla.Transform(carla.Location(x=0, y=0, z=0)),
                                                attach_to=world.camera_manager.sensor,
                                                attachment_type=carla.AttachmentType.Rigid)

        # IKS: Setup the semantic segmentation sensor
        world.semantic_seg = world.world.spawn_actor(world.camera_manager.sensors[5][-1],
                                                     carla.Transform(carla.Location(x=0, y=0, z=0)),
                                                     attach_to=world.camera_manager.sensor,
                                                     attachment_type=carla.AttachmentType.Rigid)
        #if not args.no_recording:
        # IKS: Store event camera data
        if 'event_camera' in args.sensors:
            events_queue = []
            def save_dvs(events: carla.libcarla.DVSEventArray):
                # Save the raw events
                events_queue.append(events)

            world.dvs.listen(lambda events: save_dvs(events))

        # # IKS: Store semantic segmentation data
        semantic_seg_queue = []
        def save_semantic_seg_camera(image: carla.libcarla.Image):
            semantic_seg_queue.append(image)

        world.semantic_seg.listen(lambda image: save_semantic_seg_camera(image))

        # IKS: Store rgb camera data
        image_queue = []
        #def save_rgb_camera(image: carla.libcarla.Image):
        #    image_queue.append(image)

        #world.camera_manager.injected_listener = save_rgb_camera

        # IKS: Store actor states
        world_state_queue = []

        # Storing images for saving as RGB images in the end of simulation
        image_queue = []
        # How much frames should be considered for safety monitoring using a temporal approach
        frame_interval = 1
        frame_num = 0

        array_data = []
        
        WARNING_AREA = None

        #loading obj detector model
        ml_model = ML_functions.ML_object(args.object_detector_model_type)
        object_detector = ml_model.load_model()

        #for evaluation purposes
        true_pos_SUT, true_neg_SUT, false_pos_SUT, false_neg_SUT = [], [], [], []
        show_ground_truth_bbox_screen = False # True if you want to show the ground truth bboxes
        EVALUATE = True
        
        ################################################## stream looping ##################################################
        while True:
            world.world.tick()
            clock.tick_busy_loop()

            # if controller.parse_events(client, world, clock):
            #     return

            world.render(display)
                
            if world.camera_manager.surface is not None:
                incoming_image = world.camera_manager.np_image #RGBA
                frame_num += 1

                ########################################## THREATS on the fly ###########################################
                if args.fault_type != 'none':                                                                           #
                    if args.fault_type == 'novelty' or args.fault_type == 'anomaly':                                    #
                        incoming_image = corruptions.apply_novelty(incoming_image, int(args.severity), frame_num)       #
                        incoming_image = np.array(incoming_image, dtype=np.float32)                                     #
                    else:                                                                                               #
                        incoming_image = corruptions.apply_threats(incoming_image, args.fault_type, int(args.severity)) #
                else:                                                                                                   #
                    incoming_image = numpy_rgba2rgb(world.camera_manager.np_image)                                      #
                ########################################## THREATS on the fly ###########################################
                
                image_queue.append(incoming_image)

                # rendering the image
                real_time_view = pygame.surfarray.make_surface(incoming_image.swapaxes(0, 1))
                world.camera_manager.surface = real_time_view
                #display.blit(real_time_view, (0, 0))
                
                ########################################## Safety Monitor ###################################
                # VERIFY THE INPUTS: image conditions and correct the image if necessary                    #
                SM = SM_base.Safety_monitor(incoming_image, verification='pre')                             #
                incoming_image, has_reacted = SM.run()                                                      #
                ########################################## Safety Monitor ###################################

                #incoming_image = cv.cvtColor(incoming_image, cv.COLOR_BGR2RGB)
                
                ########################################## ML model #########################################
                predictions = None                                                                          #
                try:                                                                                        #
                    predictions = ml_model.make_predictions(incoming_image)                                 #
                                                                                                            #
                except Exception as e:                                                                      #
                    with open(LOG_PERCEPTION_PATH, "a") as myfile:                                          #
                        myfile.write(LOG_PERCEPTION_TXT.format(str(args.fault_type),                        #
                            str(args.object_detector_model_type), str(e)))                                  #
                ########################################## ML model #########################################

                view_width, view_height = int(args.res.split('x')[0]), int(args.res.split('x')[1])

                # drawing boxes from predictions. Variable detected_objects is an array of bboxes ...
                # (already converted in a good format to be used in pygame functions), scores, and labels 
                if predictions is not None:
                    detected_objects_per_frame, safety_surface, WARNING_AREA, DANGER_AREA = draw_utils.draw_bboxes(
                        world, carla, world.camera_manager, display, view_width, view_height, predictions, args)
                    
                    array_data.append(detected_objects_per_frame) # to be used in temporal functions

                if frame_num == frame_interval:
                    frame_num = 0

                    ########################################## Safety Monitor #######################################
                    # VERIFY THE OUTPUTS: image conditions and correct the image if necessary                       #
                    # temporal monitoring (monitors the coherence of the ML predictions)                            #
                    dummy_surface = pygame.Surface((view_width, view_height)) #just for polygon calculation purposes#
                    SM = SM_base.Safety_monitor((array_data, dummy_surface), verification='post')                   #
                    array_data, has_reacted = SM.run()                                                              #
                    ########################################## Safety Monitor #######################################

                    ################################################## EMERGENCY BREAKING SYSTEM ################################################
                    # it is also applied in a small set of frames. To apply frame by frame just change "frame_interval=1"                       #
                    AEBS_activated = AEBS.emergency_braking(world, carla, display, safety_surface, WARNING_AREA, DANGER_AREA, array_data, args) #
                    ################################################## EMERGENCY BREAKING SYSTEM ################################################

                    array_data = [] 

                #################################
                # initialize the safety monitor object
                # verifying threats after the perception system decision

                # #we perform a monitoring according to a specific object
                # target_class = 'person' 
                # # array_data_per_frame it's an array of objects representing the target class
                # array_data_per_frame = data_utils.compose_data(predictions, args, array_data, target_class, view_width, view_height)

                # if len(array_data_per_frame) > 0:
                #     array_data.append(array_data_per_frame)

                # # temporal monitoring functions (monitors the functionality)
                # if frame_num == frame_interval:
                #     frame_num = 0
                #     if len(array_data) > 0 and AEBS_activated==False and LOCK==False:
                #         dummy_surface = pygame.Surface((view_width, view_height)) #just for polygon calculation purposes
                #         SM = SM_base.Safety_monitor((array_data, dummy_surface), verification='post')
                #         _, has_reacted = SM.run() 
                        
                #         array_data = []
                #     elif AEBS_activated:
                #         LOCK = AEBS_activated
                #print('has_reacted:', has_reacted)
                #################################

                ############################### SUT evaluation ###############################
                if EVALUATE:
                    image = semantic_seg_queue[frame_num]
                    # Convert semantic segmentation image to city scapes color palette
                    image.convert(carla.ColorConverter.CityScapesPalette)

                    true_pos_SUT_per_frame, true_neg_SUT_per_frame, false_pos_SUT_per_frame, false_neg_SUT_per_frame = evaluation_module.evaluate(world, display, image, 
                        detected_objects_per_frame, view_width, view_height, show_ground_truth_bbox_screen, WARNING_AREA)

                    true_pos_SUT.append(true_pos_SUT_per_frame)
                    true_neg_SUT.append(true_neg_SUT_per_frame)
                    false_pos_SUT.append(false_pos_SUT_per_frame)
                    false_neg_SUT.append(false_neg_SUT_per_frame)
                ############################### SUT evaluation ###############################

            pygame.display.flip()
            pygame.event.pump()

            # IKS: Iterate over all actors and save their position, orientation, 3d bounding box, ...
            # TODO: Only supports dynamic actors right now
            if not args.no_recording:
                actor_states = []
                for actor in world.world.get_actors():
                    if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                        actor_states.append({'type': actor.type_id,
                                             'bounding_box': actor.bounding_box,
                                             'transform': actor.get_transform(),
                                             'id': actor.id,
                                             'name': actor.attributes['role_name']})

                world_state_queue.append({'actor_states': actor_states,
                                          'sensor_transform': world.camera_manager.sensor.get_transform(),
                                          'camera_calibration': calculate_camera_calibration(
                                              int(world.camera_manager.sensor.attributes['image_size_x']),
                                              int(world.camera_manager.sensor.attributes['image_size_y']),
                                              float(world.camera_manager.sensor.attributes['fov']))
                                          })


            # If the scenario has ended, save the recorded data and exit
            if SCENARIO_PROCESS.poll() is not None:
                if not args.no_recording:
                    # Wait a bit in case some data is still forwarded to the sensors' listeners
                    time.sleep(1)

                    # Destroy the world to stop the simulation
                    if world is not None:
                        world.destroy()
                        world = None

                    pygame.quit()

                    print('SAVING RECORDED SENSOR DATA, DO NOT QUIT MANUALLY!')

                    # Save recorded images here instead of in the listener directly to prevent loss of data
                    rgb_dir = os.path.join(args.output_dir, 'rgb_camera')
                    os.makedirs(rgb_dir)
                    i = 1
                    for image in image_queue:
                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        cv.imwrite(os.path.join(rgb_dir, '{}.png'.format(i)),image)
                        i+=1

                    # Same with semantic segmentation data
                    if 'semantic_segmentation_camera' in args.sensors:
                        semantic_seg_dir = os.path.join(args.output_dir, 'semantic_segmentation')
                        os.makedirs(semantic_seg_dir)

                        for image in semantic_seg_queue:
                            image.save_to_disk(os.path.join(semantic_seg_dir, f'{image.frame:09d}.png'), carla.ColorConverter.CityScapesPalette)

                    # Same with the dvs data
                    if 'event_camera' in args.sensors:
                        n_events = 0
                        for event_array in events_queue:
                            n_events += len(event_array)

                        x = np.empty(shape=(n_events,), dtype=np.uint16)
                        y = np.empty(shape=(n_events,), dtype=np.uint16)
                        t = np.empty(shape=(n_events,), dtype=np.uint32)
                        pol = np.empty(shape=(n_events,), dtype=np.int8)

                        idx = 0
                        for event_array in events_queue:
                            for event in event_array:
                                x[idx] = event.x
                                y[idx] = event.y
                                t[idx] = event.t // 1000  # Microsecond resolution is enough (if not, need to change the dtype as well
                                pol[idx] = event.pol
                                idx += 1

                        with h5py.File(os.path.join(args.output_dir, 'dvs.h5'), 'w') as h5_file:
                            h5_file.create_dataset('x', data=x, compression='gzip')
                            h5_file.create_dataset('y', data=y, compression='gzip')
                            h5_file.create_dataset('t', data=t, compression='gzip')
                            h5_file.create_dataset('pol', data=pol, compression='gzip')

                    # Extract the 2d bounding boxes for each frame and also save them to disk
                    if args.annotation_format == 'kitti':
                        bbox_dir = os.path.join(args.output_dir, '2d_bounding_boxes')
                        os.makedirs(bbox_dir)

                    if args.annotation_format == 'coco':
                        coco_annotation_id = 1
                        coco = {
                            'info': {
                                'description': 'SEC-Learn CARLA Scenarios Dataset',
                                'url': '',
                                'version': '1.0',
                                'year': str(datetime.datetime.now().year),
                                'contributor': 'Fraunhofer IKS',
                                'date_created': str(datetime.datetime.now().date())
                            },
                            'licenses': [],
                            'images': [],
                            'annotations': [],
                            'categories': [
                                {'supercategory': 'person', 'id': 1, 'name': 'pedestrian'},
                                {'supercategory': 'vehicle', 'id': 2, 'name': 'bike'},
                                {'supercategory': 'vehicle', 'id': 3, 'name': 'motorcycle'},
                                {'supercategory': 'vehicle', 'id': 4, 'name': 'car'}
                            ]
                        }

                    for image, world_state in zip(semantic_seg_queue, world_state_queue):
                        # Convert semantic segmentation image to city scapes color palette
                        image.convert(carla.ColorConverter.CityScapesPalette)

                        bboxes = []
                        for actor_state in world_state['actor_states']:
                            bbox = get_2d_bounding_box(
                                actor_state['transform'],
                                actor_state['bounding_box'].extent,
                                actor_state['bounding_box'].location,
                                world_state['sensor_transform'],
                                actor_state['type'],
                                world_state['camera_calibration'],
                                image
                            )

                            if bbox is not None:
                                bboxes.append({'bbox': bbox,
                                               'type': get_class_from_actor_type(actor_state['type']),
                                               'actor_id': actor_state['id'],
                                               'actor_name': actor_state['name']})

                        # TODO: Do some post processing, e.g. removing invalid bounding boxes
                        if args.annotation_format == 'kitti':
                            with open(os.path.join(bbox_dir, f'{image.frame:09d}.csv'), 'w') as bbox_file:
                                for bbox in bboxes:
                                    bbox_file.write(bbox['type'] +
                                                    '\t0\t0\t0\t' +
                                                    '\t'.join(map(str, bbox['bbox'])) +
                                                    '\t0\t0\t0\t0\t0\t0\t0' +
                                                    '\t' + str(bbox['actor_id']) + '\t' + bbox['actor_name'])
                        elif args.annotation_format == 'coco':
                            coco['images'].append({
                                'license': 0,
                                'file_name': f'{image.frame:09d}.png',
                                'height': args.height,
                                'width': args.width,
                                'date_captured': str(datetime.datetime.now().date()),
                                "id": image.frame
                            })

                            for bbox in bboxes:
                                coco['annotations'].append({
                                    'is_crowd': 0,
                                    'image_id': image.frame,
                                    'bbox': [bbox['bbox'][0],
                                             bbox['bbox'][1],
                                             bbox['bbox'][2] - bbox['bbox'][0],
                                             bbox['bbox'][3] - bbox['bbox'][1]],
                                    'category': [x['id'] for x in coco['categories'] if x['name'] == bbox['type']][0],
                                    'id': coco_annotation_id
                                })
                                coco_annotation_id += 1

                    if args.annotation_format == 'coco':
                        with open(os.path.join(args.output_dir, 'labels.json'), 'w') as coco_file:
                            json.dump(coco, coco_file, indent=2)

                return

    finally:
        if EVALUATE:
            print("Final results...")
            print("true_pos_SUT: {} \ntrue_neg_SUT: {} \nfalse_pos_SUT: {} \nfalse_neg_SUT: {}".format(
                np.sum(true_pos_SUT), np.sum(true_neg_SUT), np.sum(false_pos_SUT), np.sum(false_neg_SUT)))

        if world is not None:
            world.destroy()

        # IKS: Kill scenario process
        try:
            os.kill(SCENARIO_PROCESS.pid, signal.SIGKILL)
        except:
            pass
        
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Scenario Generator')

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-t', '--types',
        metavar='T',
        default="aa",
        help='pair of types (a=any, h=hero, v=vehicle, w=walkers, t=trafficLight, o=others')
    # types pattern samples:
        # -t aa == any to any == show every collision (the default)
        # -t vv == vehicle to vehicle == show every collision between vehicles only
        # -t vt == vehicle to traffic light == show every collision between a vehicle and a traffic light
        # -t hh == hero to hero == show collision between a hero and another hero

    # IKS: Add arguments for the scenario generator
    argparser.add_argument('--output-dir',
                           required=True,
                           help='Location where recorded sensor and meta data is stored')

    argparser.add_argument('--scenario-path',
                           required=True,
                           help='Path to the OpenSCENARIO file which is to be run')

    argparser.add_argument('--time-of-day',
                           type=str,
                           default='day',
                           choices=['day', 'sunset', 'night'],
                           help='Time of day')

    argparser.add_argument('--rain',
                           type=str,
                           default='none',
                           choices=['none', 'light', 'medium', 'heavy'],
                           help='If and how much it should rain')

    argparser.add_argument('--clouds',
                           type=str,
                           default='none',
                           choices=['none', 'light', 'medium', 'heavy'],
                           help='If and how much clouds should be present')

    argparser.add_argument('--fog',
                           type=str,
                           default='none',
                           choices=['none', 'light', 'medium', 'heavy'],
                           help='If and how much fog should be present')

    argparser.add_argument('--sensors',
                           type=str,
                           default=['rgb_camera'],
                           nargs='+',
                           choices=['rgb_camera', 'event_camera', 'semantic_segmentation_camera'],
                           help='The sensors to record. Multiple sensors can be specified by separating them with spaces')

    argparser.add_argument('--annotation-format',
                           type=str,
                           default='kitti',
                           choices=['kitti', 'coco'],
                           help='The format to use for annotations. KITTI format produces a csv for each image'
                                '(two additional columns are added at the end to uniquely identify objects, compared to'
                                'the standard KITTI format) and COCO creates a single json file containing all'
                                'annotations.')

    argparser.add_argument('--no-recording',
                           action='store_true',
                           help='Specify this flag if the sensor data should not be recorded')

    argparser.add_argument('--integrate-object-detector',
                           action='store_true',
                           help='Specify this flag if bounding boxes should be visualised using an object detector'
                                'connected via IPC. For the integration, refer to this repository:'
                                'https://gitlab.cc-asp.fraunhofer.de/carla/experimental/carla-object-detection-integration-yolov4/-/blob/master/carla_client_object_detection.py')
    
    argparser.add_argument('--object_detector_model_type', #it works when the argument --integrate-object-detector is activated too
                           type=str,
                           default='yolo',
                           choices=['yolo', 'detr'],
                           help='object detector model type')

    argparser.add_argument('--no_intervention',
                           action='store_true',
                           help='Specify this flag if the emergency braking should not be activated')

    argparser.add_argument('--fault_type',
                           type=str,
                           default='none',
                           choices=['brightness', 'contrast', 'sun_flare', 'rain', 'snow', 'fog', 'smoke', 'pixel_trap', 'row_add_logic', 'shifted_pixel', 'channel_shuffle', 
                           'channel_dropout', 'coarse_dropout', 'grid_dropout', 'spatter', 'gaussian_noise', 'shot_noise', 'speckle_noise', 'defocus_blur', 'elastic_transform', 
                           'impulse_noise', 'gaussian_blur', 'pixelate', 'ice', 'broken_lens', 'dirty', 'condensation', 'novelty', 'anomaly'],
                           help='25 transformations from four types of OOD categories: novelty class, anomalies, distributional shift, and noise.')

    argparser.add_argument('--severity',
                           type=str,
                           default='none',
                           choices=['1', '2', '3', '4', '5'],
                           help='Severity for a fault type')

    ###########################################################################################
    ### fault_type
    #Distributional shift: 
    #   In sensor's quality: 'brightness', 'contrast'
    #   In the environment: 'sun_flare', 'rain', 'snow', 'smoke', 'fog' 

    #Anomaly: 'row_add_logic', 'shifted_pixel', 'channel_shuffle', 'channel_dropout', 'coarse_dropout', 'grid_dropout'

    #Noise: 'spatter', 'gaussian_noise', 'shot_noise', 'speckle_noise', 'defocus_blur',
    #'elastic_transform', 'impulse_noise', 'gaussian_blur', 'pixelate'

    ###########################################################################################

    args = argparser.parse_args()
    #print(args)

    args.rolename = 'hero'      # Needed for CARLA version
    args.filter = "vehicle.*"   # Needed for CARLA version
    args.gamma = 2.2   # Needed for CARLA version
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    #print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)
    finally:
        try:
            os.kill(SCENARIO_PROCESS.pid, signal.SIGKILL)
        except:
            pass


if __name__ == '__main__':
    main()
