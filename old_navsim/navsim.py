import numpy as np
import torch
import os
import pygame as pg
from pygame.locals import *
from dataclasses import dataclass
# custom classes
from classes.agent import *
from classes.food import *
from classes.spritegroup import *

class NaviSim():
    def __init__(self) -> None:

        pg.init()
        self.render = Renderer()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.sim_speed = 20

        self.isSimRunning = True
        self.isPaused = False

        self.agent_spritegroup = RenderNavSim([self.init_agent() for i in range(10)])
        self.food_spritegroup = RenderNavSim([self.init_food() for i in range(50)])

    def run_game_loop(self):
        while self.isSimRunning:
            # the core game loop
            self.handle_inputs()
            
            if not self.isPaused:   
                self.agent_spritegroup.update(self.food_spritegroup, self.clock.get_time()*self.sim_speed)
                # check for collisions of agents with food items
                collision_dict = pg.sprite.groupcollide(self.agent_spritegroup, self.food_spritegroup, False, True)
                for agent, food_list in collision_dict.items():
                    agent.nav.score = agent.nav.score + len(food_list)
                    [self.food_spritegroup.add(self.init_food()) for i in range(len(food_list))]
                                 
            self.render.update_display(display_groups=[self.food_spritegroup,self.agent_spritegroup]) 
            self.clock.tick(self.fps)

    def handle_inputs(self):
        for event in pg.event.get():
            # use a switch statement here?
            if event.type == QUIT:
                self.isSimRunning = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                self.isPaused = not self.isPaused
            # I don't know whether the following two lines will behave in the way I want it two. 
            # Actually, I think I don't want time independent physics, but rather increased fps lead to faster movement
            # think about this later
            elif event.type == pg.KEYDOWN and event.key == pg.K_PLUS:
                self.sim_speed = min(self.sim_speed*1.2, 40)
            elif event.type == pg.KEYDOWN and event.key == pg.K_MINUS:
                self.sim_speed = max(self.sim_speed*0.9, 0.1)

            elif event.type == pg.KEYDOWN and event.key == pg.K_w:
                self.render.cam.move_cam((0,-1))
            elif event.type == pg.KEYDOWN and event.key == pg.K_s:
                self.render.cam.move_cam((0,1))
            elif event.type == pg.KEYDOWN and event.key == pg.K_a:
                self.render.cam.move_cam((-1,0))
            elif event.type == pg.KEYDOWN and event.key == pg.K_d:
                self.render.cam.move_cam((1,0))
                
            # camera movement
            if event.type == pg.MOUSEWHEEL:
                SCROLL_SENS = 0.1
                self.render.cam.scale_cam(1 + SCROLL_SENS*event.y)
                # scale images 
                self.agent_spritegroup.scale_images(self.render.cam)
                self.food_spritegroup.scale_images(self.render.cam)

        return True
    
    def init_agent(self):
        init_rotation = 2*np.pi*np.random.rand()
        init_position = 1000*np.random.rand(2)
        agent = Agent(init_pos=init_position, init_rot=init_rotation)
        return agent
    
    def init_food(self):
        init_position = 1000*np.random.rand(2)
        return Food(init_position, self.render.cam.scale)
    
class Renderer():
    def __init__(self) -> None:
        # check if all modules are available
        if not pg.font:
            print("Warning, fonts disabled")
        if not pg.mixer:
            print("Warning, sound disabled")
        
        # init display
        self.screen = pg.display.set_mode((1900, 1200))
        pg.display.set_caption("NaviSim")

        # init background
        self.background = pg.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((255, 244, 228))

        # simulate a camera
        self.cam = Camera()

    def update_display(self, display_groups):
        self.screen.blit(self.background, (0,0))
        for group in display_groups:
            # transform world coordinates to screen coordinates
            group.draw_nav(self.screen, self.cam)
        pg.display.flip()

class Camera:
    def __init__(self) -> None:
        self.pos = np.zeros(2)
        self.scale: float = 1.0

        self.min_scale = 0.1
        self.max_scale = 10

        self.cam_speed = 200

    def scale_cam(self, factor):
        '''change the scale of the camera while making sure it stays within bounds'''
        self.scale = self.scale * factor
        if self.scale < self.min_scale:
            self.scale = self.min_scale
        elif self.scale > self.max_scale:
            self.scale = self.max_scale
        return self.scale

    def move_cam(self, delta_pos):
        self.pos = self.pos + np.array(delta_pos)*self.cam_speed/self.scale
        # add bounds here

    def set_cam_pos(self, pos):
        # maybe add bounds here
        self.pos = pos


nvs = NaviSim()
nvs.run_game_loop()
