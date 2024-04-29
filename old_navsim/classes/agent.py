import numpy as np
import pygame as pg
from pygame.locals import *
from classes.resources import *

class Agent(pg.sprite.Sprite):
    
    def __init__(self, init_pos, init_rot) -> None:
        pg.sprite.Sprite.__init__(self)

        self.image_original, self.rect = load_image("agent.png", scale = 0.5)
        self.image = self.image_original # this is the image we want to display

        self.pos = init_pos
        self.rot = init_rot
        self.speed = 50
        
        self.view_range = 175

        self.scale = 1
        # the navigation system of the agent that will determine its movement
        self.nav = AgentNav()


    def update(self, food_spritegroup, dt):
        self.rot = self.rotate_to_food(food_spritegroup)
        direction = np.array([np.cos(self.rot), -np.sin(self.rot)])
        self.pos = self.pos + direction*dt/1000*self.speed
        self.update_sprite()

    def update_sprite(self, camera = None):
        if camera is not None:
            self.scale = camera.scale
            self.image = pg.transform.rotozoom(self.image_original, self.rot*180/np.pi - 90,camera.scale)
        else:
            self.image = pg.transform.rotozoom(self.image_original, self.rot*180/np.pi - 90,self.scale)
        self.rect  = self.image.get_rect(center=self.pos)


    def rotate_to_food(self, food_spritegroup):
        min_dist = np.inf
        closest_food = None
        for food in food_spritegroup:
            dist = np.linalg.norm(food.pos-self.pos)
            if dist < self.view_range and dist < min_dist:
                min_dist = dist
                closest_food = food
        if closest_food is not None:
            diff = closest_food.pos - self.pos
            return np.arctan2(-diff[1], diff[0]) 
        else:
            return self.rot + 0.03*np.random.rand()
        
class AgentNav():
    '''the class that will be responsible for movement of the ant'''
    def __init__(self) -> None:
        self.score = 0

    def update(self, visible_food_items) -> float: 
        pass

