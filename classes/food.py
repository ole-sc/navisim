import numpy as np
import pygame as pg
from pygame.locals import *
from classes.resources import *

class Food(pg.sprite.Sprite):
    
    def __init__(self, pos, scale) -> None:
        pg.sprite.Sprite.__init__(self)
        self.image_original, self.rect = load_image("food.png", scale = 0.1)
        self.image = pg.transform.rotozoom(self.image_original, 0,scale)
        self.pos = pos

        
        self.rect = self.image.get_rect(center=self.pos)
       
    def update_sprite(self, camera):
        self.image = pg.transform.rotozoom(self.image_original, 0,camera.scale)
        self.rect  = self.image.get_rect(center=self.pos)