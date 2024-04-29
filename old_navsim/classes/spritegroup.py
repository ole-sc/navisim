import pygame as pg
from pygame.locals import *


class RenderNavSim(pg.sprite.Group):
    '''custom draw function'''
    def __init__(self, *sprites) -> None:
        pg.sprite.Group.__init__(self, *sprites)

    def draw_nav(
        self, surface, camera, special_flags=0
    ):  
        """draw all sprites onto the surface

        Group.draw(surface, special_flags=0): return Rect_list

        Draws all of the member sprites onto the given surface.

        """
        sprites = self.sprites()
        if hasattr(surface, "blits"):
            self.spritedict.update(
                zip(
                    sprites,
                    surface.blits(
                        (spr.image, (spr.rect.center - camera.pos)*camera.scale, None, special_flags) for spr in sprites
                    ),
                )
            )
        else:
            for spr in sprites:
                spr.rect.center = (spr.rect.center - camera.pos)*camera.scale
                self.spritedict[spr] = surface.blit(
                    spr.image, spr.rect.center, None, special_flags
                )
        self.lostsprites = []
        dirty = self.lostsprites

        return dirty
    
    def scale_images(self, camera):
        sprites = self.sprites()
        for sprite in sprites:
            sprite.update_sprite(camera)
