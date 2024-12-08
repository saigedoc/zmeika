import numpy as np
import random
import moviepy
import pygame as pg
from pygame.locals import *
import datetime as dt
from PIL import Image
import sys
from math import fabs
import os
import time
import threading


from numba import jit


from data._func.CashMaster import CASH_MASTER


cm = CASH_MASTER(Mb=500000000, Direstory='data\cash', Traceback=False)

from tqdm import tqdm
from matplotlib import pyplot as plt
from cycler import cycler





try:
    RECORD = cm.Call('Record')
except:
    RECORD = 0
pg.init()






import numpy as np
import os







def fill(S, color):
    surface = S.copy()
    """Fill all pixels of the surface with color, preserve transparency."""
    w, h = surface.get_size()
    try:
        r, g, b, _ = color
    except:
        r,g,b = color
    r,g,b = min(255,r), min(255,g),min(255,b) 
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            #print(r, g, b, a)
            surface.set_at((x, y), pg.Color(r, g, b, a))
    return surface





def change_color(img, c1, c2, p=0, noisy_bool=True):
    img = img.convert("RGBA")
    pixdata = np.array(img)
    for y in range(pixdata.shape[1]):
        for x in range(pixdata.shape[0]):
            if (pixdata[x, y][0] - p < c1[0] and c1[0] < pixdata[x, y][0] + p
                ) and (pixdata[x, y][1] - p < c1[1]
                       and c1[1] < pixdata[x, y][1] + p) and (
                           pixdata[x, y][2] - p < c1[2]
                           and c1[2] < pixdata[x, y][2] + p):
                pixdata[x, y] = list(c2) + [pixdata[x, y][-1]]

    
    
    #img = Image.fromarray(pixdata)
    return Image.fromarray(pixdata)#img


#'freesansbold.ttf'
class BUTTON:
    def __init__(self,
                 tfont='freesansbold.ttf',
                 size=24,
                 text='example',
                 pos=[0, 0],
                 width=False,
                 height=False,
                 func=False):
        #super(Button, self).__init__()
        self.size = size
        self.tfont = tfont
        self.text = text
        self.pos = pos
        self.width = width
        self.height = height
        self.func = func
        self.pressed = False

    def blit(self, screen, c, fc):
        f = pg.font.Font(self.tfont, self.size)
        if self.pressed:
            txt = f.render(self.text, True, fc)
        else:
            txt = f.render(self.text, True, c)

        if not (self.width):
            w = txt.get_size()[0]
        else:
            w = self.width

        if not (self.height):
            h = txt.get_size()[1]
        else:
            h = self.height
        txt = pg.transform.scale(txt, (w, h))
        screen.blit(txt, [self.pos[0] - (w // 2), self.pos[1] - (h // 2)])

    def press(self):
        if self.func:
            self.func()


class GAME:
    def __init__(self,
                 window_name="PIZDAZMEIKA",
                 FPS=False,
                 background=(0, 0, 0),
                 color=(255, 255, 255),
                 size=[11, 11],
                 indent=1,
                 display_size=[720, 480],
                 start_position=[5, 5, 0],
                 speed=1,
                 start_food=[3, 5, 1],
                 food_gen=1,
                 food_size_chance=[100],
                 food_color=(255, 255, 0),
                 block_type=1,
                 fullscreen=False,
                 label=False,
                 n_bl=False):
        ############
        self.ai = False
        self.ai_train = False
        self.ai_show = True
        self.label = label
        ############

        self.start_position = start_position
        self.name = window_name
        self.size = size
        self.indent = indent
        self.background = background
        self.color = color
        self.display_size = display_size
        self.FPS = FPS
        self.fg = food_gen
        self.fs = food_size_chance
        self.block_type = block_type
        self.n_blocks = n_bl
        self.f_plus = 0
        try:
            self.record = cm.Call('Record')
        except:
            self.record = 0

        self.block_name = r"data\blocks\block-%s.png" % int(block_type)

        self.full = fullscreen


        #if self.color != (255, 255, 255):
        
        self.bright = [1, min(self.size)//5, 1] # 
        self.bb = (1-0.9)/self.bright[1]
        self.block_img = Image.open(self.block_name)
        self.block_img = change_color(self.block_img, (255, 255, 255),self.color,p=20, noisy_bool=False)
        #draw = ImageDraw.Draw(self.Fblock)
        mode = self.block_img.mode
        size = self.block_img.size
        data = self.block_img.tobytes()
        self.block_img = pg.image.fromstring(data, size, mode)
        #else:
            #self.block_img = pg.image.load(self.block_name)

        self.food_bonus = np.sum(
            (np.array(list(map(lambda x: 1 + x / sum(self.fs), self.fs)))**2))
        self.score = 0
        self.RECORD = RECORD
        self.bonus = 0
        self.stage = 0

        self.stats = {}
        self.end_stats = {}

        self.ffff = False
        self.llll = False
        self.ai_out = True

        if not (self.start_position):
            start_position = [size[0] // 2 + 1, size[1] // 2 + 1, 0, self.block_img]
        else:
            start_position = start_position + [self.block_img]

        self.sb = np.array(start_position).reshape(1, 4)  # snake_blocks
        #print(self.sb)
        self.speed = speed
        self.start_food = start_food
        if start_food:
            self.feed = np.array(start_food).reshape(1, 3)
        else:
            self.feed = np.array([]).reshape(0, 3)
        self.food_color = food_color

        self.runbool = True
        self.new_v = 0

        if self.display_size[0] / (self.size[0] +
                                   self.indent * 2) <= self.display_size[1] / (
                                       self.size[1] + self.indent * 2):
            self.d_scal = self.display_size[0] / 1920
        else:
            self.d_scal = self.display_size[1] / 1080

        #self.block_scale = min(self.display_size) // (self.size[0]+self.indent*2)
        if self.display_size[0] / (self.size[0] +
                                   self.indent * 2) <= self.display_size[1] / (
                                       self.size[1] + self.indent * 2):
            self.block_scale = self.display_size[0] // (self.size[0] +
                                                        self.indent * 2)
        else:
            self.block_scale = self.display_size[1] // (self.size[1] +
                                                        self.indent * 2)

        self.Fblock = Image.open(self.block_name)
        self.Fblock = change_color(self.Fblock, (255, 255, 255),
                                   food_color,
                                   p=10, noisy_bool=False)
        mode = self.Fblock.mode
        size = self.Fblock.size
        data = self.Fblock.tobytes()
        self.Fblock = pg.image.fromstring(data, size, mode)


        self.t_b_e = True
        self.t_b_s = True
        self.t_b_m = True
        self.t_b_b = True

        self.block_brs = []
        self.br_now = 0
        self.br_v = 1
        self.get_brightness()



        self.block_indent = 0.1
        self.food_indent = 0.2
        self.min_brightness = 0.4
        #print('1:', self.sb)

    def get_brightness(self):
        self.bright = [1, min(self.size)//5, 1]
        self.bb = (1-0.5)/self.bright[1]
        self.block_brs = []
        self.br_now = 0
        self.br_v = 1
        self.bright[2] = 1
        self.block_brs.append(self.block_img)
        
        br = 1
        for _ in range(int(self.bright[1])):
            self.block_brs.append( fill(self.block_img, list(map(lambda x: round(x*(br)), self.color))) )
            br -= self.bb
            #print(br)
        #print('---------------')

    def get_record(self):
        global RECORD
        RECORD = self.record

    def window_init(self, win_x=100, win_y=100):
        if self.ai:
            self.ffff = True

        if self.ai_show:
            os.environ['Sp_VIDEO_WINDOW_POS'] = "%d,%d" % (win_x, win_y)
            if self.full:
                self.screen = pg.display.set_mode((self.display_size),
                                                  pg.FULLSCREEN)
            else:
                self.screen = pg.display.set_mode((self.display_size))

            pg.display.set_caption(self.name)
            self.screen.fill((self.background))

            pg.display.flip()

        pg.font.init()
        self.cl = pg.time.Clock()
        self.time = 0

        if self.FPS:
            self.cl.tick(self.FPS)
        #if self.ai_show:
        #self.blits()

    
    def spawn_food(self):
        tmp_f = True
        if len(self.feed) == 0:
            locks = []
            self.stage += 1
            self.stats['Stage'].text = 'Stage: %s' % (self.stage)
            if self.stage >1:
            	self.bonus += (1000 * self.stage * ((self.speed / 0.01)**1.75) *(self.n_blocks**1.25) * ((50**2 + 50**2)**0.5) /((self.size[0]**2 + self.size[1]**2)**0.5))/(self.food_bonus**2)*(self.fg/50)
            self.score += self.bonus
            self.bonus = 0
            #print(self.bonus)
            for i in range(
                    min(int(self.fg), (self.size[0] + 2) * (self.size[1] * 2) -
                        len(self.sb))):
                while tmp_f:
                    x = round(random.random() * (self.size[0] + 1)) - 1
                    y = round(random.random() * (self.size[1] + 1)) - 1
                    s = ''
                    for j in self.fs:
                        s += str(self.fs.index(j) + 1) * j
                    s = random.choice(s)
                    #print(s)
                    feeds = np.array([])
                    for ss in self.sb:
                        feeds = np.append(feeds, [int(x),
                                                  int(y)] == ss.tolist()[:2])
                        for S0 in range(int(s)):
                            for S1 in range(int(s)):
                                feeds = np.append(feeds,
                                                  [int(x) + S0,
                                                   int(y) + S1
                                                   ] == ss.tolist()[:2])

                    #print([int(x), int(y)] in locks)

                    if int(x) + int(s) > self.size[0] + 1 or int(y) + int(
                            s) > self.size[1] + 1:
                        feeds = np.append(feeds, True)
                    for m0 in range(int(s)):
                        for m1 in range(int(s)):
                            feeds = np.append(feeds,
                                              [int(x) + m0,
                                               int(y) + m1] in locks)

                    if np.any(feeds):
                        tmp_f = True
                    else:
                        tmp_f = False
                        for m0 in range(int(s)):
                            for m1 in range(int(s)):
                                locks.append([int(x) + m0, int(y) + m1])
                        self.feed = np.append(
                            self.feed,
                            [[int(x), int(y), int(s)]], axis=0)
                tmp_f = True
        self.t_b_s = True

    def new_block(self):
        if len(self.sb)%self.bright[1]==0:
            self.bright[0] *= -1
        #if self.bright[0] > 0:
            #self.bright[2]-=self.bb
        #else:
            #self.bright[2]+=self.bb
            #print()
        
            #print((1*self.br_v))
        
        self.br_now += (1*self.br_v)
        
        if (self.br_now >= len(self.block_brs)-1) or (self.br_now <= 0):
            self.br_v *=-1
        


        #print(self.sb)
        if self.sb[-1][2] == 0:  #up
            new_pos = [self.sb[-1][0], self.sb[-1][1] + 1]
        elif self.sb[-1][2] == 1:  #right
            new_pos = [self.sb[-1][0] - 1, self.sb[-1][1]]
        elif self.sb[-1][2] == 2:  #down
            new_pos = [self.sb[-1][0], self.sb[-1][1] - 1]
        elif self.sb[-1][2] == 3:  #left
            new_pos = [self.sb[-1][0] + 1, self.sb[-1][1]]
        #print(len(self.block_brs))
        #print(round(self.bright[2]/self.bb))
        #print(self.block_brs[round(self.bright[2]/self.bb)])
        #print(self.block_img)
        #print(int( (1 - self.bright[2])/self.bb))
        #print(self.br_now)
        #print(len(self.block_brs))
        #print(self.br_now, 'G')
        
        self.sb = np.append(self.sb, [[new_pos[0], new_pos[1], self.sb[-1][2],  self.block_brs[self.br_now]  ]], axis=0) #self.block_brs[round(self.bright[2]/self.bb)]  ]] #fill(self.block_img, list(map(lambda x: round(x*self.bright[2]), self.color)))

    def change_vector(self, new_v):
        if fabs(self.sb[0][2] - new_v) != 2 and self.ffff:
            self.new_v = new_v

    def move(self):
        if self.f_plus>0:
            self.new_block()
            self.f_plus -= 1

        self.sb[0][2] = self.new_v
        self.ai_out = True
        for i in range(len(self.sb) - 1, -1, -1):
            x, y, v, _ = self.sb[i]
            if v == 0:
                self.sb[i][1] = y - 1
            elif v == 1:
                self.sb[i][0] = x + 1
            elif v == 2:
                self.sb[i][1] = y + 1
            elif v == 3:
                self.sb[i][0] = x - 1
            try:
                self.sb[i + 1][2] = self.sb[i][2]
            except:
                pass

    def ai_conditions(self):
        x = get_screenshot(self)
        x = self.ai.forward(x)
        x = round(x.item() * 3)
        #print(x)
        self.change_vector(x)
        self.ai_out = False

    def stop(self):
        if self.ffff:
            self.ffff = False
        else:
            self.ffff = True

    def event_conditions(self, event):
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
            self.runbool = False
        if event.type == pg.KEYDOWN:
            #self.firststart()
            if event.key == K_ESCAPE:
                pg.quit()
                self.runbool = False
            if event.key == K_SPACE:
                self.stop()
            if event.key == K_UP:
                self.change_vector(0)
            if event.key == K_RIGHT:
                self.change_vector(1)
            if event.key == K_DOWN:
                self.change_vector(2)
            if event.key == K_LEFT:
                self.change_vector(3)

    def firststart(self):
        if self.llll or self.ai:
            self.runbool = False
        else:
            self.ffff = True

    def move_loop(self):
        #print('g')
        while self.runbool and (not self.llll):
            #print('fg')
            time.sleep(0.00001)
            for _ in range(round(self.time * self.speed - 0.5)):
                self.move()
                self.time = self.time - round(self.time * self.speed -
                                              0.5) / self.speed

    def blit_mup(self):
        pg.draw.line(
            self.screen,
            self.color,
            (
                (self.display_size[0] - self.block_scale *
                 (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                (-1 + self.indent)  #up
                ,
                (self.display_size[1] - self.block_scale *
                 (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                (-1 + self.indent)),
            ((self.display_size[0] - self.block_scale *
              (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
             (self.size[0] + 1 + self.indent),
             (self.display_size[1] - self.block_scale *
              (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
             (-1 + self.indent)),
            3)

        pg.draw.line(
            self.screen,
            self.color,
            (
                (self.display_size[0] - self.block_scale *
                 (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                (self.size[0] + 1 + self.indent)  #right
                ,
                (self.display_size[1] - self.block_scale *
                 (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                (-1 + self.indent)),
            ((self.display_size[0] - self.block_scale *
              (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
             (self.size[0] + 1 + self.indent),
             (self.display_size[1] - self.block_scale *
              (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
             (self.size[1] + 1 + self.indent)),
            3)

        pg.draw.line(
            self.screen,
            self.color,
            (
                (self.display_size[0] - self.block_scale *
                 (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                (-1 + self.indent)  #down
                ,
                (self.display_size[1] - self.block_scale *
                 (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                (self.size[1] + 1 + self.indent)),
            ((self.display_size[0] - self.block_scale *
              (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
             (self.size[0] + 1 + self.indent),
             (self.display_size[1] - self.block_scale *
              (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
             (self.size[1] + 1 + self.indent)),
            3)

        pg.draw.line(
            self.screen,
            self.color,
            (
                (self.display_size[0] - self.block_scale *
                 (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                (-1 + self.indent)  #left
                ,
                (self.display_size[1] - self.block_scale *
                 (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                (-1 + self.indent)),
            ((self.display_size[0] - self.block_scale *
              (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
             (-1 + self.indent),
             (self.display_size[1] - self.block_scale *
              (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
             (self.size[1] + 1 + self.indent)),
            3)

    def blit_blocks(self):
        #self.Sblock = pg.transform.scale(self.block_img,
                                         #(self.block_scale, self.block_scale))
        #print(self.sb)
        z=0
        try:
            br_d = (1-self.min_brightness)/(len(self.sb)-1)
        except:
            br_d = 0
        for bl in self.sb:
            x, y = bl[0], bl[1]

            if  br_d>0 and len(self.sb)>2:
                cl_now = list(map(lambda x: x*(1-z*br_d), self.color))
            else:
                cl_now = self.color
            pg.draw.rect(self.screen, cl_now, 
                ((self.display_size[0] - self.block_scale *
                  (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                 (x + self.indent) + round(self.block_indent*self.block_scale),

                 (self.display_size[1] - self.block_scale *
                  (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                 (y + self.indent) + round(self.block_indent*self.block_scale), 

                 self.block_scale*(1-(2*self.block_indent)),
                self.block_scale*(1-(2*self.block_indent)
                 )))
            z+=1



            #s_block = pg.transform.scale(bl[3], (self.block_scale, self.block_scale))

            """self.screen.blit(
                                                    s_block,
                                                    ((self.display_size[0] - self.block_scale *
                                                      (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                                                     (x + self.indent),
                                                     (self.display_size[1] - self.block_scale *
                                                      (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                                                     (y + self.indent)))"""

    def blit_feed(self):
        self.Fblock = pg.transform.scale(self.Fblock,
                                         (self.block_scale, self.block_scale))
        for f in self.feed:
            x, y, s = f.tolist()

            pg.draw.rect(self.screen, self.food_color, 
                ((self.display_size[0] - self.block_scale *
                  (self.size[0] + self.indent * 2)) // 2 + self.block_scale *
                 (x + self.indent) + round(self.food_indent*self.block_scale),

                 (self.display_size[1] - self.block_scale *
                  (self.size[1] + self.indent * 2)) // 2 + self.block_scale *
                 (y + self.indent) + round(self.food_indent*self.block_scale), 

                 self.block_scale  *(s-(2*self.food_indent)),
                 self.block_scale  *(s-(2*self.food_indent)
                 )))



            """self.screen.blit(
                                                    pg.transform.scale(
                                                        self.Fblock,
                                                        (int(self.block_scale * s), int(self.block_scale * s))),
                                                    (int((self.display_size[0] - self.block_scale *
                                                          (self.size[0] + self.indent * 2)) // 2 +
                                                         self.block_scale * (x + self.indent)),
                                                     int((self.display_size[1] - self.block_scale *
                                                          (self.size[1] + self.indent * 2)) // 2 +
                                                         self.block_scale * (y + self.indent))))"""

    def blits(self):
        self.screen.fill(self.background)
        self.blit_feed()
        self.blit_blocks()
        self.blit_mup()
        for s in self.stats.values():
            s.blit(self.screen, self.color, self.food_color)
        pg.display.flip()

    def bounds(self):
        for i in range(len(self.sb)):
            if i != 0 and self.sb[i].tolist()[:2] == self.sb[0].tolist()[:2]:
                self.lose()
        if self.sb[0][0] < -1 or self.sb[0][0] > self.size[0]:
            self.lose()
            #print('g1')
        if self.sb[0][1] < -1 or self.sb[0][1] > self.size[1]:
            self.lose()

    def eat(self):
        df = []
        for i in range(len(self.feed)):
            feeds = np.array([])

            for x in range(int(self.feed[i][-1])):
                for y in range(int(self.feed[i][-1])):
                    feeds = np.append(
                        feeds, self.feed[i].tolist()[:2] == [
                            self.sb[0].tolist()[0] - x,
                            self.sb[0].tolist()[1] - y
                        ])
            if np.any(feeds):
                if len(self.sb) >= self.size[0] * self.size[1] - 1:
                    self.bonus += 10000 * (
                        (self.speed / 0.01)**1.75) * (self.n_blocks**1.25) * (
                            (50**2 + 50**2)**0.5) / (
                                (self.size[0]**2 + self.size[1]**2)**0.5)
                self.get_score(self.feed[i][2])
                self.stats['Score'].text = 'Score: %s' % (int(self.score // 1)
                                                          )  #*100//1/100
                if self.score > RECORD:
                    self.stats['Record'].text = 'Record: %s' % (int(
                        self.score // 1))

                #if self.n_blocks:
                    #for k in range(int(self.n_blocks)):
                        #self.new_block()
                self.f_plus += self.n_blocks

                df.append(i)
        self.feed = np.delete(self.feed, df, 0)
        self.t_b_e = True

    def lose(self):
        global RECORD
        if not (self.llll):
            self.llll = True
            if self.score > self.RECORD:
                self.RECORD = self.score
            if RECORD < self.RECORD:
                RECORD = self.RECORD
            

            #print(self.RECORD)
            cm.Cash(self.RECORD, 'Record')
            #print(cm.Call('Record'))
            #print(cm.Call('Load_config'))

    def quit(self):
        pg.quit()
        self.runbool = False

    def menu_blit(self):
        self.screen.fill(self.background)
        for b in self.Buttons:
            b.blit(self.screen, self.color, self.food_color)
        pg.display.flip()

    def press_btn(self):
        self.Buttons[self.choose].press()

    def menu_cond(self, event):
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
            runbool = False
        if event.type == pg.KEYDOWN:
            #self.firststart()
            if event.key == K_ESCAPE:
                pg.quit()
                self.runbool = False
            if event.key == K_RETURN:
                self.press_btn()
            if event.key == K_SPACE:
                pass
            if event.key == K_UP:
                self.ch_up()
            if event.key == K_RIGHT:
                pass
            if event.key == K_DOWN:
                self.ch_down()
            if event.key == K_LEFT:
                pass

    def ch_up(self):
        if self.choose > 0:
            self.Buttons[self.choose].pressed = False
            self.choose -= 1
            self.Buttons[self.choose].pressed = True

    def ch_down(self):
        if self.choose < len(self.Buttons) - 1:
            self.Buttons[self.choose].pressed = False
            self.choose += 1
            self.Buttons[self.choose].pressed = True

    def menu_exit(self):
        self.runbool = False

    def new_start(self):
        '''self.__init__(window_name=self.name, FPS=self.FPS, background=self.background, color=self.color, size=self.size, indent=self.indent, 
                                         display_size=self.display_size,  start_position=self.start_position, speed=self.speed, start_food=self.start_food, food_gen=self.fg, 
                                         food_size_chance=self.fs, food_color=self.food_color, block_type=self.block_type, fullscreen=self.full, label=self.label)'''

        if not (self.start_position):
            self.new_v = 0
            start_position = [self.size[0] // 2 + 1, self.size[1] // 2 + 1, 0, self.block_img]
        else:
            self.new_v = self.start_position[-1]
            start_position = self.start_position
        self.sb = np.array(start_position).reshape(1, 4)  # snake_blocks
        #print(self.sb)

        if self.start_food:
            self.feed = np.array(self.start_food).reshape(1, 3)
        else:
            self.feed = np.array([]).reshape(0, 3)

        self.bright = [1, min(self.size)//10, 1]
        self.bb = (1-0.01)/self.bright[1]
        self.runbool = True
        self.llll = False
        self.ffff = False
        self.score = 0
        self.stage = 0
        self.stats = {}

        self.t_b_e = True
        self.t_b_s = True
        self.t_b_m = True
        self.t_b_b = True

        

        self.run()

    def menu(self):

        # 3 buttons
        self.Buttons = []

        # choose
        self.choose = 0

        #START
        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(100 * self.d_scal // 1),
            'text': 'Start',
            'pos': [self.display_size[0] // 2, self.display_size[1] / 5 * 2],
            'width': False,
            'height': False,
            'func': self.new_start,
        }
        self.Buttons.append(BUTTON(**mn_conf))
        self.Buttons[0].pressed = True
        #SETTING
        #print(100*self.d_scal)
        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(100 * self.d_scal // 1),
            'text': 'Setting',
            'pos': [self.display_size[0] // 2, self.display_size[1] / 5 * 3],
            'width': False,
            'height': False,
        }
        self.Buttons.append(BUTTON(**mn_conf))

        #EXIT
        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(100 * self.d_scal // 1),
            'text': 'Exit',
            'pos': [self.display_size[0] // 2, self.display_size[1] / 5 * 4],
            'width': False,
            'height': False,
            'func': self.menu_exit
        }
        self.Buttons.append(BUTTON(**mn_conf))
        #print('3:', self.sb)
        self.runbool = True
        while self.runbool:
            if self.FPS:
                t = self.cl.tick(self.FPS)
                if self.ffff:
                    self.time += t

            for event in pg.event.get():
                self.menu_cond(event)

            if not (self.runbool):
                pg.quit()
                break
            self.menu_blit()
            time.sleep(0.05)

    def get_score(self, s):
        self.score += (
            (1 + len(self.sb) / (self.size[0] * self.size[1]))**
            0.5) * ((self.speed / 0.01)**1.75) * (self.n_blocks**1.25) * (
                (50**2 + 50**2)**0.5) / (
                    (self.size[0]**2 + self.size[1]**2)**0.5) / s + self.bonus
        if self.bonus > 0:
            self.bonus = 0

    def end_blit(self):
        self.screen.fill(self.background)
        for b in self.end_stats.values():
            b.blit(self.screen, self.color, self.food_color)
        pg.display.flip()

    def run(self):

        self.get_brightness()

        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(20 * self.d_scal // 1),
            'text': 'Stage: 0',
            'pos': [self.display_size[0] * 1 // 4, self.display_size[1] / 100],
            'width': False,
            'height': False,
            'func': False,
        }

        self.stats['Stage'] = BUTTON(**mn_conf)

        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(20 * self.d_scal // 1),
            'text': 'Score: %s' % (int(self.score // 1)),
            'pos': [self.display_size[0] * 2 // 4, self.display_size[1] / 100],
            'width': False,
            'height': False,
            'func': False,
        }

        self.stats['Score'] = BUTTON(**mn_conf)

        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(20 * self.d_scal // 1),
            'text': 'Record: %s' % (int(RECORD // 1)),
            'pos': [self.display_size[0] * 3 // 4, self.display_size[1] / 100],
            'width': False,
            'height': False,
            'func': False,
        }

        self.stats['Record'] = BUTTON(**mn_conf)

        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(50 * self.d_scal // 1),
            'text': 'Stage: %s' %(int(self.stage// 1)),
            'pos': [self.display_size[0] //2, self.display_size[1] *1//4],
            'width': False,
            'height': False,
            'func': False,
        }

        self.end_stats['Stage'] = BUTTON(**mn_conf)


        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(50 * self.d_scal // 1),
            'text': 'Record: %s' %(int(RECORD // 1)),
            'pos': [self.display_size[0] // 2, self.display_size[1] * 2// 4],
            'width': False,
            'height': False,
            'func': False,
        }

        self.end_stats['Record'] = BUTTON(**mn_conf)

        mn_conf = {
            'tfont': 'freesansbold.ttf',
            'size': int(50 * self.d_scal // 1),
            'text': 'Your score: %s' %(int(self.score // 1)),
            'pos': [self.display_size[0] //2, self.display_size[1] *3//4],
            'width': False,
            'height': False,
            'func': False,
        }

        self.end_stats['Score'] = BUTTON(**mn_conf)


        
        

        self.time = 0
        self.blits()
        if self.t_b_m:
            self.t_b_m = False
            my_thread = threading.Thread(target=self.move_loop, daemon=True, args=())
            my_thread.start()
        while self.runbool:
            if self.FPS:
                t = self.cl.tick(self.FPS)
                if self.ffff:
                    self.time += t

            for event in pg.event.get():
                self.event_conditions(event)

            if self.llll:
                self.runbool = False

            if self.ffff:

                
                if self.t_b_e:
                    self.t_b_e = False
                    my_thread = threading.Thread(target=self.eat, daemon=True, args=())
                    my_thread.start()
                #self.eat()
                if self.t_b_s:
                    self.t_b_s = False
                    my_thread = threading.Thread(target=self.spawn_food, daemon=True, args=())
                    my_thread.start()
                #print('h')
                #self.spawn_food()
                #self.move_loop()

                self.bounds()

                if self.llll:
                    self.ffff = True
                    self.end_stats['Stage'].text = 'Stage: %s' %self.stage 
                    self.end_stats['Record'].text = 'Record: %s' %int(self.RECORD//1)
                    self.end_stats['Score'].text = 'Your score: %s' %int(self.score//1)
                    while self.ffff:
                        self.end_blit()

                        for event in pg.event.get():
                            if event.type == pg.KEYDOWN:
                                if event.key == K_SPACE:
                                    self.ffff = False 

                if not (self.llll):
                    self.blits()
                else:
                    break
            else:
                time.sleep(0.1)
        self.runbool = True

    #pg.quit()


#################################################################################
arguments = {
    'FPS': 120,
    'background': (0, 0, 0),
    'color': (255, 255, 255),
    'size': [20, 20],
    'indent': 1,
    'display_size': [1280, 720],
    'start_position': [10, 10, 0],  # 0-up, 1-right, 2-down, 3-left
    'speed': 0.01,
    'start_food': False,  #[3, 5, 1],
    'food_gen': 1,  # how many food we will generate
    'food_size_chance': [80, 15, 5],
    'food_color': (255, 255, 0),
    'block_type': 1,  #1/2
    'fullscreen': False,
    'label': 10,
    'n_bl': 1
}
#################################################################################

#################################################################################
params = {
    'device': 'cuda',
    'nd': 64,
    'bias': False,
    'convlayers_max': 10,
    'nd_max_multip': 64,
    'up_chanels': True,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
}
#################################################################################

ARG_S = arguments
try:
    model = cm.Call('model')
except:
    model = False


def start():
    global RECORD
    try:
        RECORD = cm.Call('Record')
    except:
        RECORD = 0
    G = GAME(**arguments)

    G.window_init()
    G.menu()


try:
    if cm.Call('Load_config'):
        arguments = cm.Call('Config')
        print('Config was loaded.')
        [print('') for _ in range(10)]
except:
    pass

while True:
    os.system('cls')
    print('1) Начать.')
    print('2) Настройки.')
    print('3) Выход.')
    answer = input()
    if answer == '1':
        start()
    elif answer == '2':
        while True:
            #
            os.system('cls')
            print("Введите название переменной, а затем необходимое значение.")
            print('Чтобы выйти из настроек введите "back".')
            print('Чтобы сохранить настройки введите "save".')
            print('Чтобы вернуть настройки поумолчанию введите "reset"')
            print('')
            print('')

            for name, x in arguments.items():
                print(name + ':', x)

            try:
                print('load_config:', cm.Call('Load_config'))
            except:
                print('load_config:', False)
            print('')
            print('')

            name = input('Переменная: ')

            if name == 'back':
                break

            if name == 'load_config':
                x = input('Значение: ')
                #cm.Cash([True, RECORD], 'Load_config')
                try:
                    if x == 'True':
                        x = True
                    elif x == "False":
                        x = False
                    else:
                        1 / 0
                    print('h')
                    cm.Cash(x, 'Load_config')
                    print('g')
                except:
                    print('Вы ввели недопустимое значение.')
                    input('Нажмите любую клавишу чтобы продолжить...')
            elif name == 'reset':
                arguments = ARG_S
            elif name == 'save':
                cm.Cash(arguments, 'Config')
                print('Конфиг сохранён.')
                input('Нажмите любую клавишу чтобы продолжить...')
            else:
                x = input('Значение: ')
                while True:

                    if x == 'True':
                        x = True
                        break
                    elif x == "False":
                        x = False
                        break
                    else:
                        try:
                            x = float(x)
                            break
                        except:
                            try:
                                x = list(map(int, (x.split())))
                                break
                            except:
                                print('Вы ввели недопустимое значение.')
                                input(
                                    'Нажмите любую клавишу чтобы продолжить...'
                                )

                if name in arguments:
                    arguments[name] = x
                else:
                    print('Этого имени нету, введите правильное имя.')
                    input('Нажмите любую клавишу чтобы продолжить...')
    elif answer == '3':
        break

    else:
        print('Вы ввели недопустимое значение.')
        input('Нажмите любую клавишу чтобы продолжить...')
