# -*- coding: utf-8 -*-

# Importar las librerías
import pygame, sys
from pygame.locals import *

MaxX=800
MaxY=600

# Inicializar la librería de pygame
pygame.init()
#ventana
window = pygame.display.set_mode((MaxX,MaxY))
window.fill((0,0,0))
fps=60
# Creamos el objeto reloj para sincronizar el juego
reloj = pygame.time.Clock()
radious=20

RED = (255,0,0)

movX=10+radious
movY=500-radious
# Posición de la raqueta del jugador 1 y jugador 2
raqueta1X = 100
raqueta1Y = MaxY/2
raqueta2X = MaxX-50
raqueta2Y = MaxY/2


# Tamano de cada raqueta
tamanoRaquetaX = 100
tamanoRaquetaY = 100

# Permitimos que la tecla este pulsada
pygame.key.set_repeat(1, 25)

# Eliminamos el raton
pygame.mouse.set_visible(False)

while True:
        # Hacemos que el reloj espere a un determinado fps
        reloj.tick(fps)

        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                pygame.quit()
                sys.exit()
           
            if event.type==KEYDOWN:
                if event.key== K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                elif event.key == K_w:
                    if movY>4 + radious:
                        movY-=2
                
                elif event.key == K_a:
                    if (movY >300 - radious) and (movX==250 + radious) :
                        movX=movX
                    elif movX>2 + radious:
                        movX-=2

                elif event.key == K_s:

                    if (movX>200 and movX<250) and movY>299-radious:
                        movY=movY                   
                    elif movY<500-radious:
                        movY+=2

                elif event.key == K_d:
                    if (movY > 301 - radious) and (movX>200-radious and movX<250 - radious) :
                        movX=movX
                    elif  movX < 800 - radious:
                        movX+=2
# Rellenamos la pantalla de color negro
        window.fill((0,0,0))
# Dibujamos un círculo de color blanco en esa posición en el buffer
        pygame.draw.circle(window, RED, (movX,movY),radious,0)
   
 # Dibujamos como un rectángulo cada raqueta
        pygame.Rect(0, 550, 1200, 15)
        pygame.draw.rect(window, RED, (0,500,1200,15))
        pygame.draw.rect(window, RED, (200,300,50,200))

        # Actualizamos la pantalla
        pygame.display.update()


