import pyautogui
import time
import random

while True:
    # Obtém as dimensões da tela
    screen_width, screen_height = pyautogui.size()

    # Calcula uma posição aleatória na tela
    random_x = random.randint(0, screen_width)
    random_y = random.randint(0, screen_height)

    # Move o mouse para a posição aleatória
    pyautogui.moveTo(random_x, random_y, duration=0.5)  # opcional: adiciona uma animação de movimento

    # Espera um minuto
    time.sleep(1)
