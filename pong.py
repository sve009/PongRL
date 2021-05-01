import pygame

from agent import *

pygame.init()

screen = pygame.display.set_mode([640, 480])

b_x = 0
b_y = 240
b_xv = 1
b_yv = 1

y1 = 0
y2 = 0

a1 = 0
a2 = 0

running = True

model = create_model()

def update_paddle_1(a):
    global y1

    if a == 0:
        y1 += 0
    elif a == 1:
        y1 += 2
        if y1 > 400:
            y1 = 400
    elif a == 2:
        y1 -= 2
        if y1 < 0:
            y1 = 0

def update_paddle_2(a):
    global y2

    if a == 0:
        y2 += 0
    elif a == 1:
        y2 += 2
        if y2 > 400:
            y2 = 400
    elif a == 2:
        y2 -= 2
        if y2 < 0:
            y2 = 0

def update(a1, a2):
    global b_x, b_y
    global b_xv, b_yv
    global y1, y2
    global running

    reward1 = -1
    reward2 = -1

    if a1 == 1 and b_y > y1 + 80:
        reward1 = 5
    elif a1 == 2 and b_y < y1:
        reward1 = 5

    if a2 == 1 and b_y > y2 + 80:
        reward2 = 5
    elif a2 == 2 and b_y < y2:
        reward2 = 5
    
    b_x += b_xv
    b_y += b_yv

    #b_xv += 1
    #b_yv += 1

    update_paddle_1(a1)
    update_paddle_2(a2)

    if b_y < 0:
        b_y = 0
        b_yv *= -1
    elif b_y > 460:
        b_y = 460
        b_yv *= -1

    if b_x < 20 and b_y > y1 - 20 and b_y < y1 + 100:
        reward1 = 50
        b_xv *= -1
    elif b_x > 600 and b_y > y2 - 20 and b_y < y2 + 100:
        reward2 = 50
        b_xv *= -1

    if b_x < 0:
        reward1 = -1000
        b_x = 320
        b_y = 240
    elif b_x > 620:
        reward2 = -1000
        b_x = 320
        b_y = 240

    return (reward1, reward2)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            a1 = 2
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            a1 = 1
        elif event.type == pygame.KEYUP:
            a1 = 0

    pb_x = b_x
    pb_y = b_y
    pb_xv = b_xv
    pb_yv = b_yv
    p_y1 = y1
    p_y2 = y2

    xd = b_x - 320
    rx = 320 + (-1 * xd)

    pb_x2 = rx
    pb_xv2 = -1 * b_xv

    in11 = tf.convert_to_tensor([rx, b_y, -1 * b_xv, b_yv, y1])
    in11 = tf.expand_dims(in11, axis=0)
    in21 = tf.convert_to_tensor([pb_x2, pb_y, pb_xv2, pb_yv, p_y1])
    in21 = tf.expand_dims(in21, axis=0)
    in12 = tf.convert_to_tensor([b_x, b_y, b_xv, b_yv, y2])
    in12 = tf.expand_dims(in12, axis=0)
    in22 = tf.convert_to_tensor([pb_x, pb_y, pb_xv, pb_yv, p_y2])
    in22 = tf.expand_dims(in22, axis=0)

    a1 = return_action(model, in11)
    a2 = return_action(model, in21)

    (r1, r2) = update(a1, a2)

    train_step(model, in11, a1, in21, r1, .1)
    train_step(model, in12, a2, in22, r2, .1)

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (255, 0, 0), (0, y1, 20, 80))
    pygame.draw.rect(screen, (255, 0, 0), (620, y2, 20, 80))
    pygame.draw.circle(screen, (255, 0, 0), (b_x, b_y), 20)

    pygame.display.flip()

pygame.quit()
