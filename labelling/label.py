import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpldatacursor
import pygame
import os, random
from PIL import Image, ImageDraw

# color definitions
white = (255, 255, 255)
black = (0, 0, 0)
grey_8 = (51, 51, 51)
grey_4 = (153, 153, 153)
grey_2 = (204, 204, 204)
grey_1 = (230, 230, 230)
red = (180, 0, 0)
bright_red = (255, 0, 0)
green = (0, 180, 0)
bright_green = (0, 255, 0)
blue = (0, 0, 180)
bright_blue = (0, 0, 255)

# coordinates definitions
image_x = 44
image_y = 104  # (44, 104) coordinate
image_len = 512  # image is 512 x 512

# paths
input_path = 'cleaned_data'
output_path_data = 'labelled_data/data'
output_path_label = 'labelled_data/label'
output_path_history = 'labelled_data/history'

# global variables
first_click = True
line = np.zeros((2, 2))
line_list = []
image = ''


# create a blank img
def blank_image():
    array = np.zeros([256, 256, 3], dtype=np.uint8)
    array.fill(255)
    img = Image.fromarray(array)
    return img


# makes a button with
# msg = test in button
# x, y = coordinates of top left point of button
# w, h = width and height of button
# ic, ac = inactive and active colors
# action = function to be called
def button(msg,x,y,w,h,ic,ac,action = None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(display_surface, ac,(x,y,w,h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(display_surface, ic,(x,y,w,h))

    textsurf, textrect = text_objects(msg, "freesansbold.ttf", 20)
    textrect.center = ((x+(w/2)), (y+(h/2)))
    display_surface.blit(textsurf, textrect)


# define text object
def text_objects(text, font, size):
    smalltext = pygame.font.Font(font, size)
    textsurface = smalltext.render(text, True, black)
    return textsurface, textsurface.get_rect()


# save current image and load next image
def done():
    # save label and image to new path
    save_label()
    save_image()

    # reset line list
    global line_list
    line_list = []

    # set and load next image
    set_next_image()
    reset_display(display_surface)
    pygame.time.wait(200)  # 15ms delay to debounce


def draw_lines():
    img = blank_image()
    d = ImageDraw.Draw(img)
    for lines in line_list:
        p0 = tuple(lines[0]/2)  # half due to resize
        p1 = tuple(lines[1]/2)
        d.line([p0, p1], fill=black, width=1)
        draw_fuzzy(d, p0, p1)
    del d
    return img


def draw_fuzzy(d, p0, p1):
    # get line vector between 2 points
    vector = np.subtract(p0, p1)

    # compare dot product of vector with unit vectors of 2 directions
    ver = np.dot(vector, np.array([0,1]))
    hor = np.dot(vector, np.array([1,0]))
    if abs(ver) < abs(hor):
        # more horizontal than vertical
        # create grey lines above and below original line
        draw(d, p0, p1, (0, 1), grey_8, 1)
        draw(d, p0, p1, (0, -1), grey_8, 1)
        draw(d, p0, p1, (0, 2), grey_4, 1)
        draw(d, p0, p1, (0, -2), grey_4, 1)
        draw(d, p0, p1, (0, 3), grey_2, 1)
        draw(d, p0, p1, (0, -3), grey_2, 1)
        draw(d, p0, p1, (0, 4), grey_1, 1)
        draw(d, p0, p1, (0, -4), grey_1, 1)
    else:
        # more vertical than horizontal
        # create grey lines left and right of original line
        draw(d, p0, p1, (1, 0), grey_8, 1)
        draw(d, p0, p1, (-1, 0), grey_8, 1)
        draw(d, p0, p1, (2, 0), grey_4, 1)
        draw(d, p0, p1, (-2, 0), grey_4, 1)
        draw(d, p0, p1, (3, 0), grey_2, 1)
        draw(d, p0, p1, (-3, 0), grey_2, 1)
        draw(d, p0, p1, (4, 0), grey_1, 1)
        draw(d, p0, p1, (-4, 0), grey_1, 1)


def draw(d, p0, p1, v, fill, width):
    f0 = np.add(p0, v)
    f1 = np.add(p1, v)
    d.line([tuple(f0), tuple(f1)], fill=fill, width=width)


# redundant for now
def img2array(img):  # take in Image type argument
    # convert to grayscale
    img_gray = img.convert('L')
    array = np.asarray(img_gray.getdata(), dtype=np.uint8).reshape((256, 256))
    return array/255.0


def save_label():  # take in Image type argument
    # draw lines on a blank image
    label = draw_lines()

    # convert to grayscale
    label_gray = label.convert('L')

    # define path
    file_name = os.path.split(image)[1]
    img_path = os.path.join(output_path_label, file_name)

    # save image
    label_gray.save(img_path)


def save_image():
    # resize image
    img = cv2.imread(image)
    dim = (256, 256)
    new_img = cv2.resize(img, dim)

    # save image to path
    file_name = os.path.split(image)[1]
    img_path = os.path.join(output_path_data, file_name)
    cv2.imwrite(img_path, new_img)
    img_path = os.path.join(output_path_history, file_name)
    cv2.imwrite(img_path, img)

    # DELETE IMAGE FROM input_path FOLDER
    os.remove(image)


# undo previous operation
def undo():
    if len(line_list) != 0:
        line_list.pop()
        redraw_lines()
        pygame.time.wait(200)  # 15ms delay to debounce


# redraw all lines up till undo
def redraw_lines():
    reset_display(display_surface)
    for lines in line_list:
        pygame.draw.line(display_surface, black, lines[0], lines[1], 1)
        pygame.draw.circle(display_surface, red, lines[0], 2)
        pygame.draw.circle(display_surface, red, lines[1], 2)


# reset display to show blank slate
def reset_display(surface):
    # set the pygame window name
    pygame.display.set_caption('Label Tool')

    # completely fill the surface object
    # with white colour
    surface.fill(white)

    # create a surface object, image is drawn on it.
    img = pygame.image.load(image)
    surface.blit(img, (image_x, image_y))


def quit_program():
    # deactivates the pygame library
    pygame.quit()
    # quit the program.
    quit()


def set_next_image():
    file = random.choice(os.listdir(input_path))
    global image
    image = os.path.join(input_path, file)
    print(image)


# initiate pygame and give permission to use pygame's functionality.
pygame.init()
set_next_image()  # define image name

# create the display surface object
display_surface = pygame.display.set_mode((600, 660))
reset_display(display_surface)

# infinite loop
while True:
    # iterate over the list of Event objects returned by pygame.event.get()
    for event in pygame.event.get():
        # detect mouse clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            # checks if click occurs over image
            if image_x < x < image_x + image_len and image_y < y < image_y + image_len:
                pygame.draw.circle(display_surface, red, (x, y), 2)
                if first_click:
                    line[0] = (x, y)
                    first_click = False
                else:
                    line[1] = (x, y)
                    line_list.append(line.astype(int))
                    redraw_lines()
                    pygame.draw.line(display_surface, black, line_list[-1][0], line_list[-1][1], 1)
                    first_click = True

        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            if not first_click and image_x < x < image_x + image_len and image_y < y < image_y + image_len:
                redraw_lines()
                pygame.draw.line(display_surface, black, line[0], (x, y), 1)

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            if first_click:
                undo()
            else:
                first_click = True
                redraw_lines()

        # if event object type is QUIT then quit both pygame and program
        if event.type == pygame.QUIT:
            quit_program()

        button("DONE", 10, 10, 80, 50, green, bright_green, done)
        button("UNDO", 110, 10, 80, 50, blue, bright_blue, undo)
        button("QUIT", 210, 10, 80, 50, red, bright_red, quit_program)
        # Draws the surface object to the screen.
        pygame.display.update()
