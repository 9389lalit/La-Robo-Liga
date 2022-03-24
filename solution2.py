# This is just a basic template for solution

import LRL_main_arena
import gym
import time
import pybullet as p
import cv2
import os
import numpy as np
import math


def solve(ball_color, goal_color):
    detect_ball(ball_color)
    move_to_ball(ball_color)
    targetCX = detect_goal(goal_color, 1)
    height_difference, alignment = align(goal_color)
    while height_difference > 15:
        targetCX = detect_goal(goal_color, alignment)
        height_difference, alignment = align(goal_color)
    targetCX = detect_goal(goal_color, alignment)
    move_to_goal(goal_color)
    move_back(goal_color, targetCX)


def detect_ball(ball_color):
    rotate_velocity = 5
    i = 0
    flag = 0
    while True:
        if i % 10 == 0:
            img, ball_mask, area, cx, cy = get_image(ball_color)
        p.stepSimulation()
        cv2.imshow("camera 1", img)
        cv2.imshow("camera 2", ball_mask)
        cv2.waitKey(1)
        i += 1
        if area == -1 and cx == -1 and cy == -1:
            env.move_husky(rotate_velocity, -rotate_velocity, rotate_velocity, -rotate_velocity)
        elif cx <= 297:
            env.move_husky(-rotate_velocity*(298-cx)/100, rotate_velocity*(298-cx)/100, -rotate_velocity*(298-cx)/100, rotate_velocity*(298-cx)/100)
        elif 297 < cx < 300:
            env.move_husky(0, 0, 0, 0)
            print("BALL DETECTED")
            p.stepSimulation()
            break
        else:
            env.move_husky(rotate_velocity*(cx-299)/100, -rotate_velocity*(cx-299)/100, rotate_velocity*(cx-299)/100, -rotate_velocity*(cx-299)/100)


def move_to_ball(ball_color):
    i = 0
    low_range = 10000
    up_range = 12000

    balls_in_path, area_list = check_balls(ball_color, "ball")
    print(balls_in_path)
    if not balls_in_path:
        env.open_husky_gripper()

    while True:
        if i % 10 == 0:
            img, ball_mask, area, cx, cy = get_image(ball_color)

        p.stepSimulation()
        cv2.imshow("camera 1", img)
        cv2.imshow("camera 2", ball_mask)
        cv2.waitKey(1)

        if i % 20 == 0 and balls_in_path:
            print(i, "11111")
            avoid_collision(balls_in_path, "ball")
            detect_ball(ball_color)
            if not balls_in_path:
                env.open_husky_gripper()

        if low_range < area < up_range:
            low_range = up_range = 0
            env.move_husky(0, 0, 0, 0)
            detect_ball(ball_color)
        if area >= 45000:
            env.move_husky(0, 0, 0, 0)
            env.close_husky_gripper()
            print("REACHED TO BALL")
            break
        else:
            print(i, "moving")
            env.move_husky(5, 5, 5, 5)
        p.stepSimulation()
        i += 1


def check_balls(ball_color, status):
    print("hi")
    color_list = []
    area_list = []
    removing_colors = []
    for color in ball_colors_hsv:
        img, img_mask, area, cx, cy = get_image(color)
        # p.stepSimulation()
        # cv2.imshow("camera 1", img)
        # cv2.imshow("camera 2", img_mask)
        # cv2.waitKey(1)
        if color == ball_color:
            ball_area = area
        else:
            if area==-1 and cx==-1 and cy==-1:
                pass
            else:
                if 270 < cx < 330:
                    color_list.append(color)
                    area_list.append(area)

    print(color_list)
    print(area_list)
    print(ball_area)
    if status == "ball":
        for i in range(len(color_list)):
            if area_list[i] < ball_area:
                removing_colors.append(color_list[i])

        for color in removing_colors:
            color_list.remove(color)

    return color_list, area_list


def avoid_collision(color_list, status):
    print("hello")
    threshold = 10000 if status == "ball" else 4000
    for color in color_list:
        img, img_mask, area, cx, cy = get_image(color)
        # p.stepSimulation()
        # cv2.imshow("camera 1", img)
        # cv2.imshow("camera 2", img_mask)
        # cv2.waitKey(1)
        if area > threshold:
            print("22222")
            env.close_husky_gripper()
            if cx > 298:
                for i in range(2500):
                    env.move_husky(-0.5, 0.5, -0.5, 0.5)
                    p.stepSimulation()
            else:
                for i in range(2500):
                    env.move_husky(0.5, -0.5, 0.5, -0.5)
                    p.stepSimulation()
            for i in range(5000):
                env.move_husky(0.5, 0.5, 0.5, 0.5)
                p.stepSimulation()
            if cx > 298:
                for i in range(2500):
                    env.move_husky(0.5, -0.5, 0.5, -0.5)
                    p.stepSimulation()
            else:
                for i in range(2500):
                    env.move_husky(-0.5, 0.5, -0.5, 0.5)
                    p.stepSimulation()
            env.move_husky(0, 0, 0, 0)
            color_list.remove(color)


def detect_goal(goal_color, alignment):
    rotate_velocity = 2
    i = 0
    while True:
        if i % 10 == 0:
            img, goal_mask, area, cx, cy = get_image(goal_color)
        p.stepSimulation()
        cv2.imshow("camera 1", img)
        cv2.imshow("camera 2", goal_mask)
        cv2.waitKey(1)
        i += 1
        if area == -1 and cx == -1 and cy == -1:
            env.move_husky(rotate_velocity*alignment, -rotate_velocity*alignment, rotate_velocity*alignment, -rotate_velocity*alignment)
        elif cx < 297:
            env.move_husky(-rotate_velocity*(298-cx)/40, rotate_velocity*(298-cx)/40, -rotate_velocity*(298-cx)/40, rotate_velocity*(298-cx)/40)
        elif 297 <= cx <= 299:
            env.move_husky(0, 0, 0, 0)
            print("GOAL DETECTED")
            return cx
        else:
            env.move_husky(rotate_velocity*(cx-299)/40, -rotate_velocity*(cx-299)/40, rotate_velocity*(cx-299)/40, -rotate_velocity*(cx-299)/40)


def align(goal_color):
    img = env.get_camera_image()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # left_part = hsv[:, :w // 2]
    # right_part = hsv[:, w // 2:]
    kernel = np.ones((2, 2), np.uint8)
    complete_mask = cv2.inRange(hsv, np.array(goal_color[0]), np.array(goal_color[1]))
    complete_erosion = cv2.erode(complete_mask, kernel, iterations=2)
    complete_dilation = cv2.dilate(complete_erosion, kernel, iterations=2)
    complete_contours, _ = cv2.findContours(complete_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    complete_contour = complete_contours[len(complete_contours)-1]
    cv2.drawContours(img, [complete_contour], 0, (0, 255, 0), 3)

    (h, w) = complete_dilation.shape
    left_part = complete_dilation[:, :w // 2]
    right_part = complete_dilation[:, w // 2:]
    # left_mask = cv2.inRange(left_part, np.array(goal_color[0]), np.array(goal_color[1]))
    # right_mask = cv2.inRange(right_part, np.array(goal_color[0]), np.array(goal_color[1]))
    # left_erosion = cv2.erode(left_mask, kernel, iterations=2)
    # left_dilation = cv2.dilate(left_erosion, kernel, iterations=2)
    left_contours, _ = cv2.findContours(left_part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    left_contour = left_contours[len(left_contours)-1]
    left_contour = sorted(left_contour, key=lambda x: x[0][1])
    # right_erosion = cv2.erode(right_mask, kernel, iterations=2)
    # right_dilation = cv2.dilate(right_erosion, kernel, iterations=2)
    right_contours, _ = cv2.findContours(right_part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    right_contour = right_contours[len(right_contours)-1]
    right_contour = sorted(right_contour, key=lambda x: x[0][1])
    height_difference = left_contour[len(left_contour)-1][0][1] - right_contour[len(right_contour)-1][0][1]
    # env.move_husky(0.5, 0.5, 0.5, 0.5)
    complete_contour = sorted(complete_contour, key=lambda x: x[0][0])
    pole_distance = complete_contour[0][0][0] - complete_contour[len(complete_contour)-1][0][0]
    alignment = 1
    print("POLE", pole_distance)
    print("HEIGHT", height_difference)
    if math.fabs(pole_distance) < 100:
        print("POLE")
        for i in range(2000):
            if height_difference > 0:
                env.move_husky(1, -1, 1, -1)
                alignment = -1
            else:
                env.move_husky(-1, 1, -1, 1)
                alignment = 1
            p.stepSimulation()
        for i in range(5000):
            env.move_husky(1, 1, 1, 1)
            p.stepSimulation()
    else:
        print("HEIGHT")

        for i in range(height_difference*100):
            if height_difference > 0:
                alignment = -1
                env.move_husky(0.5, -0.5, 0.5, -0.5)
            else:
                alignment = 1
                env.move_husky(-0.5, 0.5, -0.5, 0.5)
            p.stepSimulation()
        for i in range(height_difference*150):
            env.move_husky(0.5, 0.5, 0.5, 0.5)
            p.stepSimulation()
    env.move_husky(0, 0, 0, 0)
    p.stepSimulation()
    cv2.imshow("camera 1", img)
    # cv2.imshow("camera 2", left_mask)
    # cv2.imshow("camera 3", right_mask)
    cv2.waitKey(1)
    return math.fabs(height_difference), alignment


def move_to_goal(goal_color):
    i = 0
    balls_in_path, area_list = check_balls(ball_colors_hsv[goal_colors_hsv.index(goal_color)], "goal")
    while True:
        if i % 10 == 0:
            img, goal_mask, area, cx, cy = get_image(goal_color)
        p.stepSimulation()
        cv2.imshow("camera 1", img)
        cv2.imshow("camera 2", goal_mask)
        cv2.waitKey(1)

        area_list = sorted(area_list, reverse=True)
        iterations = 50
        if area_list:
            max_area = area_list[0]
            if max_area < 2500:
                iterations = 150

        if i % iterations == 0 and balls_in_path:
            avoid_collision(balls_in_path, "goal")
            detect_goal(goal_color, 1)
        i += 1
        if cy <= 25:
            env.move_husky(0, 0, 0, 0)
            env.open_husky_gripper()
            break
        env.move_husky(6, 6, 6, 6)


def move_back(goal_color, targetCX):
    i = 0
    while True:
        if i % 10 == 0:
            img, goal_mask, area, cx, cy = get_image(goal_color)
            # targetCX = cx if i == 0 else targetCX
        p.stepSimulation()
        cv2.imshow("camera 1", img)
        cv2.imshow("camera 2", goal_mask)
        # cv2.imshow("camera 3", left_part)
        cv2.waitKey(1)
        i += 1
        # print(targetCX, cx, cy)
        if cy > 100 and targetCX - 10 <= cx <= targetCX + 10:
            env.move_husky(0, 0, 0, 0)
            env.close_husky_gripper()
            break
        env.move_husky(-3, -3, -3, -3)


def get_image(color):
    img = env.get_camera_image()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_mask = cv2.inRange(hsv, np.array(color[0]), np.array(color[1]))
    # erosion dilation
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(image_mask, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # print("XXX")
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        area = cv2.contourArea(contours[len(contours) - 1])
        M = cv2.moments(contours[len(contours) - 1])
        if M['m00'] == 0:
            cx = int(M['m10'] / (M['m00'] + 0.0001))
            cy = int(M['m01'] / (M['m00'] + 0.0001))
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 3, (255, 0, 0), 5)
        # print(area, cx, cy)
    else:
        area, cx, cy = -1, -1, -1

    return img, image_mask, area, cx, cy


if __name__ == '__main__':
    parent_path = os.path.dirname(os.getcwd()) # This line is to switch the directories for getting resources.
    os.chdir(parent_path)      # you don't need to change anything in here.

    # new_balls_location = dict({
    #     'red': [6, 6, 1.5],
    #     'yellow': [-6, 6, 1.5],
    #     'blue': [-6, -6, 1.5],
    #     'purple': [6, -6, 1.5]
    # })
    #
    # new_husky_pos = [0, 0, 0.3]
    # new_husky_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
    #
    # env = gym.make(
    #     "la_robo_liga_arena-v0",
    #     ball_locations=new_balls_location,
    #     husky_pos=new_husky_pos,
    #     husky_orn=new_husky_orn
    # )

    env = gym.make("la_robo_liga_arena-v0")    # This loads the arena.
    # Initialize your global variables/constants here.

    ball_colors_hsv = [[[80, 100, 100], [100, 255, 255]], [[25, 80, 80], [35, 200, 200]],
                       [[10, 100, 100], [25, 255, 255]], [[140, 100, 100], [170, 230, 230]]]
    goal_colors_hsv = [[[100, 100, 100], [125, 255, 200]], [[25, 150, 120], [35, 255, 255]],
                       [[0, 100, 100], [9, 255, 255]], [[140, 135, 0], [160, 255, 255]]]

    # while True:       # main loop to run the simulation.

    p.stepSimulation()

    solve(ball_colors_hsv[0], goal_colors_hsv[0])  # blue ball
    solve(ball_colors_hsv[1], goal_colors_hsv[1])  # yellow ball
    solve(ball_colors_hsv[2], goal_colors_hsv[2])  # orange ball
    solve(ball_colors_hsv[3], goal_colors_hsv[3])  # purple ball

