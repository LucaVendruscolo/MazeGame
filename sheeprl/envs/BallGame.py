import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from sheeprl.envs import BallDetection
import serial
import time
import cv2
from threading import Thread
import torchvision.transforms as T
import csv
import os
import threading
import torchvision.transforms.functional as TF
from PIL import Image

try:
    arduino = serial.Serial('COM3', 115200)
    time.sleep(2) 
    print("Connected to Arduino on COM3")
except serial.SerialException as e:
    print(f"Failed to connect to Arduino on COM3: {e}")


class BallEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.screen_width = 640
        self.screen_height = 480
        print("before cap")
        self.model, self.device = BallDetection.StartModelAndCap()

        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "position": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.screen_width, self.screen_height]), dtype=np.float32),
            "QR_position": gym.spaces.Box(low=np.array([600, 350]), high=np.array([730, 480]), dtype=np.float32),
            "ball_speed": gym.spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32),
            "QR_speed": gym.spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32),
            'waypoint_position1': gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.screen_width, self.screen_height]), dtype=np.float32),
            'waypoint_position2': gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.screen_width, self.screen_height]), dtype=np.float32)
        })

        self.lastKnownBallPosition = np.array([0, 0])
        self.fail_counter = 0
        self.totalReward = 0
        self.lastKnownQRPosition = np.array([0, 0])
        self.RewardForGames = []
        self.waypoints = np.array([[256, 405],
 [412, 405],
 [444, 364],
 [476, 405],
 [532, 390],
 [572, 333],
 [575, 288],
 [514, 272],
 [514, 234],
 [567, 228],
 [567, 197],
 [503, 180],
 [503, 112],
 [568, 80],
 [580, 50],
 [476, 50],
 [430, 90],
 [360, 106],
 [360, 202],
 [455, 234],
 [455, 302],
 [375, 310],
 [341, 366],
 [260, 252],
 [308, 138],
 [251, 50],
 [105, 65],
 [130,130],
 [206, 188],
 [170,265],
 [220,320],
 [205, 368],
 [115,375],
 [109, 234]])

        self.lastKnownProgress = 0.0
        self.video_stream = WebcamVideoStream().start()
        self.transform = T.Compose([T.ToTensor()])

        self.LastBallReading = np.array([[0, 0],[0, 0]])
        self.LastQRReading = np.array([[0, 0],[0, 0]])
        self.noProgressCount = 0
        self.time = time.perf_counter()
        self.maxProgress = 0
        self.xAxisOffset = 0
        self.yAxisOffset = 0

        self.cumulative_distances = [0.0]  
        for i in range(1, len(self.waypoints)):
            distance = np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1])
            self.cumulative_distances.append(self.cumulative_distances[-1] + distance)

        self.total_path_length = self.cumulative_distances[-1]



        self.reset()

    def step(self, action):
        low = -30
        high = 30

        motor1_deg, motor2_deg = low + (0.5 * (action + 1.0) * (high - low))
        self.move_motors(motor1_deg, motor2_deg)
        done = False

        timeSinceLastStep = time.perf_counter() - self.time
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < 1/45 - timeSinceLastStep:
            pass

        actionsPerSecond = 1.00 / (time.perf_counter() - self.time)
        if actionsPerSecond < 44:
            print("actions per second", actionsPerSecond)        
        self.time = time.perf_counter()

        first_frame = self.video_stream.read()

        qr_position_1, ball_position_1 = BallDetection.geoCoordinates(first_frame, self.model, self.device, self.transform)

        ball_speed, qr_speed = self.calculate_velocities(self.LastBallReading, ball_position_1, self.LastQRReading, qr_position_1)
        self.LastBallReading = ball_position_1
        self.LastQRReading = qr_position_1

        if ball_position_1 is not None:
            self.lastKnownBallPosition = np.array([(ball_position_1[0] + ball_position_1[2]) / 2, (ball_position_1[1] + ball_position_1[3]) / 2])
            self.fail_counter = 0

        else:
            print("no ball detection")
            self.fail_counter += 1

        if qr_position_1 is not None:
            self.lastKnownQRPosition = np.array([(qr_position_1[0] + qr_position_1[2]) / 2, (qr_position_1[1] + qr_position_1[3]) / 2])

        progress = self.calculate_progress_percentage(self.waypoints, self.lastKnownBallPosition)


        if self.maxProgress < progress:
            self.noProgressCount = 0
            self.maxProgress = progress
        else:
            self.noProgressCount += 1

        reward = (progress - self.lastKnownProgress) * 100
        self.lastKnownProgress = progress

        distance_traveled = progress * self.total_path_length

        next_waypoint_index = None
        for i in range(1, len(self.cumulative_distances)):
            if self.cumulative_distances[i] > distance_traveled:
                next_waypoint_index = i
                break

        # If no next waypoint is found, it means we are at the last waypoint
        if next_waypoint_index > len(self.waypoints) -3:
            next_waypoint1 = self.waypoints[-1]
            next_waypoint2 = self.waypoints[-2]
        else:
            next_waypoint1 = self.waypoints[next_waypoint_index]
            next_waypoint2 = self.waypoints[next_waypoint_index+1]



        observation = {
            'position': self.lastKnownBallPosition,
            'QR_position': self.lastKnownQRPosition,
            'ball_speed': ball_speed,
            'QR_speed': qr_speed,
            'waypoint_position1': next_waypoint1, 
            'waypoint_position2': next_waypoint2
        }     

        # first_frame_copy = np.copy(first_frame)
        # self.renderFrame(first_frame_copy, qr_speed, ball_speed)

        if self.fail_counter >= 1 or self.noProgressCount > 2000 or reward > 5 or self.isOutOfBounds(self.lastKnownBallPosition):
            print("fails", self.fail_counter, "no progress", self.noProgressCount, "skipped", reward, "is out of bounds", self.isOutOfBounds(self.lastKnownBallPosition))

            self.RewardForGames.append(self.totalReward)
            self.totalReward = 0
            done = True
            self.maxProgress = 0
            reward = 0
            self.noProgressCount = 0

        self.totalReward += reward

        return observation, reward, False, done, {}

    def calculate_velocities(self, ball_position_1, ball_position_2, qr_position_1, qr_position_2):
        ball_speed = np.array([0.0, 0.0])  # Initialize speeds to zero
        qr_speed = np.array([0.0, 0.0])


        # Check if the positions are not None before calculating speed
        if ball_position_1 is not None and ball_position_2 is not None:
            ball_speed = ((np.array([(ball_position_2[0] + ball_position_2[2]) / 2, (ball_position_2[1] + ball_position_2[3]) / 2]) -
                        np.array([(ball_position_1[0] + ball_position_1[2]) / 2, (ball_position_1[1] + ball_position_1[3]) / 2])))

        if qr_position_1 is not None and qr_position_2 is not None:
            qr_speed = ((np.array([(qr_position_2[0] + qr_position_2[2]) / 2, (qr_position_2[1] + qr_position_2[3]) / 2]) -
                        np.array([(qr_position_1[0] + qr_position_1[2]) / 2, (qr_position_1[1] + qr_position_1[3]) / 2])))

        return ball_speed, qr_speed



    def render(self, mode='human'):
        rgb_array = np.zeros((640, 480, 3), dtype=np.uint8)
        
        return rgb_array

    def reset(self, seed=None, return_info=False, options=None):
        print("reset ran start")
        self.DCSwitch(1)
        # qr_target_position = np.array([649, 380])
        # self.correct_motor_offsets(qr_target_position)    
        # time.sleep(0.2)
        # print(self.xAxisOffset, self.yAxisOffset)
        self.move_motors(10, 30)

        if len(self.RewardForGames) % 50 == 0:
            self.renderGraph()
        else:
            print("games until graph", 50 - len(self.RewardForGames) % 50)
        info = {}
        ball_position = None
        qr_position = None
        self.lastKnownProgress = 0
        self.lastKnownBallPosition = np.array([0, 0]) 
        self.lastKnownQRPosition = np.array([0, 0])
        ball_speed = np.array([0.0, 0.0])  
        qr_speed = np.array([0.0, 0.0])
        count = 0
        while True:
            frame = self.video_stream.read()
            qr_position, ball_position = BallDetection.geoCoordinates(frame, self.model, self.device,self.transform)
            if ball_position is not None and qr_position is not None:
                x1, y1, x2, y2 = ball_position
                ball_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                print(ball_center)
                if 234 < ball_center[0] < 281 and 400 < ball_center[1] < 472:
                    self.lastKnownBallPosition = ball_center
                    x1, y1, x2, y2 = qr_position
                    self.lastKnownQRPosition = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    print(self.lastKnownQRPosition)
                    break
            time.sleep(1)
            count += 1
            if count > 10:
                count = 0
                self.move_motors(38, 38)
                time.sleep(0.7)
                self.move_motors(38, -38)
                time.sleep(0.7)
                self.move_motors(-38, -38)
                time.sleep(0.7)
                self.move_motors(-38, 38)
                time.sleep(0.7)
                self.move_motors(38, 38)
                time.sleep(0.7)
                self.move_motors(20, 20)
                time.sleep(0.5)


        self.DCSwitch(0)

        observation = {
            'position': self.lastKnownBallPosition,
            'QR_position': self.lastKnownQRPosition,
            'ball_speed': ball_speed,
            'QR_speed': qr_speed,
            'waypoint_position1': [412, 405],
            'waypoint_position2': [444, 364]
        }

        
        #print("OBSERVATION", observation)


        self.LastBallReading = ball_position
        self.LastQRReading = qr_position
        return observation, info



    def get_centered_image(self, frame, ball_position, crop_size=64*3, final_size=64):
        h, w = frame.shape[:2]
        half_crop_size = crop_size // 2

        x_start = int(ball_position[0] - half_crop_size)
        y_start = int(ball_position[1] - half_crop_size)

        x_start = max(0, min(x_start, w - crop_size))
        y_start = max(0, min(y_start, h - crop_size))

        sub_image = frame[y_start:y_start + crop_size, x_start:x_start + crop_size]
        
        # Resize the image to the final size
        resized_image = cv2.resize(sub_image, (final_size, final_size))

        return resized_image
    
    def isOutOfBounds(self, BallPosition):
        
        if  545 < BallPosition[0] < 600 and 135 < BallPosition[1] < 160:
            return True
        
        if  390 < BallPosition[0] < 440 and 250 < BallPosition[1] < 270:
            return True

        return False


    def renderGraph(self):
        # plt.figure(figsize=(10, 6)) 
        # plt.plot(np.arange(len(self.RewardForGames)), self.RewardForGames, marker='o', linestyle='-')
        # plt.title('Rewards for Each Game')
        # plt.xlabel('Game Number')
        # plt.ylabel('Total Reward')
        # plt.grid(True)
        # plt.show(block=False)
        # plt.pause(6) 
        # plt.close()
        time.sleep(6)
        rgb_array = np.zeros((640, 480, 3), dtype=np.uint8)

        base_path = r'sheeprl\\sheeprl\\envs'
        base_filename = 'rewards_log.csv'
        full_path = os.path.join(base_path, base_filename)

        # Find a unique file name
        counter = 1
        while os.path.exists(full_path):
            new_filename = f'rewards_log_{counter}.csv'
            full_path = os.path.join(base_path, new_filename)
            counter += 1

        # Save the rewards to the unique file
        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for reward in self.RewardForGames:
                writer.writerow([reward])
                        
        return rgb_array
    
    def renderFrame(self, frame,QR_speed,ball_speed):
        waypoints_np = np.array(self.waypoints, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [waypoints_np], isClosed=False, color=(0, 0, 255), thickness=2)

        total_length = sum(np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) for i in range(len(self.waypoints)-1))
        progress_length = total_length * self.lastKnownProgress
        accumulated_length = 0

        for i in range(len(self.waypoints) - 1):
            start, end = np.array(self.waypoints[i], np.float32), np.array(self.waypoints[i+1], np.float32)
            segment_length = np.linalg.norm(end - start)

            if accumulated_length + segment_length >= progress_length:
                segment_progress = (progress_length - accumulated_length) / segment_length
                progress_point = start + segment_progress * (end - start)

                if i > 0:
                    prev_path_np = np.array(self.waypoints[:i+1], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [prev_path_np], isClosed=False, color=(0, 255, 0), thickness=2)

                cv2.line(frame, tuple(self.waypoints[i].astype(int)), tuple(progress_point.astype(int)), (0, 255, 0), 2)
                break
            accumulated_length += segment_length

        if self.lastKnownBallPosition is not None:
            cv2.circle(frame, tuple(self.lastKnownBallPosition.astype(int)), 5, (255, 0, 0), -1)
            ball_vector = ball_speed * 1  
            cv2.line(frame, tuple(self.lastKnownBallPosition.astype(int)), tuple((self.lastKnownBallPosition + ball_vector).astype(int)), (255, 165, 0), 2)

        if self.lastKnownQRPosition is not None:
            cv2.circle(frame, tuple(self.lastKnownQRPosition.astype(int)), 5, (0, 255, 0), -1)
            qr_vector = QR_speed * 1  
            cv2.line(frame, tuple(self.lastKnownQRPosition.astype(int)), tuple((self.lastKnownQRPosition + qr_vector).astype(int)), (0, 255, 255), 2)

        cv2.imshow("Frame with Path Overlay", frame)
        cv2.waitKey(1)

    def close(self):
        plt.close()
        self.move_motors(0, 0)
        self.video_stream.stop()
        cv2.destroyAllWindows()
        if arduino is not None:
            arduino.close()

    def move_motors(self, motor1_deg, motor2_deg):
        motor1_deg += self.xAxisOffset
        motor2_deg += self.yAxisOffset
        command = f"Move,{-motor1_deg},{-motor2_deg}\n"
        arduino.write(command.encode())

    def DCSwitch(self, onOrOff):
        onOrOff = int(onOrOff)
        if onOrOff == 1:
            command = f"MotorON\n"
            arduino.write(command.encode())
        else:
            command = f"MotorOFF\n"
            arduino.write(command.encode())
    

    def closest_point_on_segment(self, a, b, p):
        ab = b - a
        ap = p - a
        ab_norm = ab / np.linalg.norm(ab)

        projection = np.dot(ap, ab_norm) * ab_norm
        closest_point = a + projection

        if np.dot(projection, ab) > np.linalg.norm(ab)**2 or np.dot(projection, ab) < 0:
            if np.linalg.norm(p - a) < np.linalg.norm(p - b):
                closest_point = a
            else:
                closest_point = b
        return closest_point

    def calculate_progress_percentage(self, waypoints, ball_position):
        total_path_length = sum(np.linalg.norm(np.array(self.waypoints[i+1]) - np.array(self.waypoints[i])) for i in range(len(self.waypoints)-1))

        closest_distance = float('inf')
        accumulated_path_length = 0
        progress = 0

        for i in range(len(waypoints) - 1):
            start, end = waypoints[i], waypoints[i + 1]
            segment_length = np.linalg.norm(end - start)

            closest_point = self.closest_point_on_segment(start, end, np.array(ball_position))
            distance_to_start = np.linalg.norm(closest_point - start)
            
            # Calculate the distance from the ball to the closest point on the path segment
            distance_to_ball = np.linalg.norm(ball_position - closest_point)
            
            if distance_to_ball < closest_distance:
                closest_distance = distance_to_ball
                progress = (accumulated_path_length + distance_to_start) / total_path_length
            
            accumulated_path_length += segment_length

        return progress



class WebcamVideoStream:
    def __init__(self, src=1, width=640, height=480, fps=60):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify if the resolution has been set correctly
        actual_width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('Requested resolution: {} x {}'.format(width, height))
        print('Actual resolution: {} x {}'.format(int(actual_width), int(actual_height)))
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()





#python sheeprl.py exp=dreamer_v3 env=BallGame algo.mlp_keys.encoder=[position,QR_position,ball_speed,QR_speed,waypoint_position1,waypoint_position2] algo.mlp_keys.encoder=[position,QR_position,ball_speed,QR_speed,waypoint_position1,waypoint_position2] algo.cnn_keys.encoder=[] algo.cnn_keys.decoder=[]