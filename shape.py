import cv2
import numpy as np
import nidaqmx
import pyautogui
import threading
import time


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def get_random_node(goal_sample_rate, goal, width, height):
    if np.random.rand() < goal_sample_rate:
        return goal
    else:
        return Node(np.random.uniform(0, width), np.random.uniform(0, height))


def step_forward(nearest_node, random_node, step_length=35):
    theta = np.arctan2((random_node.y - nearest_node.y), (random_node.x - nearest_node.x))
    new_node = Node(nearest_node.x + step_length * np.cos(theta), nearest_node.y + step_length * np.sin(theta))
    return new_node


def grid_based_fitting(width1,height1,image_path=r"D:\Helmhots\real_blood vessels\2.png", grid_size=10):

    rectangles = []

    return rectangles


def is_obstacle(node, obstacles, safety_margin=4):
    for (x, y, w, h) in obstacles:
        if (x - safety_margin <= node.x <= x + w + safety_margin and
                y - safety_margin <= node.y <= y + h + safety_margin and
                    safety_margin<=node.x and node.y<=height):
            return True
    return False


def create_transparent_layer(width, height):
    return np.zeros((height, width, 4))


def draw_obstacles(layer, obstacles, alpha=255):
    for (x, y, w, h) in obstacles:
        cv2.rectangle(layer, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 0, alpha), -1)
    return layer


def draw_tree(layer, nodes, alpha=128):
    for node in nodes:
        if node.parent:
            cv2.line(layer, (int(node.x), int(node.y)), (int(node.parent.x), int(node.parent.y)), (0, 255, 0, alpha), 1)
    return layer


def draw_path(layer, nodes, goal, alpha=255, dis_accuracy=5):
    final_node = None
    min_distance = float('inf')
    path = [  (300, 80), (303, 90), (306, 100), (310, 110), (313, 119), (316, 129), (319, 139),
  (323, 149), (326, 159), (329, 169), (332, 179), (335, 188), (339, 198), (342, 208),
  (345, 218), (355, 218), (366, 218), (376, 218), (387, 218), (397, 218), (407, 218),
  (418, 218), (428, 218), (439, 218), (449, 218), (459, 218), (470, 218), (480, 218),
  (490, 218), (482, 224), (474, 230), (466, 236), (457, 242), (449, 248), (441, 254),
  (433, 259), (424, 265), (416, 271), (408, 277), (400, 283), (391, 289), (383, 295),
  (373, 304), (376, 313), (379, 323), (383, 333), (386, 343), (389, 352), (392, 362),
  (395, 372), (398, 382), (402, 392), (405, 401), (408, 411), (411, 421), (415, 431),
  (418, 442), (409, 436), (401, 430), (393, 423), (384, 417), (376, 411), (367, 405),
  (359, 399), (351, 393), (342, 387), (334, 380), (325, 374), (317, 368), (308, 362),
  (300, 356), (292, 362), (283, 368), (275, 374), (266, 380), (258, 387), (249, 393),
  (241, 399), (233, 405), (224, 411), (216, 417), (207, 423), (199, 430), (191, 436),
  (182, 442), (185, 431), (189, 421), (192, 411), (195, 401), (198, 392), (202, 382),
  (205, 372), (208, 362), (211, 352), (214, 343), (217, 333), (221, 323), (224, 313),
  (227, 304), (217, 295), (209, 289), (200, 283), (192, 277), (184, 271), (176, 265),
  (167, 259), (159, 254), (151, 248), (143, 242), (134, 236), (126, 230), (118, 224),
  (110, 218), (120, 218), (130, 218), (141, 218), (151, 218), (161, 218), (172, 218),
  (182, 218), (193, 218), (203, 218), (213, 218), (224, 218), (234, 218), (245, 218),
  (255, 218), (258, 208), (261, 198), (265, 188), (268, 179), (271, 169), (274, 159),
  (277, 149), (281, 139), (284, 129), (287, 119), (290, 110), (294, 100), (297, 90), (300, 80)]


    for i in range(len(path) - 1):
        cv2.line(layer, path[i], path[i + 1], (255, 0, 0, alpha), 2)
    return layer, path


def draw_start_goal(layer, start, goal, alpha=255):
    start_color = (0, 255, 0, alpha)
    goal_color = (0, 165, 255, alpha)
    cv2.circle(layer, (int(start.x), int(start.y)), 10, start_color, -1)
    cv2.circle(layer, (int(goal.x), int(goal.y)), 10, goal_color, -1)
    return layer


def virtual_add(background, rgba_layer):
    rgb = rgba_layer[:, :, :3]
    alpha = rgba_layer[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    result = (background * (1 - alpha) + rgb * alpha).astype(np.uint8)
    return result


def RRT(start, goal, width=640, height=480, goal_sample_rate=0.05, max_step=10000, dis_accuracy=20):
    nodes = [start]
    obstacles = grid_based_fitting(width,height)

    for i in range(max_step):
        random_node = get_random_node(goal_sample_rate, goal, width, height)
        nearest_node = min(nodes, key=lambda n: distance(random_node, n))
        new_node = step_forward(nearest_node, random_node)

        if is_obstacle(new_node, obstacles):
            continue
        else:
            new_node.parent = nearest_node
            nodes.append(new_node)
            if distance(goal, new_node) < dis_accuracy:
                print("路径规划成功！")
                break
    if distance(goal, nodes[-1]) >= dis_accuracy:
        print("达到最大步数")
    return nodes, obstacles


class HelmholtzControl:
    def __init__(self):
        self.device_name = "cDAQ9181-1E68CBFMod1"
        self.sample_rate = 10000
        self.running = False
        self.buffer_size = 10000
        self.params = [10, 10, 90.0, 0.0]  # [B, f, α, β]
        self.task = None

    def generate_signals(self, samples=10000):
        B, f, alpha_deg, beta_deg = self.params
        alpha = np.deg2rad(alpha_deg)
        beta = np.deg2rad(beta_deg)
        t = np.linspace(0, samples / self.sample_rate, samples)
        a0 = B / 3.816793893 * np.sin(2 * np.pi * f * t) * np.cos(alpha) * np.cos(beta) + B / 3.816793893 * np.sin(
            2 * np.pi * f * t + np.pi / 2) * np.sin(beta)
        a1 = B / 7.246376812 * np.sin(2 * np.pi * f * t) * np.cos(alpha) * np.sin(beta) + B / 7.246376812 * np.sin(
            2 * np.pi * f * t + np.pi / 2) * np.cos(beta)
        a2 = B / 17.24137931 * np.sin(2 * np.pi * f * t) * np.sin(alpha)
        return np.array([a0, a1, a2])

    def start(self):
        try:
            self.task = nidaqmx.Task()
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao{0}")
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao{1}")
            self.task.ao_channels.add_ao_voltage_chan(f"{self.device_name}/ao{2}")
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=self.buffer_size
            )
            self.task.out_stream.regen_mode = nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION

            data = self.generate_signals(self.buffer_size)
            self.task.write(data, auto_start=True)
            self.running = True
            print("亥姆霍兹线圈开始输出")
            return True
        except Exception as e:
            print(f"亥姆霍兹线圈启动失败: {e}")
            return False

    def update_beta(self, new_beta):
        if self.task and self.running:
            self.task.stop()
            self.params[3] = new_beta
            data = self.generate_signals(self.buffer_size)
            self.task.write(data, auto_start=True)


    def stop(self):
        self.running = False
        if self.task:
            try:
                self.task.stop()
                zero_data = np.zeros((3, 1000))
                self.task.write(zero_data, auto_start=False)
                self.task.close()
            except:
                pass
        print("亥姆霍兹线圈输出已停止")


def path_tracking(shared_data, path, helmholtz_control):
    i = 1
    while shared_data['running']:
        time.sleep(0.01)

        if i >= len(path):
            print("已到达路径终点")
            break

        current_x = shared_data['center_x']
        current_y = shared_data['center_y']
        target_x = path[i][0]
        target_y = path[i][1]
        beta = helmholtz_control.params[3]

        dx = target_x - current_x
        dy = target_y - current_y

        if abs(dx) > 10 or abs(dy) > 10:
            angle_rad = np.arctan2(-dy, dx)
            beta = np.degrees(np.pi-angle_rad)
            helmholtz_control.update_beta(beta)
        else:
            i += 1
            print(f"到达路径点 {i - 1}, 前进到路径点 {i}")

        print(
            f"当前beta: {beta:.1f}°, 当前目标点序号: {i}, 当前坐标: ({current_x}, {current_y}), 目标坐标: ({target_x}, {target_y})")


print('将鼠标移到截屏左上角，回车确认')
input()
x1, y1 = pyautogui.position()
print('将鼠标移到截屏右下角，回车确认')
input()
x2, y2 = pyautogui.position()
region = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

width = int(x2 - x1)
height = int(y2 - y1)
current_frame = None
previous_frame = None
tracker = cv2.TrackerCSRT_create()

start = Node(width-10, height-10)
goal = Node(30, 240)

nodes, obstacles = RRT(start, goal, width, height)

screenshot = pyautogui.screenshot(region=region)
pil_array = np.array(screenshot)
new_frame = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)
previous_frame = current_frame
current_frame = new_frame
bbox = cv2.selectROI(current_frame, False)
ok = tracker.init(current_frame, bbox)
layer = create_transparent_layer(width, height)
layer = draw_obstacles(layer, obstacles)

layer, reversed_paths = draw_path(layer, nodes, goal)
layer = draw_start_goal(layer, start, goal)
paths = reversed_paths[::-1]
print(paths)


shared_data = {
    'center_x': 0,
    'center_y': 0,
    'running': True
}

helmholtz_control = HelmholtzControl()

if helmholtz_control.start():

    monitor_thread = threading.Thread(target=path_tracking, args=(shared_data, paths, helmholtz_control))
    monitor_thread.daemon = True
    monitor_thread.start()

    while True:
        screenshot = pyautogui.screenshot(region=region)
        pil_array = np.array(screenshot)
        new_frame = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)
        previous_frame = current_frame
        current_frame = new_frame

        ok, bbox = tracker.update(current_frame)
        if ok:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            center_x = x + w // 2
            center_y = y + h // 2

            shared_data['center_x'] = center_x
            shared_data['center_y'] = center_y

        result = virtual_add(current_frame, layer)
        cv2.imshow('RRT Path Planning', result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            shared_data['running'] = False
            break

    helmholtz_control.stop()
    cv2.destroyAllWindows()
else:
    print("亥姆霍兹线圈启动失败，程序退出")