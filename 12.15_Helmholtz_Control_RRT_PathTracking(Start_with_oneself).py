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


def create_obstacles(width, height):
    obstacles = []
    w1, h1 = 5, height - 100
    w2, h2 = 5, height - 100
    w3, h3 = 5, 100
    w4, h4 = 200, 5

    x_1 = 100
    y_1 = 100
    x_2 = 200
    y_2 = 0
    x_3 = width - 100
    y_3 = height - 100
    x_4 = width - 200
    y_4 = height - 200
    obstacles.append((x_1, y_1, w1, h1))
    obstacles.append((x_2, y_2, w2, h2))
    obstacles.append((x_3, y_3, w3, h3))
    obstacles.append((x_4, y_4, w4, h4))
    # wall

    obstacles.append((250, height - 150, 40, 40))
    obstacles.append((450, height - 400, 80, 80))
    obstacles.append((400, height - 100, 80, 80))

    return obstacles


def is_obstacle(height, node, obstacles, safety_margin=30):
    for (x, y, w, h) in obstacles:
        if (x - safety_margin <= node.x <= x + w + safety_margin and
                y - safety_margin <= node.y <= y + h + safety_margin and
                safety_margin <= node.x and node.y <= height):
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


def draw_path(layer, nodes, goal, alpha=255, dis_accuracy=20):
    final_node = None
    min_distance = float('inf')
    path = []
    for node in nodes:
        dist = distance(node, goal)
        if dist < min_distance:
            min_distance = dist
            final_node = node
    if final_node and distance(final_node, goal) < dis_accuracy:
        current = final_node
        while current:
            path.append((int(current.x), int(current.y)))
            current = current.parent
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


def RRT(start, goal, width, height, goal_sample_rate=0.05, max_step=10000, dis_accuracy=20):
    nodes = [start]
    obstacles = create_obstacles(width, height)

    for i in range(max_step):
        random_node = get_random_node(goal_sample_rate, goal, width, height)
        nearest_node = min(nodes, key=lambda n: distance(random_node, n))
        new_node = step_forward(nearest_node, random_node)

        if is_obstacle(height,new_node, obstacles):
            continue
        else:
            new_node.parent = nearest_node
            nodes.append(new_node)
            if distance(goal, new_node) < dis_accuracy:
                print("路径规划成功！")
                break
    if distance(goal, nodes[-1]) >= dis_accuracy:
        print("达到最大步数，未能规划路径")

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


class PathTracker:
    def __init__(self, paths):
        self.paths = paths
        self.current_path_index = 0
        self.reached_final = False

    def update_current_position(self, current_pos):
        if self.reached_final:
            return

        target_pos = self.paths[self.current_path_index]
        distance_to_target = np.sqrt((current_pos[0] - target_pos[0]) ** 2 +
                                     (current_pos[1] - target_pos[1]) ** 2)
        if distance_to_target < 15:  # 到达阈值
            if self.current_path_index < len(self.paths) - 1:
                self.current_path_index += 1
                print(f"到达路径点 {self.current_path_index - 1}, 前进到路径点 {self.current_path_index}")
            else:
                print("已到达路径终点")
                self.reached_final = True

    def get_current_target(self):
        if self.current_path_index < len(self.paths):
            return self.paths[self.current_path_index]
        else:
            return self.paths[-1]  # 返回最后一个点

    def is_finished(self):
        return self.reached_final


def path_tracking(shared_data, path_tracker, helmholtz_control):
    while shared_data['running'] and not path_tracker.is_finished():
        time.sleep(0.02)

        current_x = shared_data['center_x']
        current_y = shared_data['center_y']

        path_tracker.update_current_position((current_x, current_y))
        target_x, target_y = path_tracker.get_current_target()

        dx = target_x - current_x
        dy = target_y - current_y

        if np.sqrt(dx ** 2 + dy ** 2) > 10:
            angle_rad = np.arctan2(-dy, dx)
            beta = np.degrees(np.pi - angle_rad)
            helmholtz_control.update_beta(beta)

        print(f"当前目标点: {path_tracker.current_path_index}, 当前位置: ({current_x:.0f}, {current_y:.0f}，目标位置: ({target_x:.0f}, {target_y:.0f}，距离: {np.sqrt(dx ** 2 + dy ** 2):.1f}")

    print("路径跟踪完成")


def main():
    print('将鼠标移到截屏左上角，回车确认')
    input()
    x1, y1 = pyautogui.position()
    print('将鼠标移到截屏右下角，回车确认')
    input()
    x2, y2 = pyautogui.position()

    region = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    width = int(x2 - x1)
    height = int(y2 - y1)

    goal = Node(width - 1, 1)

    print("请选择要追踪的物体")
    screenshot = pyautogui.screenshot(region=region)
    pil_array = np.array(screenshot)
    current_frame = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)

    bbox = cv2.selectROI("选择要追踪的物体", current_frame, False)
    cv2.destroyWindow("选择要追踪的物体")

    tracker = cv2.TrackerCSRT_create()

    x, y, w, h = [int(v) for v in bbox]
    start_x = x + w // 2
    start_y = y + h // 2
    start = Node(start_x, start_y)

    print(f"路径规划起点: ({start_x}, {start_y})")
    print(f"路径规划终点: ({goal.x}, {goal.y})")

    print("正在进行路径规划...")
    nodes, obstacles = RRT(start, goal, width, height)

    layer = create_transparent_layer(width, height)
    layer = draw_obstacles(layer, obstacles)
    layer, reversed_paths = draw_path(layer, nodes, goal)
    layer = draw_start_goal(layer, start, goal)

    paths = reversed_paths[::-1]

    if not paths:
        print("未能找到有效路径！")
        return

    print(f"路径点 {paths}")

    path_tracker = PathTracker(paths)

    shared_data = {
        'center_x': start_x,
        'center_y': start_y,
        'running': True
    }

    helmholtz_control = HelmholtzControl()

    if helmholtz_control.start():
        monitor_thread = threading.Thread(target=path_tracking,args=(shared_data, path_tracker, helmholtz_control))
        monitor_thread.daemon = True
        monitor_thread.start()

        while shared_data['running']:
            screenshot = pyautogui.screenshot(region=region)
            pil_array = np.array(screenshot)
            new_frame = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)

            ok, bbox = tracker.update(new_frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2
                center_y = y + h // 2

                shared_data['center_x'] = center_x
                shared_data['center_y'] = center_y

                cv2.rectangle(new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(new_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                target_x, target_y = path_tracker.get_current_target()
                cv2.circle(new_frame, (int(target_x), int(target_y)), 8, (255, 255, 0), -1)
                cv2.arrowedLine(new_frame, (center_x, center_y),
                                (int(target_x), int(target_y)), (255, 0, 255), 2)

            result = virtual_add(new_frame, layer)
            cv2.imshow('RRT Path Planning', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                shared_data['running'] = False
                break

        helmholtz_control.stop()
        cv2.destroyAllWindows()
        print("程序已退出")
    else:
        print("亥姆霍兹线圈启动失败，程序退出")


if __name__ == "__main__":
    main()