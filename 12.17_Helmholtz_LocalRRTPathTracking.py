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


def get_random_node(goal_sample_rate, goal, min_x, max_x, min_y, max_y):
    if np.random.rand() < goal_sample_rate:
        return goal
    else:
        return Node(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))


def step_forward(nearest_node, random_node, step_length=35):
    theta = np.arctan2((random_node.y - nearest_node.y), (random_node.x - nearest_node.x))
    new_node = Node(nearest_node.x + step_length * np.cos(theta),
                    nearest_node.y + step_length * np.sin(theta))
    return new_node


def create_obstacles(width, height):
    obstacles = []

    obstacles.append((100, 150, 80, 50))#(x,y,w,h)
    obstacles.append((200, 300, 60, 60))
    obstacles.append((250, 160, 40, 40))
    obstacles.append((350, height - 60, 40, 80))
    obstacles.append((370, height - 300, 80, 40))
    obstacles.append((530, 350, 60, 40))

    return obstacles


def get_obstacles_in_view(obstacles, view_center, view_radius):
    view_obstacles = []
    cx, cy = view_center
    for (x, y, w, h) in obstacles:
        closest_x = max(x, min(cx, x + w))
        closest_y = max(y, min(cy, y + h))
        distance = np.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)

        if distance <= view_radius:
            visible_x = max(x, cx - view_radius)
            visible_y = max(y, cy - view_radius)
            visible_w = min(x + w, cx + view_radius) - visible_x
            visible_h = min(y + h, cy + view_radius) - visible_y
            if visible_w > 0 and visible_h > 0:
                view_obstacles.append((visible_x, visible_y, visible_w, visible_h))

    return view_obstacles


def is_obstacle(node, obstacles, safety_margin=20):
    for (x, y, w, h) in obstacles:
        if (x - safety_margin <= node.x <= x + w + safety_margin and
                y - safety_margin <= node.y <= y + h + safety_margin):
            return True
    return False


def create_transparent_layer(width, height):
    return np.zeros((height, width, 4), dtype=np.uint8)


def draw_obstacles(layer, obstacles, alpha=255):
    for (x, y, w, h) in obstacles:
        cv2.rectangle(layer, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 0, alpha), -1)
    return layer


def virtual_add(background, rgba_layer):
    rgb = rgba_layer[:, :, :3]
    alpha = rgba_layer[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    result = (background * (1 - alpha) + rgb * alpha).astype(np.uint8)
    return result


def local_RRT(start,width,height,goal, obstacles, view_center, view_radius, goal_sample_rate=0.1, max_step=1000, dis_accuracy=5):
    min_x = max(0, view_center[0] - view_radius)
    max_x = min(view_center[0] + view_radius, width)
    min_y = max(0, view_center[1] - view_radius)
    max_y = min(view_center[1] + view_radius, height)

    nodes = [start]

    for i in range(max_step):
        random_node = get_random_node(goal_sample_rate, goal, min_x, max_x, min_y, max_y)
        nearest_node = min(nodes, key=lambda n: distance(random_node, n))
        new_node = step_forward(nearest_node, random_node, step_length=25)

        new_node.x = max(min_x, min(max_x, new_node.x))
        new_node.y = max(min_y, min(max_y, new_node.y))

        if is_obstacle(new_node, obstacles):
            continue

        new_node.parent = nearest_node
        nodes.append(new_node)

        if distance(goal, new_node) < dis_accuracy:
            print(f"局部路径规划成功！步数: {i}")
            break

    if distance(goal, nodes[-1]) >= dis_accuracy:
        print("达到最大步数，未能找到局部路径")
        nearest_to_goal = min(nodes, key=lambda n: distance(n, goal))
        return nodes, nearest_to_goal

    return nodes, new_node


def generate_local_goal(current_pos, global_goal, obstacles, view_radius):
    dx = global_goal.x - current_pos[0]
    dy = global_goal.y - current_pos[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    if dist < view_radius:
        return Node(global_goal.x, global_goal.y)

    dx_norm = dx / dist if dist > 0 else 0
    dy_norm = dy / dist if dist > 0 else 0
    local_goal_x = current_pos[0] + dx_norm * view_radius * 0.8
    local_goal_y = current_pos[1] + dy_norm * view_radius * 0.8

    temp_node = Node(local_goal_x, local_goal_y)
    if is_obstacle(temp_node, obstacles):
        for angle in np.linspace(0, 2 * np.pi, 12):
            test_x = current_pos[0] + np.cos(angle) * view_radius * 0.8
            test_y = current_pos[1] + np.sin(angle) * view_radius * 0.8
            test_node = Node(test_x, test_y)
            if not is_obstacle(test_node, obstacles):
                return test_node

    return temp_node


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


class LocalPathTracker:
    def __init__(self,width,height,global_goal, obstacles, view_radius):
        self.width = width
        self.height = height
        self.global_goal = global_goal
        self.obstacles = obstacles
        self.view_radius = view_radius
        self.current_local_goal = None
        self.current_path = []
        self.reached_global_goal = False
        self.current_local_nodes = []

    def update(self, current_pos):
        if self.reached_global_goal:
            return self.global_goal

        if distance(Node(current_pos[0], current_pos[1]), self.global_goal) < 20:
            print("到达全局目标！")
            self.reached_global_goal = True
            return self.global_goal

        if (self.current_local_goal is None or distance(Node(current_pos[0], current_pos[1]), self.current_local_goal) < 15):
            view_obstacles = get_obstacles_in_view(self.obstacles, current_pos, self.view_radius)
            self.current_local_goal = generate_local_goal(current_pos, self.global_goal, view_obstacles,self.view_radius)
            print(f"生成新的局部目标: ({self.current_local_goal.x:.1f}, {self.current_local_goal.y:.1f})")

            start_node = Node(current_pos[0], current_pos[1])
            self.current_local_nodes, final_node = local_RRT(start_node,self.width,self.height,self.current_local_goal, view_obstacles,current_pos,self.view_radius)
            self.current_path = []
            if final_node:
                current = final_node
                while current:
                    self.current_path.append((int(current.x), int(current.y)))
                    current = current.parent
                self.current_path = self.current_path[::-1]
            print(f"局部路径规划完成")
        return self.current_local_goal

    def get_current_target(self):
        return self.current_path[0] if self.current_path else None

    def is_finished(self):
        return self.reached_global_goal

def create_local_view_layer(current_pos, view_radius, width, height):
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.circle(mask, (int(current_pos[0]), int(current_pos[1])),view_radius, (255, 255, 255, 255), -1)
    inverted_mask = cv2.bitwise_not(mask)
    inverted_mask[:, :, 3] = 150  # 设置透明度

    return inverted_mask


def path_tracking(shared_data, path_tracker, helmholtz_control):
    while shared_data['running'] and not path_tracker.is_finished():
        time.sleep(0.05)

        current_x = shared_data['center_x']
        current_y = shared_data['center_y']

        path_tracker.update((current_x, current_y))
        target_point = path_tracker.get_current_target()

        if target_point:
            target_x, target_y = target_point
            dx = target_x - current_x
            dy = target_y - current_y
            distance_to_target = np.sqrt(dx ** 2 + dy ** 2)

            if distance_to_target < 10 and len(path_tracker.current_path) > 1:
                path_tracker.current_path.pop(0)
                if path_tracker.current_path:
                    target_x, target_y = path_tracker.current_path[0]
                    dx = target_x - current_x
                    dy = target_y - current_y

            if np.sqrt(dx ** 2 + dy ** 2) > 10:
                angle_rad = np.arctan2(-dy, dx)
                beta = np.degrees(np.pi - angle_rad)
                helmholtz_control.update_beta(beta)

            print(f"当前位置: ({current_x:.0f}, {current_y:.0f}), "
                  f"目标位置: ({target_x:.0f}, {target_y:.0f}), "
                  f"距离: {distance_to_target:.1f}")

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
    VIEW_RADIUS=150

    goal = Node(width - 5, 5)

    print("请选择要追踪的物体")
    screenshot = pyautogui.screenshot(region=region)
    pil_array = np.array(screenshot)
    current_frame = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)

    bbox = cv2.selectROI("选择要追踪的物体", current_frame, False)
    cv2.destroyWindow("选择要追踪的物体")

    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(current_frame, bbox)

    x, y, w, h = [int(v) for v in bbox]
    start_x = x + w // 2
    start_y = y + h // 2
    start = Node(start_x, start_y)
    print(f"起点: ({start_x}, {start_y})")
    print(f"全局目标: ({goal.x}, {goal.y})")

    obstacles = create_obstacles(width, height)

    path_tracker = LocalPathTracker(width,height,goal, obstacles, VIEW_RADIUS)
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
            current_frame = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)

            ok, bbox = tracker.update(current_frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2
                center_y = y + h // 2

                shared_data['center_x'] = center_x
                shared_data['center_y'] = center_y

                layer = create_transparent_layer(width, height)
                view_obstacles = get_obstacles_in_view(obstacles, (center_x, center_y), VIEW_RADIUS)
                layer = draw_obstacles(layer, view_obstacles)

                if path_tracker.current_path:
                    for i in range(len(path_tracker.current_path) - 1):
                        cv2.line(layer,
                                 path_tracker.current_path[i],
                                 path_tracker.current_path[i + 1],
                                 (255, 0, 0, 200), 2)

                if path_tracker.current_local_goal:
                    cv2.circle(layer,(int(path_tracker.current_local_goal.x),int(path_tracker.current_local_goal.y)),8, (255, 255, 0, 255), -1)

                cv2.circle(layer, (int(goal.x), int(goal.y)), 10, (0, 165, 255, 255), -1)
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(current_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.circle(current_frame, (center_x, center_y), VIEW_RADIUS, (100, 100, 100), 2)
                target_point = path_tracker.get_current_target()
                if target_point:
                    target_x, target_y = target_point
                    cv2.circle(current_frame, (int(target_x), int(target_y)), 6, (0, 255, 255), -1)
                    cv2.arrowedLine(current_frame, (center_x, center_y),(int(target_x), int(target_y)), (255, 0, 255), 2)
                view_mask = create_local_view_layer((center_x, center_y), VIEW_RADIUS, width, height)

                result = virtual_add(current_frame, layer)
                result = virtual_add(result, view_mask)
            else:
                result = current_frame

            cv2.imshow('Local RRT Path Planning', result)

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