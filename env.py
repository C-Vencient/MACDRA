import os
import gym
import numpy as np
import random
import heapq
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from gym import spaces


class Task:
    def __init__(self, task_id: int, task_type: str, task_size: float, cpu_cycles: float,
                 generation_time: float, source_device: int):

        self.id = task_id
        self.type = task_type
        self.size = task_size
        self.total_cpu_cycles = cpu_cycles
        self.remaining_cpu_cycles = cpu_cycles
        self.generation_time = generation_time
        self.source_device = source_device
        self.current_device = None
        self.start_time = None
        self.completion_time = None
        self.queue_entry_time = None
        self.transmission_start_time = None
        self.assigned_resource_ratio = None
        self.discarded = False
        self.arrive_time = 0
        self.transmission_time = 0

    def __lt__(self, other):
        # 定义 Task 实例的比较规则
        # 按照 completion_time 排序
        return self.generation_time < other.generation_time

    @property
    def delay(self) -> float:
        """计算任务总时延（如果已完成）"""
        if self.completion_time is None:
            return 0
        return self.completion_time - self.generation_time

class Device:
    def __init__(self, device_id: int, device_type: str, compute_power: float, max_delay: float):

        self.id = device_id
        self.type = device_type
        self.compute_power = compute_power
        self.max_delay = max_delay

        self.processing_tasks = []

        self.waiting_queue = deque()

        self.agent_id = f"{device_type}_{device_id}"
        self.current_time = 0.0
        self.cpu_utilization = 0.0

        self.attack_history = deque(maxlen=20)
        self.confidence = 1.0
        self.continuous_safe_slots = 0
        self.confidence = 1.0

    def add_task(self, task: Task, current_time: float):
        """添加任务到设备"""

        task.current_device = self.id
        task.queue_entry_time = current_time

        if self._try_assign_resource(task, current_time):
            return
        else:
            self.waiting_queue.append((current_time, task))

    def _try_assign_resource(self, task: Task, current_time: float) -> bool:
        """尝试为任务分配计算资源"""
        resource_ratio = random.uniform(0.1, 0.7)

        if self.cpu_utilization + resource_ratio <= 1.0:
            # 分配资源
            task.start_time = current_time
            task.assigned_resource_ratio = resource_ratio
            self.processing_tasks.append((task, resource_ratio, current_time))
            self.cpu_utilization += resource_ratio
            return True
        return False

    def process_tasks(self, current_time: float, time_slot: float) -> List[Task]:
        completed_tasks = []
        elapsed_time = 0.0


        while elapsed_time < time_slot:
            min_remaining_time = float('inf')
            next_completed_task = None
            next_completed_idx = -1


            for idx, (task, resource_ratio, start_time) in enumerate(self.processing_tasks):

                processing_rate = self.compute_power * 1e9 * resource_ratio  # 周期/秒
                remaining_time = task.remaining_cpu_cycles / processing_rate if processing_rate > 0 else float('inf')

                # 更新最早完成的任务
                if remaining_time < min_remaining_time:
                    min_remaining_time = remaining_time
                    next_completed_task = (task, resource_ratio, idx)

            available_time = min(time_slot - elapsed_time, min_remaining_time)

            if available_time <= 0:
                break

            for idx, (task, resource_ratio, start_time) in enumerate(self.processing_tasks):

                processing_rate = self.compute_power * 1e9 * resource_ratio

                processed_cycles = processing_rate * available_time

                task.remaining_cpu_cycles -= processed_cycles

                if task.remaining_cpu_cycles <= 0:

                    task.completion_time = (current_time + task.transmission_time) + elapsed_time + available_time
                    completed_tasks.append(task)

            new_processing_tasks = []
            for idx, (task, resource_ratio, start_time) in enumerate(self.processing_tasks):

                if task.remaining_cpu_cycles > 0:
                    new_processing_tasks.append((task, resource_ratio, start_time))
                else:
                    self.cpu_utilization -= resource_ratio
            self.processing_tasks = new_processing_tasks

            elapsed_time += available_time

            while self.waiting_queue:
                arrival_time, task = self.waiting_queue[0]


                actual_arrival_time = current_time + elapsed_time
                if arrival_time > actual_arrival_time:

                    break

                self.waiting_queue.popleft()

                if not self._try_assign_resource(task, actual_arrival_time):
                    self.waiting_queue.appendleft((arrival_time, task))
                    break

        if elapsed_time < time_slot:
            remaining_time = time_slot - elapsed_time

            # 更新所有任务进度
            for idx, (task, resource_ratio, start_time) in enumerate(self.processing_tasks):

                processing_rate = self.compute_power * 1e9 * resource_ratio
                processed_cycles = processing_rate * remaining_time

                task.remaining_cpu_cycles -= processed_cycles

                if task.remaining_cpu_cycles <= 0:
                    # 任务完成
                    task.completion_time = current_time + time_slot
                    completed_tasks.append(task)

            new_processing_tasks = []
            for idx, (task, resource_ratio, start_time) in enumerate(self.processing_tasks):

                if task.remaining_cpu_cycles > 0:
                    new_processing_tasks.append((task, resource_ratio, start_time))
                else:
                    # 释放资源
                    self.cpu_utilization -= resource_ratio
            self.processing_tasks = new_processing_tasks

        return completed_tasks


    @property
    def queue_length(self) -> int:
        """获取等待队列长度"""
        return len(self.waiting_queue)


class TerminalDevice(Device):
    def __init__(self, device_id: int, max_delay: float):
        super().__init__(device_id, 'terminal', 0, max_delay)  # 终端设备没有计算能力
        self.scheduling_queue = deque()  # 终端设备特有的调度队列

    def generate_tasks(self, num_tasks: int, current_time: float,
                       task_size_range: Tuple[float, float],
                       cpu_cycles_range: Tuple[float, float]) -> List[Task]:
        """生成新任务"""
        tasks = []
        for _ in range(num_tasks):
            task_size = random.uniform(*task_size_range)
            cpu_cycles = random.uniform(*cpu_cycles_range)
            task = Task(
                task_id=random.randint(1, 10 ** 9),
                task_type='binary',
                task_size=task_size,
                cpu_cycles=cpu_cycles,
                generation_time=current_time,
                source_device=self.id
            )
            self.scheduling_queue.append(task)
            tasks.append(task)
        return tasks



class EdgeDevice(Device):
    def __init__(self, device_id: int, compute_power: float, max_delay: float):
        super().__init__(device_id, 'edge', compute_power, max_delay)


class CloudDevice(Device):
    def __init__(self, device_id: int, compute_power: float, max_delay: float):
        super().__init__(device_id, 'cloud', compute_power, max_delay)


class Network:
    def __init__(self, bandwidth_matrix: Dict[Tuple[int, int], float], max_delay: float):
        """
        网络类，管理任务传输
        :param bandwidth_matrix: 设备间带宽矩阵 (Mbps)
        :param max_delay: 最大可容忍时延
        """
        self.bandwidth_matrix = bandwidth_matrix
        self.transmitting_tasks = []  # (完成时间, 任务, 目标设备)
        self.current_time = 0.0
        self.max_delay = max_delay

    def start_transmission(self, task: Task, source_device: int, target_device: int, start_time: float):
        """开始传输任务"""

        # 计算传输速率 (Mbps -> bps)
        bandwidth = self.bandwidth_matrix.get((source_device, target_device), 10)  # 默认10 Mbps
        transmission_rate = bandwidth * 1e6  # bps

        # 计算传输时间
        transmission_time = task.size / transmission_rate
        task.transmission_time = transmission_time

        completion_time = start_time + transmission_time
        task.transmission_start_time = start_time

        heapq.heappush(self.transmitting_tasks, (completion_time, task, target_device))

class SecurityDetectionEnv():
    def __init__(self,
                 num_terminals: int = 10,
                 num_edges: int = 3,
                 num_clouds: int = 1,
                 time_slot_duration: float = 1.0,  # 每个时隙持续时间(秒)
                 task_per_device_range: Tuple[int, int] = (5, 10),
                 task_size_range: Tuple[float, float] = (8192, 81920),  # bits (1-10 KB)
                 binary_cpu_range: Tuple[float, float] = (1e8, 5e8),  # CPU cycles
                 multiclass_cpu_range: Tuple[float, float] = (5e8, 20e8),  # CPU cycles
                 malicious_prob: float =-1,
                 max_delay: float = 1.0,  # 最大可容忍时延(秒)
                 bandwidth_matrix: Optional[Dict[Tuple[int, int], float]] = None):
        super().__init__()

        """
        安全检测环境

        :param num_terminals: 终端设备数量
        :param num_edges: 边缘设备数量
        :param num_clouds: 云端设备数量
        :param time_slot_duration: 每个时隙持续时间(秒)
        :param task_per_device_range: 每台设备生成任务数量范围
        :param task_size_range: 任务大小范围(bits)
        :param binary_cpu_range: 二分类任务CPU周期范围
        :param multiclass_cpu_range: 多分类任务CPU周期范围
        :param malicious_prob: 恶意流量概率
        :param max_delay: 最大可容忍时延(秒)，用于归一化和超时检查
        :param bandwidth_matrix: 设备间带宽矩阵(Mbps)
        """
        self.prev_accuracy = 0
        self.max_delay = max_delay
        self.num_terminals = num_terminals
        self.num_edges = num_edges
        self.num_clouds = num_clouds
        # 添加以下属性
        self.is_discrete = False  # 明确标识为连续动作空间
        self.is_multi_agent = True  # 明确标识为多智能体环境
        self.sequence_length = 5  # 使用最近5个时间步的数据
        # 新增任务调度计数器
        self.scheduled_tasks_count = defaultdict(int)  # 记录每个设备被调度的任务数
        self.total_generated_tasks = 0  # 记录当前时隙生成的总任务量

        self.discarded_tasks = []


        self.terminals = [TerminalDevice(i, max_delay) for i in range(num_terminals)]

        # 边缘设备: num_terminals 到 num_terminals+num_edges-1
        self.edges = [EdgeDevice(num_terminals + i,
                                 compute_power=random.uniform(10, 20.0),
                                 max_delay=max_delay)
                      for i in range(num_edges)]

        # 云端设备: num_terminals+num_edges 到 num_terminals+num_edges+num_clouds-1
        self.clouds = [CloudDevice(num_terminals + num_edges + i,
                                   compute_power=random.uniform(30.0, 40.0),
                                   max_delay=max_delay)
                       for i in range(num_clouds)]

        # 所有设备列表
        self.all_devices = self.terminals + self.edges + self.clouds
        self.device_map = {d.id: d for d in self.all_devices}

        # 网络初始化
        if bandwidth_matrix is None:
            bandwidth_matrix = self._default_bandwidth_matrix()
        self.network = Network(bandwidth_matrix, max_delay)
        self.bandwidth_matrix = bandwidth_matrix  # 保存带宽矩阵

        # 环境参数
        self.time_slot_duration = time_slot_duration
        self.task_per_device_range = task_per_device_range
        self.task_size_range = task_size_range
        self.binary_cpu_range = binary_cpu_range
        self.multiclass_cpu_range = multiclass_cpu_range
        self.malicious_prob = malicious_prob

        # 状态跟踪
        self.current_time = 0.0
        self.time_slot = 0
        self.completed_tasks = []
        self.pending_tasks = []
        self.global_acc = 0.5  # 初始F1分数

        # 任务统计
        self.edge_binary_completed = 0
        self.edge_multiclass_completed = 0
        self.cloud_binary_completed = 0
        self.cloud_multiclass_completed = 0
        self.edge_correct = 0
        self.cloud_correct = 0

        # 智能体ID映射
        self.agent_ids = [d.agent_id for d in self.all_devices]
        self.agent_id_to_device = {d.agent_id: d for d in self.all_devices}

        # 动作和观测空间
        self.action_spaces = {}
        self.observation_spaces = {}

        # 初始化动作和观测空间
        self._init_spaces()
        # rnn适配
        self.obs_history = {agent_id: deque(maxlen=self.sequence_length)
                            for agent_id in self.agent_ids}
        self.is_sequential = True

        self.agents = self.agent_ids
        self.num_agents = len(self.agents)
        self.agent_groups = [self.agents]
        self.max_episode_steps = 20
        state_dim = len(self._build_global_state())
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        self.observation_space = {}
        self.action_space = {}
        for agent_id in self.agents:
            obs_shape = self.observation_spaces[agent_id]
            act_shape = self.action_spaces[agent_id]
            feat_dim = self.observation_spaces[agent_id][0]

            self.observation_space[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )

            self.action_space[agent_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=act_shape,
                dtype=np.float32
            )

        # 攻击状态
        self.attack_targets = []  # 当前被攻击的设备

        # 置信度历史记录
        self.confidence_history = {d.id: [] for d in self.edges + self.clouds}
        self._state = self._build_global_state()

    def _build_global_state(self) -> np.ndarray:
        """构建完整的全局状态向量"""
        observations = self._get_observations()
        all_obs = []
        for agent_id in self.agent_ids:
            all_obs.append(observations[agent_id])

        global_state = np.concatenate(all_obs, axis=0)
        return global_state

    def _default_bandwidth_matrix(self) -> Dict[Tuple[int, int], float]:
        """生成默认带宽矩阵"""
        bandwidth = {}

        # 获取所有设备ID
        terminal_ids = [t.id for t in self.terminals]
        edge_ids = [e.id for e in self.edges]
        cloud_ids = [c.id for c in self.clouds]
        all_ids = terminal_ids + edge_ids + cloud_ids

        for terminal in self.terminals:
            for edge in self.edges:
                bandwidth[(terminal.id, edge.id)] = random.uniform(30, 60)
            for cloud in self.clouds:
                bandwidth[(terminal.id, cloud.id)] = random.uniform(30, 60)

        for i, edge1 in enumerate(self.edges):
            for j, edge2 in enumerate(self.edges):
                if i != j:
                    bandwidth[(edge1.id, edge2.id)] = random.uniform(30, 60)
                else:
                    bandwidth[(edge1.id, edge1.id)] = 1e9

        for edge in self.edges:
            for cloud in self.clouds:
                bandwidth[(edge.id, cloud.id)] = random.uniform(30, 60)
                bandwidth[(cloud.id, edge.id)] = random.uniform(30, 60)

        # 云端之间: 100-200 Mbps
        for i, cloud1 in enumerate(self.clouds):
            for j, cloud2 in enumerate(self.clouds):
                if i != j:
                    bandwidth[(cloud1.id, cloud2.id)] = random.uniform(30, 60)
                else:
                    # 设备到自身的带宽设为极大值
                    bandwidth[(cloud1.id, cloud1.id)] = 1e9

        # 确保所有可能的设备对都有带宽定义
        for src in all_ids:
            for dst in all_ids:
                if (src, dst) not in bandwidth:
                    # 设置默认带宽
                    if src == dst:
                        bandwidth[(src, dst)] = 1e9  # 内部传输
                    else:
                        bandwidth[(src, dst)] = 10  # 默认带宽

        return bandwidth

    def _init_spaces(self):
        """初始化动作和观测空间"""
        num_edge_cloud = len(self.edges) + len(self.clouds)

        obs_length = 1 + 3 * num_edge_cloud

        for device in self.all_devices:

            self.action_spaces[device.agent_id] = (num_edge_cloud,)

            self.observation_spaces[device.agent_id] = (obs_length,)

    def _get_global_states(self) -> Tuple[List[float], List[float], List[float]]:
        """获取所有边缘和云端设备的全局状态"""
        cpu_utilizations = []
        waiting_queue_lengths = []
        processing_queue_lengths = []
        confidences = []
        cpu_capacities=[]

        # 收集边缘设备状态
        for edge in self.edges:
            cpu_utilizations.append(edge.cpu_utilization)
            waiting_queue_lengths.append(len(edge.waiting_queue))
            processing_queue_lengths.append(len(edge.processing_tasks))
            confidences.append(edge.confidence)
            cpu_capacities.append(edge.compute_power)

        # 收集云端设备状态
        for cloud in self.clouds:
            cpu_utilizations.append(cloud.cpu_utilization)
            waiting_queue_lengths.append(len(cloud.waiting_queue))
            confidences.append(cloud.confidence)
            cpu_capacities.append(cloud.compute_power)

        return cpu_capacities, confidences, waiting_queue_lengths,cpu_utilizations


    def get_confidences(self) -> Dict[int, float]:
        """获取所有边缘和云端设备的置信度"""
        confidences = {}
        for device in self.edges + self.clouds:
            confidences[device.id] = device.confidence
        return confidences

    def reset(self):
        """重置环境"""
        self.current_time = 0.0
        self.time_slot = 0
        self.completed_tasks = []
        self.pending_tasks = []
        self.global_acc = 0.5
        self.attack_targets = []

        # 重置任务统计
        self.edge_binary_completed = 0
        self.edge_multiclass_completed = 0
        self.cloud_binary_completed = 0
        self.cloud_multiclass_completed = 0
        self.edge_correct = 0
        self.cloud_correct = 0
        # 重置任务计数器
        self.scheduled_tasks_count.clear()
        self.total_generated_tasks = 0
        # 重置所有设备
        for device in self.all_devices:
            device.processing_tasks = []
            device.waiting_queue = deque()
            device.cpu_utilization = 0.0
            device.attack_history.clear()
            device.confidence = 1.0
            if device.type == 'terminal':
                device.scheduling_queue = deque()

        self.network = Network(self.network.bandwidth_matrix, self.max_delay)

        self.confidence_history = {d.id: [] for d in self.edges + self.clouds}

        self._generate_tasks()
        observations = self._get_observations()

        self._state = self._build_global_state()
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos



    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有智能体的观测（统一结构）"""

        cpu_cap, confidences, wait_lens, cpu_utils= self._get_global_states()

        observations = {}

        for device in self.all_devices:
            # 确定自身队列长度
            if device.type == 'terminal':
                queue_length = len(device.scheduling_queue)
            else:
                queue_length = len(device.waiting_queue)


            # 获取带宽向量
            bandwidth_vector = self._get_bandwidth_vector(device.id)

            obs_vector = [queue_length]
            obs_vector.extend(cpu_utils)
            obs_vector.extend(cpu_cap)
            obs_vector.extend(bandwidth_vector)
            observations[device.agent_id] = np.array(obs_vector, dtype=np.float32)

        return observations

    def _check_global_timeout(self, current_time: float):
        """全局检查所有未完成任务是否超时（包括传输队列、等待队列和处理队列）"""
        # 1. 检查网络传输队列
        new_transmitting = []
        while self.network.transmitting_tasks:
            completion_time, task, target_device = heapq.heappop(self.network.transmitting_tasks)
            heapq.heappush(new_transmitting, (completion_time, task, target_device))
        self.network.transmitting_tasks = new_transmitting

        # 2. 检查所有设备的等待队列
        for device in self.all_devices:
            new_waiting_queue = deque()
            while device.waiting_queue:
                arrival_time, task = device.waiting_queue.popleft()

                new_waiting_queue.append((arrival_time, task))
            device.waiting_queue = new_waiting_queue

        # 3. 检查所有设备的处理队列
        for device in self.all_devices:
            new_processing_tasks = []
            for task_info in device.processing_tasks:
                task = task_info[0]

                new_processing_tasks.append(task_info)
            device.processing_tasks = new_processing_tasks

        # 4. 检查终端设备的调度队列
        for terminal in self.terminals:
            new_scheduling_queue = deque()
            while terminal.scheduling_queue:
                task = terminal.scheduling_queue.popleft()
                new_scheduling_queue.append(task)
            terminal.scheduling_queue = new_scheduling_queue



    def step(self, actions: Dict[str, np.ndarray]):

        prev_accuracy = self.global_acc
        self.edge_binary_completed = 0
        self.edge_multiclass_completed = 0
        self.cloud_binary_completed = 0
        self.cloud_multiclass_completed = 0
        self.edge_correct = 0
        self.cloud_correct = 0
        self.discarded_tasks = []
        self._process_actions(actions)

        slot_end_time = self.current_time + self.time_slot_duration

        time_step = 0.1
        while self.current_time < slot_end_time:

            step_size = min(time_step, slot_end_time - self.current_time)
            step_end_time = self.current_time + step_size  # 当前步结束时间
            arrived_tasks = self.network.update_transmissions(step_end_time)
            for device_id, tasks_info in arrived_tasks.items():
                device = self.device_map[device_id]
                for task_info in tasks_info:

                    device.add_task(task_info[0], task_info[1])

            completed_tasks = []
            discarded_tasks = []
            for device in self.all_devices:
                if device.type != 'terminal':  # 只有边缘和云端设备执行任务
                    completed = device.process_tasks(self.current_time, step_size)

                    completed_tasks.extend(completed)

            self.discarded_tasks.extend(discarded_tasks)

            self._handle_completed_tasks(completed_tasks)
            self.current_time = step_end_time
        self._update_f1_from_detection()
        self.time_slot += 1
        total_task_num = self._generate_tasks()
        observations = self._get_observations()
        rewards = self._calculate_rewards(prev_accuracy)
        dones = {agent_id: False for agent_id in self.agent_ids}  # 持续任务，不会结束

        avg_delay = self._calculate_avg_delay()
        infos = {
            'accuracy': self.global_acc,
            'avg_delay': avg_delay,
            'completed_tasks': len(self.completed_tasks),
            'pending_tasks': len(self.pending_tasks),
            'edge_binary': self.edge_binary_completed,
            'edge_multiclass': self.edge_multiclass_completed,
            'cloud_binary': self.cloud_binary_completed,
            'cloud_multiclass': self.cloud_multiclass_completed,
            'attack_targets': self.attack_targets,
            'confidences': self.get_confidences(),
            'discarded_tasks':len(self.discarded_tasks)
        }
        self._state = self._build_global_state()
        self.completed_tasks = []
        self.scheduled_tasks_count.clear()
        self.discarded_tasks.clear()
        self.total_generated_tasks = 0

        infos.update({
            'bi': infos.get('edge_binary', 0) + infos.get('cloud_binary', 0),
            'mu': infos.get('edge_multiclass', 0) + infos.get('cloud_multiclass', 0)
        })

        truncated = self.time_slot >= self.max_episode_steps
        global_reward = list(rewards.values())[0]
        done = truncated

        return global_reward, done, infos

    #=================================================================================================
    def get_obs(self):
        """返回所有智能体的观测数组 (N, obs_dim)"""
        obs_dict = self._get_observations()
        return np.array([obs_dict[agent_id] for agent_id in self.agent_ids])
    def get_state(self):
        """返回全局状态向量 (state_dim,)"""
        return self._state
    def get_avail_actions(self):
        """返回可用动作矩阵 (全1矩阵表示所有动作可用)"""
        action_dim = self.action_spaces[self.agents[0]][0]
        return np.ones((self.num_agents, action_dim))
    def get_env_info(self):
        """返回环境信息字典"""
        return {
            "n_agents": self.num_agents,
            "obs_shape": self.observation_spaces[self.agents[0]][0],
            "state_shape": self.state_space.shape[0],
            "action_dim": self.action_spaces[self.agents[0]][0],  # 连续动作维度
            "episode_limit": self.max_episode_steps
        }

    def state(self) -> np.ndarray:
        return self._state

    @property
    def seq_len(self):
        return self.sequence_length

    # 新增空方法
    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def agent_mask(self):
        """返回所有活跃代理的掩码"""
        return {agent_id: True for agent_id in self.agents}

    def get_groups_info(self):
        return {
            'num_groups': len(self.agent_groups),
            'agent_groups': self.agent_groups,
            'observation_space_groups': [
                {agent_id: self.observation_space[agent_id] for agent_id in group}
                for group in self.agent_groups
            ],
            'action_space_groups': [
                {agent_id: self.action_space[agent_id] for agent_id in group}
                for group in self.agent_groups
            ],
            'num_agents_groups': [len(group) for group in self.agent_groups]
        }


    def _update_f1_from_detection(self):

        total_tasks = (self.edge_binary_completed + self.edge_multiclass_completed +
                       self.cloud_binary_completed + self.cloud_multiclass_completed)

        if total_tasks > 0:
            correct_count = 0

            self.edge_correct = (self.edge_binary_completed + self.edge_multiclass_completed) * 0.856

            self.cloud_correct = (self.cloud_binary_completed + self.cloud_multiclass_completed) * 0.972
            correct_count = self.edge_correct + self.cloud_correct
            accuracy = correct_count / total_tasks
            self.global_acc = accuracy
        else:
            self.global_acc = 0.5
            print(f"acc:{self.global_acc}")


    def _generate_tasks(self):
        """在终端设备生成新任务，并记录总任务量"""
        total_tasks_num = 0
        for terminal in self.terminals:
            num_tasks = random.randint(*self.task_per_device_range)
            total_tasks_num += num_tasks
            new_tasks = terminal.generate_tasks(
                num_tasks,
                self.current_time,
                self.task_size_range,
                self.binary_cpu_range
            )
            self.pending_tasks.extend(new_tasks)

        # 记录当前时隙生成的总任务量
        self.total_generated_tasks = total_tasks_num
        return total_tasks_num

    def _process_actions(self, actions: Dict[str, np.ndarray]):

        for agent_id, action in actions.items():
            pos_action = self._normalize_prob(action, temperature=1.0, alpha=0.05)
            device = self.agent_id_to_device[agent_id]

            if device.type == 'terminal':
                self._process_terminal_action(device, pos_action)
            else:
                self._process_edge_cloud_action(device, pos_action)

    def _process_terminal_action(self, device: TerminalDevice, action: np.ndarray):

        action = np.maximum(action, 0)
        action_sum = action.sum()
        target_devices = self.edges + self.clouds
        if action_sum > 0:
            action /= action_sum
        else:
            # 若全为0或负，回退为均匀分配，避免后续索引越界
            action = np.ones(len(target_devices), dtype=np.float32) / max(1, len(target_devices))
        # 计算每个目标设备分配的任务数量
        total_tasks = len(device.scheduling_queue)
        if total_tasks == 0:
            return

        counts = np.zeros(len(target_devices), dtype=int)
        remaining = total_tasks

        for i in range(len(target_devices)):
            count = int(action[i] * total_tasks)
            counts[i] = count
            remaining -= count

        if remaining > 0:
            fractional = [(action[i] * total_tasks) - counts[i] for i in range(len(target_devices))]
            indices = np.argsort(fractional)[::-1]
            for i in range(remaining):
                counts[indices[i % len(indices)]] += 1

        for i, count in enumerate(counts):
            for _ in range(count):
                if device.scheduling_queue:
                    task = device.scheduling_queue.popleft()

                    target_device = target_devices[i]
                    self.scheduled_tasks_count[target_device.id] += 1

                    self.network.start_transmission(
                        task,
                        source_device=device.id,
                        target_device=target_device.id,
                        start_time=self.current_time
                    )

    def _process_edge_cloud_action(self, device: Device, action: np.ndarray):

        target_devices = self.edges + self.clouds


        action = np.maximum(action, 0)
        action_sum = action.sum()
        if action_sum > 0:
            action /= action_sum
        else:
            action = np.ones(len(target_devices), dtype=np.float32) / max(1, len(target_devices))
        total_tasks = len(device.waiting_queue)
        if total_tasks == 0:
            return

        counts = np.zeros(len(target_devices), dtype=int)
        remaining = total_tasks

        for i in range(len(target_devices)):
            count = int(action[i] * total_tasks)
            counts[i] = count
            remaining -= count

        if remaining > 0:
            fractional = [(action[i] * total_tasks) - counts[i] for i in range(len(target_devices))]
            indices = np.argsort(fractional)[::-1]

            for i in range(remaining):
                counts[indices[i % len(indices)]] += 1

        for i, count in enumerate(counts):
            for _ in range(count):
                if device.waiting_queue:
                    arrival_time, task = device.waiting_queue.popleft()

                    target_device = target_devices[i]
                    self.scheduled_tasks_count[target_device.id] += 1

                    if target_device.id == device.id:

                        if device._try_assign_resource(task, self.current_time):
                            pass
                        else:

                            device.waiting_queue.appendleft((arrival_time, task))
                    else:

                        self.network.start_transmission(
                            task,
                            source_device=device.id,
                            target_device=target_device.id,
                            start_time=self.current_time
                        )

    def _handle_completed_tasks(self, completed_tasks: List[Task]):
        for task in completed_tasks:

            if task in self.pending_tasks:
                self.pending_tasks.remove(task)

            self.completed_tasks.append(task)
            if task.current_device in [d.id for d in self.edges]:
                if task.type == 'binary':
                    self.edge_binary_completed += 1
                else:
                    self.edge_multiclass_completed += 1
            elif task.current_device in [d.id for d in self.clouds]:
                if task.type == 'binary':
                    self.cloud_binary_completed += 1
                else:
                    self.cloud_multiclass_completed += 1

            if task.type == 'binary' and random.random() < self.malicious_prob:
                multiclass_task = Task(
                    task_id=random.randint(1, 10 ** 9),
                    task_type='multiclass',
                    task_size=task.size,
                    cpu_cycles=random.uniform(*self.multiclass_cpu_range),
                    generation_time=self.current_time,
                    source_device=task.source_device
                )
                multiclass_task.current_device = task.current_device
                self.pending_tasks.append(multiclass_task)

                device = self.device_map[task.current_device]
                device.add_task(multiclass_task, self.current_time)

    def _calculate_avg_delay(self) -> float:
        if not self.completed_tasks:
            return 0
        valid_completed = [t for t in self.completed_tasks]

        return sum(t.delay for t in valid_completed) / len(valid_completed)

    def _calculate_rewards(self,prev_accuracy:float) -> Dict[str, float]:
        """计算所有智能体的奖励"""
        avg_delay = self._calculate_avg_delay()
        
        total_num = self.edge_binary_completed + self.edge_multiclass_completed + \
                    self.cloud_binary_completed + self.cloud_multiclass_completed
        reward_value = 0.5 * (0 - avg_delay) + 0.5 * self.global_acc

        return {agent_id: reward_value for agent_id in self.agent_ids}

    def get_task_statistics(self) -> Tuple[int, int, int, int]:

        return (self.edge_binary_completed,
                self.edge_multiclass_completed,
                self.cloud_binary_completed,
                self.cloud_multiclass_completed)

    def _normalize_prob(self, x: np.ndarray, temperature: float = 1.0, eps: float = 1e-8, alpha: float = 0.05) -> np.ndarray:
        if x.ndim != 1:
            x = np.asarray(x).reshape(-1)
        t = max(temperature, eps)
        z = x - np.max(x)
        exp = np.exp(z / t)
        probs = exp / (np.sum(exp) + eps)
        k = probs.shape[0]
        probs = (1 - alpha) * probs + alpha / max(1, k)
        probs = probs / (np.sum(probs) + eps)
        return probs.astype(np.float32)
