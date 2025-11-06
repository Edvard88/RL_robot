import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation

class AdvancedGo1Env(gym.Env):
    """
    Усовершенствованная среда для обучения робота Unitree Go1
    с комплексной системой вознаграждений и безопасными ограничениями
    """
    
    metadata = {"render_modes": ["human"], 'render_fps': 60}
    
    def __init__(self, xml_path, max_episode_steps=1000):
        super().__init__()
        
        # Загрузка модели MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Параметры времени
        self.dt = self.model.opt.timestep
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.current_episode_time = 0.0
        
        # Безопасные ограничения
        self.max_joint_velocity = 8.0  # рад/с
        self.max_joint_acceleration = 20.0  # рад/с²
        self.previous_joint_vel = np.zeros(12)
        
        # Параметры вознаграждения
        self._setup_reward_parameters()
        
        # Пространства действий и наблюдений
        self._setup_spaces()
        
        # Кэш для вычислений
        self.previous_action = np.zeros(12)
        self.previous_x_pos = 0.0
        self.episode_reward_info = {
            'velocity': 0, 'energy': 0, 'smoothness': 0, 
            'survival': 0, 'orientation': 0, 'contact': 0,
            'velocity_tracking': 0, 'joint_limits': 0, 'hip_position': 0,
            'gait_pattern': 0, 'lateral_movement': 0, 'time_penalty': 0,
            'rhythmic': 0, 'jerk_penalty': 0, 'slowness_penalty': 0
        }

    def _setup_reward_parameters(self):
        """Настройка параметров системы вознаграждений"""
        self.reward_weights = {
            'velocity': 2.0,           # Движение вперед
            'energy': -0.02,           # Энергопотребление  
            'smoothness': -0.1,        # Плавность движений
            'survival': 0.05,          # Выживание
            'orientation': -0.5,       # Стабильность ориентации
            'contact': 0.1,            # Качество контактов
            'velocity_tracking': 1.0,  # Следование целевой скорости
            'joint_limits': -2.0,      # Ограничения суставов
            'hip_position': -0.5,      # Положение бедер
            'gait_pattern': 0.3,       # Паттерн походки
            'lateral_movement': -1.0,  # Движение вбок
            'time_penalty': -0.01,     # Штраф за время
            'rhythmic': 0.5,           # Ритмичность движений
            'jerk_penalty': -0.1,      # Штраф за резкие движения
            'slowness_penalty': -0.1,  # Штраф за медлительность
        }
        
        # Целевые параметры
        self.target_velocity = 0.5  # м/с вперед
        self.target_height = 0.3    # м
        self.upright_threshold = 0.7  # w-компонент кватерниона
        self.ideal_hip_positions = np.array([0.0] * 4)  # Нулевое положение для всех бедер

    def _setup_spaces(self):
        """Настройка пространств действий и наблюдений"""
        # Пространство действий: нормализованное [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Пространство наблюдений
        obs_dict = {
            'joint_positions': spaces.Box(
                low=-np.pi, high=np.pi, shape=(12,), dtype=np.float32
            ),
            'joint_velocities': spaces.Box(
                low=-self.max_joint_velocity, 
                high=self.max_joint_velocity, 
                shape=(12,), dtype=np.float32
            ),
            'trunk_orientation': spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            ),
            'trunk_linear_velocity': spaces.Box(
                low=-5.0, high=5.0, shape=(3,), dtype=np.float32
            ),
            'trunk_angular_velocity': spaces.Box(
                low=-3.0, high=3.0, shape=(3,), dtype=np.float32
            ),
            'foot_contacts': spaces.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            ),
            'command': spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            'previous_actions': spaces.Box(
                low=-1.0, high=1.0, shape=(12,), dtype=np.float32
            )
        }
        
        self.observation_space = spaces.Dict(obs_dict)

    def _get_obs(self):
        """Получение наблюдения с проверкой безопасности"""
        mujoco.mj_forward(self.model, self.data)
        
        # Позиции и скорости суставов (только hinge joints)
        joint_pos = self.data.qpos[7:19].copy()
        joint_vel = self.data.qvel[6:18].copy()
        
        # Ориентация и скорость туловища
        trunk_quat = self.data.qpos[3:7].copy()
        trunk_lin_vel = self.data.qvel[0:3].copy()
        trunk_ang_vel = self.data.qvel[3:6].copy()
        
        # Контакты ног с землей
        foot_contacts = self._get_foot_contacts()
        
        observation = {
            'joint_positions': joint_pos.astype(np.float32),
            'joint_velocities': joint_vel.astype(np.float32),
            'trunk_orientation': trunk_quat.astype(np.float32),
            'trunk_linear_velocity': trunk_lin_vel.astype(np.float32),
            'trunk_angular_velocity': trunk_ang_vel.astype(np.float32),
            'foot_contacts': foot_contacts.astype(np.float32),
            'command': np.array([self.target_velocity, 0, 0], dtype=np.float32),
            'previous_actions': self.previous_action.astype(np.float32)
        }
        
        return observation

    def _get_foot_contacts(self):
        """Определение контактов ног с землей"""
        foot_contacts = np.zeros(4)
        foot_sites = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        for i, site_name in enumerate(foot_sites):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                # Определяем контакт по высоте и силе
                foot_height = self.data.site_xpos[site_id][2]
                foot_contacts[i] = 1.0 if foot_height < 0.03 else 0.0
                
        return foot_contacts

    def reset(self, seed=None, options=None):
        """Сброс среды в начальное состояние"""
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Установка в позу "home"
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
        # Случайное начальное состояние для разнообразия
        if options and options.get('random_start', False):
            self._randomize_start_position()
        
        self.current_step = 0
        self.current_episode_time = 0.0
        self.previous_action = np.zeros(12)
        self.previous_joint_vel = np.zeros(12)
        self.previous_x_pos = self.data.qpos[0].copy()
        self.episode_reward_info = {k: 0 for k in self.episode_reward_info}
        
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Выполнение одного шага симуляции"""
        # Применяем действие с проверкой безопасности
        scaled_action = self._scale_action(action)
        self.data.ctrl[:] = scaled_action
        
        # Сохраняем для следующего шага
        self.previous_action = action.copy()
        
        # Шаг симуляции
        mujoco.mj_step(self.model, self.data)
        
        # Обновляем время эпизода
        self.current_episode_time += self.dt
        
        # Получаем наблюдение и информацию
        observation = self._get_obs()
        reward, reward_components = self._compute_reward(observation, action)
        terminated = self._check_termination(observation)
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info(reward_components)
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info

    def _scale_action(self, action):
        """Масштабирование действия из [-1, 1] в реальные углы"""
        # Диапазоны суставов из модели
        joint_ranges = np.array([
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],  # FR
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],  # FL  
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888],  # RR
            [-0.863, 0.863], [-0.686, 4.501], [-2.818, -0.888]   # RL
        ])
        
        scaled = np.zeros_like(action)
        for i in range(12):
            mid = (joint_ranges[i, 0] + joint_ranges[i, 1]) / 2
            scale = (joint_ranges[i, 1] - joint_ranges[i, 0]) / 2
            scaled[i] = np.clip(action[i], -1.0, 1.0) * scale + mid
            
        return scaled

    def _compute_reward(self, obs, action):
        """Вычисление комплексного вознаграждения"""
        components = {}
        
        # 1. Движение вперед (изменение позиции по X)
        current_x = self.data.qpos[0]
        delta_x = current_x - self.previous_x_pos
        components['velocity'] = delta_x / self.dt  # скорость вперед
        self.previous_x_pos = current_x
        
        # 2. Следование целевой скорости
        forward_vel = obs['trunk_linear_velocity'][0]
        vel_error = abs(forward_vel - self.target_velocity)
        components['velocity_tracking'] = -vel_error
        
        # 3. Энергопотребление (нормированные действия)
        components['energy'] = -np.mean(np.abs(action))
        
        # 4. Плавность движений
        action_diff = np.linalg.norm(action - self.previous_action)
        components['smoothness'] = -action_diff
        
        # 5. Стабильность ориентации (штраф за наклон)
        trunk_pitch = self._get_trunk_pitch(obs['trunk_orientation'])
        trunk_roll = self._get_trunk_roll(obs['trunk_orientation'])
        orientation_penalty = -(abs(trunk_pitch) + abs(trunk_roll)) * 0.1
        components['orientation'] = orientation_penalty
        
        # 6. Качество контактов
        contact_pattern = self._analyze_contact_pattern(obs['foot_contacts'])
        components['contact'] = contact_pattern
        
        # 7. Ограничения суставов
        joint_vel = obs['joint_velocities']
        joint_accel = np.abs(joint_vel - self.previous_joint_vel) / self.dt
        velocity_violation = np.sum(joint_vel > self.max_joint_velocity)
        acceleration_violation = np.sum(joint_accel > self.max_joint_acceleration)
        components['joint_limits'] = -(velocity_violation + acceleration_violation)
        
        # 8. Награда за выживание (пропорционально времени)
        components['survival'] = 1.0 * self.dt
        
        # 9. Положение бедер (предпочтительно нулевое)
        hip_positions = obs['joint_positions'][0:4]  # первые 4 сустава - бедра
        hip_deviation = np.mean(np.abs(hip_positions - self.ideal_hip_positions))
        components['hip_position'] = -hip_deviation
        
        # 10. Паттерн походки (диагональные пары ног)
        gait_quality = self._evaluate_gait_pattern(obs['foot_contacts'])
        components['gait_pattern'] = gait_quality
        
        # 11. Штраф за движение вбок
        lateral_vel = abs(obs['trunk_linear_velocity'][1])  # скорость по Y
        components['lateral_movement'] = -lateral_vel
        
        # 12. Штраф за время
        components['time_penalty'] = -self.current_episode_time
        
        # 13. Периодические награды (для ритмичных движений)
        phase = self.current_episode_time % 1.0  # фаза 1 секунды
        rhythmic_bonus = 0.5 if 0.4 < phase < 0.6 else 0.0
        components['rhythmic'] = rhythmic_bonus
        
        # 14. Штраф за резкие движения (jerk penalty)
        action_change = np.linalg.norm(action - self.previous_action)
        jerk_penalty = -0.1 * action_change if action_change > 2.0 else 0.0
        components['jerk_penalty'] = jerk_penalty
        
        # 15. Штраф за медлительность
        slowness_penalty = -0.1 if (self.current_episode_time > 5.0 and 
                                   self._is_moving_slow(forward_vel)) else 0.0
        components['slowness_penalty'] = slowness_penalty
        
        # Сохраняем для следующего шага
        self.previous_joint_vel = joint_vel.copy()
        
        # Взвешенная сумма
        total_reward = sum(components[key] * self.reward_weights.get(key, 0) 
                          for key in components)
        
        # Обновляем статистику эпизода
        for key in components:
            if key in self.episode_reward_info:
                self.episode_reward_info[key] += components[key] * self.reward_weights.get(key, 0)
        
        return total_reward, components

    def _analyze_contact_pattern(self, foot_contacts):
        """Анализ паттерна контактов для оценки качества походки"""
        num_contacts = np.sum(foot_contacts)
        
        if num_contacts == 3:
            return 1.0  # Идеально для статической стабильности
        elif num_contacts == 2:
            return 0.6  # Тротинг или pacing
        elif num_contacts == 4:
            return 0.3  # Все ноги на земле - неэффективно
        else:
            return -1.0  # Падение или прыжок

    def _evaluate_gait_pattern(self, foot_contacts):
        """Оценка качества паттерна походки (диагональные пары)"""
        # Индексы: 0=FL, 1=FR, 2=RL, 3=RR
        diagonal_pair1 = (foot_contacts[0] > 0.5 and foot_contacts[3] > 0.5 and 
                         foot_contacts[1] < 0.5 and foot_contacts[2] < 0.5)
        diagonal_pair2 = (foot_contacts[1] > 0.5 and foot_contacts[2] > 0.5 and 
                         foot_contacts[0] < 0.5 and foot_contacts[3] < 0.5)
        
        if diagonal_pair1 or diagonal_pair2:
            return 1.0  # Идеальный диагональный паттерн
        else:
            return 0.0

    def _is_moving_slow(self, forward_vel):
        """Проверка, движется ли робот слишком медленно"""
        return forward_vel < 0.1  # м/с

    def _get_trunk_pitch(self, quat):
        """Получить угол pitch из кватерниона"""
        w, x, y, z = quat
        pitch = np.arcsin(2.0 * (w * y - z * x))
        return pitch

    def _get_trunk_roll(self, quat):
        """Получить угол roll из кватерниона"""
        w, x, y, z = quat
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return roll

    def _check_termination(self, obs):
        """Проверка условий окончания эпизода"""
        # Падение по ориентации
        trunk_w = obs['trunk_orientation'][0]
        if trunk_w < self.upright_threshold:
            return True
        
        # Падение по высоте
        trunk_height = self.data.qpos[2]
        if trunk_height < 0.1:
            return True
        
        # Превышение скорости суставов
        if np.any(np.abs(obs['joint_velocities']) > self.max_joint_velocity * 1.5):
            return True
            
        return False

    def _get_info(self, reward_components=None):
        """Получение дополнительной информации"""
        info = {
            'trunk_height': float(self.data.qpos[2]),
            'forward_velocity': float(self.data.qvel[0]),
            'orientation_w': float(self.data.qpos[3]),
            'current_step': self.current_step,
            'episode_time': self.current_episode_time,
            'episode_reward_components': self.episode_reward_info.copy()
        }
        
        if reward_components:
            info['step_reward_components'] = reward_components
            
        return info

    def _randomize_start_position(self):
        """Случайное начальное положение для разнообразия"""
        # Небольшие случайные отклонения от позы "home"
        noise = self.np_random.uniform(-0.1, 0.1, size=12)
        self.data.qpos[7:19] += noise
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        """Рендеринг среды"""
        # Реализация зависит от вашей системы визуализации
        pass

    def close(self):
        """Корректное закрытие среды"""
        pass


# Пример использования
if __name__ == "__main__":
    env = AdvancedGo1Env('robot_models/unitree_go1/scene.xml')
    
    # Тестирование
    obs, info = env.reset()
    print("Начальное наблюдение:", {k: v.shape for k, v in obs.items()})
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Эпизод завершен на шаге {i}")
            print("Компоненты награды за эпизод:", info['episode_reward_components'])
            obs, info = env.reset()
    
    env.close()