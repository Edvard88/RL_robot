import os
import numpy as np

import mujoco

import gymnasium as gym
from gymnasium.utils.env_checker import check_env


class RobotGo1Env(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, xml_path):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.reward = None
        # Параметры времени
        self.dt = self.model.opt.timestep
        

        # Параметры времени
        self.dt = self.model.opt.timestep  # шаг времени в секундах
                
        # Установим максимальное время с начсала эпизода в секундах
        # Те эпизод не может длится более 100сек, собака может идти 100сек, далее trancked
        self.max_episode_time = 10      # максимальное время эпизода в секундах
        self.max_episode_steps = int(self.max_episode_time / self.dt)  # в шагах
        
        # Трекинг времени
        self.current_episode_time = 0
        self.current_episode_steps = 0


        self.qpos = None
        self.qvel = None


        free_joint = np.array([[-np.inf, np.inf], # Для X
                                [-np.inf, np.inf], # Для Y
                                [0, 10], # Для Z (условно поставили, что робот не может уйти под пол и взлетететь на 10м вверх)
                                [-1, 1], # w - действительная часть кватерниона Определяет величину поворота w = cos(θ/2),  где θ - угол поворота
                                [-1, 1], # x - ось вращения по x
                                [-1, 1], # y - ось вращения по y
                                [-1, 1], # z - ось вращения по z
                ])

        
        
        # В массиве есть ограничения на qpos (углы поворота) и я беру нижнюю и верхнюю границы
        # Функция выдает model.jnt_range выдает ограничения на углы в модели, на углу joint 
        # Мне всегдла выдаются, нижнее и верхнее ограничение, поэтому беру первое значение и второе.

        # ограничения на FR, FL, RR, RL ноги 
        # model.jnt_range[1:]  (нулевой выкидываем, тк он равен 0 - это free_joint, для него мы отдельно выше )
        hinge_joints = self.model.jnt_range[1:] 
        
        # Используем более широкие границы для суставов
        # Иногда при динамике среды джойнты выходят за границу суставов.
        # Mujoco использует численное интегрирование
        # Каждый шаг mj_step() добавляет небольшую ошибку или 
        joint_margin = 0.03  # Запас 3%
        hinge_joints[:,0] -= joint_margin
        hinge_joints[:,1] += joint_margin

        qpos_low = np.concatenate([free_joint, hinge_joints])[:,0].astype(np.float32)
        qpos_high = np.concatenate([free_joint, hinge_joints])[:,1].astype(np.float32)

        # Проверяем размерности, что они совпадают после преобразований 
        assert len(qpos_low) == self.model.nq, "qpos_low dimension mismatch"
        assert len(qpos_high) == self.model.nq, "qpos_high dimension mismatch"

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {   

                # array([[ 0.   ,  0.   ],
                #    [-0.863,  0.863],
                #    [-0.686,  4.501],
                #    [-2.818, -0.888],
                #    [-0.863,  0.863],
                # model.actuator_forcerange ограничения поданные на акуатор
                 "qpos": gym.spaces.Box(low = qpos_low, high = qpos_high,  shape=(self.model.nq,), dtype=np.float32),   #  углы с ограничениями
                 "qvel": gym.spaces.Box(low = -np.inf ,high = np.inf, shape=(self.model.nu,), dtype=np.float32),  # [x, y] coordinates
            }
        )

        # Здесь ставим ограничения актуаторов: нижняя и верхняя границы
        qvel_low = self.model.actuator_ctrlrange[:,0].astype(np.float32)
        qpos_high = self.model.actuator_ctrlrange[:,1].astype(np.float32)
        self.action_space = gym.spaces.Box(low = qvel_low, high = qpos_high, shape=(self.model.nu,), dtype=np.float32)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """

        qpos = np.float32(self.data.qpos.copy())
        
        # Берем только скорости ног, первые 6 выкидываем, тк это скорость тела frejoint
        qvel = np.float32(self.data.qvel[6:].copy())

        # print("_get_obs len qpos", len(qpos))
        # print("_get_obs qpos", qpos)
        # print("\n")
        # print("_get_obs len qvel", len(qvel))
        # print("_get_obs qvel", qvel)

        # print("=========")
        # print("shape .observation_space['qpos'].shape", self.observation_space['qpos'].shape)
        # print("self.observation_space", self.observation_space)

        # # Проверяем соответствие пространству наблюдений
        # assert self.observation_space["qpos"].contains(qpos), "qpos out of bounds"
        # assert self.observation_space["qvel"].contains(qvel), "qvel out of bounds"
    
        observation_space = {
                "qpos": qpos,  # координаты 
                "qvel": qvel,  # скорости
            }
        
        return observation_space
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        
        return {
                "qpos": self.data.qpos.copy(), # координаты 
                "qvel": self.data.qvel.copy(),  # скорости
                'is_terminate': self._is_terminated(),
                'is_trancate': self._is_terminated(),

            }
    
    def reset(self, seed=None, options=None):

        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)

        self.current_episode_time = 0
        self.current_episode_steps = 0

 
        
        #  Рандомная инициализация углов ног.
        # !!!! Улучшить эту часть с помощью рандома
        # Вовзращает в состоанияе keyframe, те
        # <keyframe>
        #     <key name="home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"
        #     ctrl="0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"/>
        # </keyframe>
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            # Устанавливаем состояние из ключевого кадра
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            # Если ключевого кадра нет, используем обычный сброс
            mujoco.mj_resetData(self.model, self.data)

        # Если устаналивать в key позу, то можно это не писать, тк это автоматически сделаемся в
        # mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.qpos = self.data.qpos.copy()
        self.qvel = self.data.qvel.copy()

        mujoco.mj_forward(self.model, self.data)

        # print("len qpos", len(self.qpos))
        # print("qpos", self.qpos)
        # print("=====")
        # print("len qvel", len(self.qvel))
        # print("qvel", self.qvel)

        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    

    def _is_terminated(self):
        trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        trunk_position = self.data.xpos[trunk_body_id]  # [x, y, z] в метрах

        # hight_on_floor = self.data.qpos[2] # Берем координату Z
        hight_on_floor = trunk_position[2]

        # Если опустился ниже 10см (опытным путем на симуляции) и выше 2метров (верхнее ограничение так на всякий случай)
        if hight_on_floor < 0.10 or hight_on_floor > 2:
            return True
        return False
        

    def _is_truncated(self):
        if self.current_episode_time >= self.max_episode_time or \
            self.current_episode_steps >= self.max_episode_steps: 
            return True
        return False
    
    def _compute_reward(self, prev_qpos):
        # Получаем ID тела для точного доступа
        trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if trunk_body_id < 0:
            trunk_body_id = 1  # fallback
        
        rewards = {
            'forward_velocity': self.data.qvel[0],  # линейная скорость по X
            'height_penalty': -abs(self.data.qpos[2] - 0.3),  # высота из qpos
            'orientation_penalty': 1.0 - abs(self.data.qpos[3] - 1.0),  # кватернион w
        }
        
        # Взвешиваем компоненты награды
        self.reward = (
            rewards['forward_velocity'] * 1.0 +
            rewards['height_penalty'] * 0.1 +
            rewards['orientation_penalty'] * 0.5
        )

        return self.reward

        
    def step(self, action):

        # Проверяем действие перед применением
        if not self.action_space.contains(action):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # Предыдущие значения qpos сохраняем для расчета награды
        prev_qpos = self.data.qpos.copy()

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        self.current_episode_time += self.dt
        self.current_episode_steps +=1

        observation = self._get_obs()
        info = self._get_info()

        reward = self._compute_reward(prev_qpos)
        terminated = self._is_terminated()
        truncated = self._is_truncated()


        # Добавляем в info
        info.update({
            'reward' : reward,
            'terminated' : terminated,
            'truncated' : truncated
        })

        return observation, reward, terminated, truncated, info

    def render(self):
        """- Renders the environments to help visualise what the agent see, examples modes are “human”, “rgb_array”, “ansi” for text."""
        pass

    def close(self):
        pass
