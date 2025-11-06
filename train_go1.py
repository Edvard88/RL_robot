import os
import time
import numpy as np

import mujoco

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robot_env import RobotGo1Env
from robot_env_adv_v2 import AdvancedGo1Env



# checkpoint_callback = CheckpointCallback(
#     save_freq=1_000_000,
#     save_path="train_models/current_model/",
#     name_prefix="go1_ppo"
# )


class SmartCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, initial_step: int = 0, verbose=0):
        super(SmartCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.initial_step = initial_step
        self.last_save = initial_step
        
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        total_steps = self.initial_step + self.num_timesteps
        
        if total_steps - self.last_save >= self.save_freq:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{total_steps}_steps")
            self.model.save(model_path)
            print(f"Сохранена модель: {model_path}")
            self.last_save = total_steps
            
        return True


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Получаем информацию из среды
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            # Накапливаем награду за эпизод
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
            
            # Если эпизод завершен, если ключа нет, возвращаем None
            if info.get('terminated', None) or info.get('truncated', None):
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                # Логируем в TensorBoard
                if len(self.episode_rewards) > 0:
                    self.logger.record('train/episode_reward', self.current_episode_reward)
                    self.logger.record('train/episode_length', self.current_episode_length)
                    self.logger.record('train/mean_episode_reward', np.mean(self.episode_rewards))
                    self.logger.record('train/mean_episode_length', np.mean(self.episode_lengths))
                
                # Сбрасываем счетчики
                self.current_episode_reward = 0
                self.current_episode_length = 0
                
        return True

def format_training_time(seconds):
    """Форматирует время обучения в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.1f} секунд"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} минут"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} часов"

if __name__ == "__main__":
    start_time = time.time()

    XML_PATH = 'robot_models/unitree_go1/scene.xml'

    # Функция для создания среды
    def make_env(XML_PATH):
        return RobotGo1Env(XML_PATH)
        #return AdvancedGo1Env(XML_PATH)
    
    # Проверка одной среды
    test_env = make_env(XML_PATH)  # ✅ Передаем XML_PATH как аргумент
    try:
        check_env(test_env)
        print("✅ Среда прошла проверку check_env()")
    except Exception as e:
        print(f"❌ Ошибка в проверке среды: {e}")
    finally:
        test_env.close()

    
    # Создание векторной среды  для параллельности
    env = make_vec_env(
        lambda: make_env(XML_PATH),
        n_envs=8,           # Количество параллельных сред
        vec_env_cls= SubprocVecEnv #DummyVecEnv #SubprocVecEnv  # Тип векторной среды
    )


    # Использование автосохранения
    checkpoint_callback = SmartCheckpointCallback(
        initial_step=0,  # Начинаем с 2 млн шагов
        save_freq=1_000_000,       # Сохранять каждые 100к шагов
        save_path="rl_models/current_model/",
        name_prefix="go1_ppo",
        verbose=1
    )

    # Проверка на валиджационной выборке
    eval_callback = EvalCallback(
        env,
        best_model_save_path="rl_models/best/",
        log_path="logs/",
        eval_freq=1_000_000,
        n_eval_episodes=5, 
        deterministic=True,
        render=False
    )

    # Добавьте этот callback в список
    tensorboard_callback = TensorboardCallback()


    model = PPO("MultiInputPolicy",  
                env,
                tensorboard_log="./ppo_tensorboard_logs/",
                n_steps  = 8192,
                batch_size = 256,
                n_epochs=10,
                verbose=1)

    # model = PPO.load("train_models/current_model/go1_ppo_4000000_steps", env=robot_go1)  

    model.learn(total_timesteps=10_000_000, callback=[checkpoint_callback, eval_callback,  tensorboard_callback])
    model.save("rl_models/full_iteration/go1_full_iteration")

    training_time = time.time() - start_time
    
    print(f"✅ Обучение завершено за {format_training_time(training_time)}")
