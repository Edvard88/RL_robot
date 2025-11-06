import time
import numpy as np

import mujoco
import mujoco.viewer

from stable_baselines3 import PPO

from robot_env import RobotGo1Env


def load_model_and_env(xml_path, model_path):
    """Загрузка среды и обученной модели"""
    go1_env = RobotGo1Env(xml_path)
    obs, info = go1_env.reset()

    ppo_model = PPO.load(model_path)
    
    return go1_env, ppo_model, obs


def run_simulation(go1_env, ppo_model, obs, step_num=10000, render_delay=0.002):
    """Запуск симуляции с визуализацией"""
    with mujoco.viewer.launch_passive(go1_env.model, go1_env.data) as viewer:
        for i in range(step_num):
            if not viewer.is_running():
                print("Problems with viewer.is_running")
                break
                
            # Получаем действие от модели
            action, _states = ppo_model.predict(obs)
            
            # Выполняем шаг в среде
            obs, rewards, terminated, truncated, info = go1_env.step(action)
            
            # Выводим информацию о состоянии
            print(f"Шаг {i}: reward={rewards:.3f}, terminated={terminated}, truncated={truncated}")
            
            # Синхронизируем визуализацию
            viewer.sync()
            time.sleep(render_delay)
            
            # # Сброс среды если эпизод завершен
            # if terminated or truncated:
            #     print(f"Эпизод завершен на шаге {i}, сброс среды")
            #     obs, info = go1_env.reset()

    viewer.close()



def print_environment_info(go1_env, obs):
    """Вывод информации о среде и начальном состоянии"""
    print("=== ИНФОРМАЦИЯ О СРЕДЕ ===")
    print(f"Пространство действий: {go1_env.action_space}")
    print(f"Пространство наблюдений: {go1_env.observation_space}")
    print(f"Начальное наблюдение: {obs}")
    print("==========================")


if __name__ == "__main__":
    # Параметры
    XML_PATH = 'robot_models/unitree_go1/scene.xml'
    MODEL_PATH = "rl_models/current_model/go1_ppo_10000000_steps"
    #MODEL_PATH = "rl_models/best/best_model"
    
    STEP_NUM = 10000
    RENDER_DELAY = 0.002
    
    try:
        # Загрузка модели и среды
        print("🔄 Загрузка среды и модели...")
        go1_env, ppo_model, obs = load_model_and_env(XML_PATH, MODEL_PATH)
        
        # Вывод информации
        print_environment_info(go1_env, obs)
        
        # Запуск симуляции
        print("🎮 Запуск симуляции...")
        run_simulation(go1_env, ppo_model, obs, STEP_NUM, RENDER_DELAY)
        
        print("✅ Симуляция завершена успешно!")
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Файл не найден - {e}")
    except Exception as e:
        print(f"❌ Ошибка во время выполнения: {e}")
    # finally:
    #     # Корректное закрытие
    #     if 'viewer' in locals():
    #         viewer.close()








# # Cмотрим как отработал
# from go1_env import RobotLearning



# xml_path = 'robot_models/unitree_go1/scene.xml'
# go1_env = RobotLearning(xml_path)
# obs, info = go1_env.reset()

# ppo_model = PPO.load("train_models/current_model/go1_ppo_3000000_steps_v1")  

# STEP_NUM = 10000
# def controller(model, data, ppo_model, obs):

#     action, _states = ppo_model.predict(obs)
#     data.ctrl[:] = action
    


# with mujoco.viewer.launch_passive(go1_env.model, go1_env.data) as viewer:
#     for i in range(STEP_NUM):
#         if viewer.is_running():

#             action, _states = ppo_model.predict(obs)
#             obs, rewards, terminated, truncated, info = go1_env.step(action)
#             print("obs", obs)
#             print("rewards", rewards)
#             print("terminated", terminated)
#             print("truncated", truncated)
            
        
#             # mujoco.set_mjcb_control(controller(model, ppo_model, obs))
#             # mujoco.mj_step(model, data)

#             viewer.sync()
#             time.sleep(0.002)
#         else:
#             break
# viewer.close()


# if __name__ == "__main__":
