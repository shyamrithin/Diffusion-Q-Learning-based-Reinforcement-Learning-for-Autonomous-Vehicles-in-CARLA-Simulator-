# ==========================================================
# record_sac.py
# SAC inference with full CSV recording for paper graphs
# Records: position, control signals, speed, reward
#
# Usage:
#   python3 record_sac.py --traffic light --episodes 3
#   python3 record_sac.py --traffic medium --episodes 3
#   python3 record_sac.py --traffic heavy --episodes 3
# ==========================================================

import os, sys, math, time, csv, logging, argparse
import traceback
import numpy as np
import torch
import pygame
import carla
import gymnasium as gym
import rlcarla

from agents.sac               import SAC
from rlcarla.core.obs_builder import OBS_DIM

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("record_sac")
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints_sac")
parser.add_argument("--traffic", default="light",
    choices=["empty","light","medium","heavy","chaos"])
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--width",    type=int, default=1280)
parser.add_argument("--height",   type=int, default=720)
args = parser.parse_args()

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR  = f"results/sac_{args.traffic}"
os.makedirs(OUT_DIR, exist_ok=True)

pygame.init()
screen = pygame.display.set_mode((args.width, args.height))
pygame.display.set_caption(f"SAC — {args.traffic}")
clock  = pygame.font.SysFont("monospace", 16)
font   = pygame.font.SysFont("monospace", 16)

def load_model():
    path = os.path.join(args.checkpoint, "sac_actor_9999.pth")
    agent = SAC(state_dim=OBS_DIM, action_dim=3,
                max_action=1.0, device=DEVICE)
    agent.load_model(args.checkpoint, id=9999)
    agent.actor.eval()
    logger.info(f"Loaded SAC from {path}")
    return agent

def get_action(agent, state, prev, alpha=0.7):
    with torch.no_grad():
        st = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)
        action, _ = agent.actor.sample(st)
    raw = action.cpu().numpy().flatten().astype(np.float32)
    sm  = (alpha * raw + (1-alpha) * prev).astype(np.float32)
    result = np.array([
        float(np.clip(sm[0],  0.0, 1.0)),
        float(np.clip(sm[1], -1.0, 1.0)),
        float(np.clip(sm[2],  0.0, 1.0)),
    ], dtype=np.float32)
    return result

def save_csv(records, path):
    if not records: return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {len(records)} steps → {path}")

def main():
    agent = load_model()
    env   = gym.make("RLCarla-v0")
    env.unwrapped._traffic_preset    = args.traffic
    env.unwrapped.cfg.TRAFFIC_LIGHTS = "off"
    env.unwrapped.cfg.SPECTATOR_MODE = "follow"

    all_summaries = []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        env.unwrapped.set_camera_view("third_person")

        prev_action = np.zeros(3, dtype=np.float32)
        ep_reward   = 0.0
        step        = 0
        records     = []
        term_reason = None

        logger.info(f"Episode {ep}/{args.episodes} | traffic={args.traffic}")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); return
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                env.close(); pygame.quit(); return

            action = get_action(agent, state, prev_action)

            next_state, reward, done, trunc, info = env.step(action)

            # Record telemetry
            if env.unwrapped.vehicle:
                loc = env.unwrapped.vehicle.get_location()
                vel = env.unwrapped.vehicle.get_velocity()
                tf  = env.unwrapped.vehicle.get_transform()
                spd = math.sqrt(vel.x**2+vel.y**2+vel.z**2)*3.6
                records.append({
                    "episode"  : ep,
                    "step"     : step,
                    "x"        : round(loc.x, 3),
                    "y"        : round(loc.y, 3),
                    "heading"  : round(tf.rotation.yaw, 2),
                    "speed_kmh": round(spd, 3),
                    "throttle" : round(float(action[0]), 4),
                    "steer"    : round(float(action[1]), 4),
                    "brake"    : round(float(action[2]), 4),
                    "reward"   : round(reward, 4),
                    "ep_reward": round(ep_reward, 2),
                    "collision": int(info.get("collision_flag", False)),
                    "offroad"  : int(info.get("offroad_flag", False)),
                    "term"     : "",
                })

            state        = next_state
            prev_action  = action
            ep_reward   += reward
            step        += 1

            # Render
            frame = env.unwrapped.get_camera_frame()
            if frame is not None:
                import cv2
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
                surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))
                screen.blit(surf, (0,0))

            # HUD
            lines = [
                f"SAC | {args.traffic.upper()}",
                f"Ep: {ep}/{args.episodes}",
                f"Step: {step}",
                f"Reward: {ep_reward:.1f}",
                f"Speed: {records[-1]['speed_kmh'] if records else 0:.1f} km/h",
                f"Throttle: {action[0]:.3f}",
                f"Steer: {action[1]:.3f}",
                f"Brake: {action[2]:.3f}",
            ]
            bg = pygame.Surface((230, len(lines)*22+10), pygame.SRCALPHA)
            bg.fill((0,0,0,160))
            screen.blit(bg, (8,8))
            for i, line in enumerate(lines):
                txt = font.render(line, True, (255,255,255))
                screen.blit(txt, (14, 14+i*22))

            pygame.display.flip()
            pygame.time.Clock().tick(30)

            if done or trunc:
                term_reason = info.get("term_reason","done")
                if records:
                    records[-1]["term"] = term_reason
                break

        # Save CSV
        csv_path = os.path.join(OUT_DIR, f"ep{ep:02d}.csv")
        save_csv(records, csv_path)

        summary = {
            "episode"    : ep,
            "traffic"    : args.traffic,
            "steps"      : step,
            "reward"     : round(ep_reward, 2),
            "term_reason": term_reason,
            "collision"  : int(term_reason=="collision"),
            "offroad"    : int(term_reason=="offroad"),
            "completion" : int(term_reason=="max_steps"),
            "avg_speed"  : round(
                np.mean([r["speed_kmh"] for r in records]) if records else 0, 2),
            "avg_throttle": round(
                np.mean([r["throttle"] for r in records]) if records else 0, 3),
            "avg_steer_abs": round(
                np.mean([abs(r["steer"]) for r in records]) if records else 0, 3),
            "avg_brake"  : round(
                np.mean([r["brake"] for r in records]) if records else 0, 3),
        }
        all_summaries.append(summary)

        logger.info(
            f"Ep {ep} done | reward={ep_reward:.1f} | "
            f"steps={step} | end={term_reason}"
        )
        time.sleep(1.5)

    # Save summary CSV
    sum_path = os.path.join(OUT_DIR, "summary.csv")
    save_csv(all_summaries, sum_path)

    # Print summary
    print("\n" + "="*55)
    print(f"SAC | {args.traffic.upper()} TRAFFIC | {args.episodes} episodes")
    print("="*55)
    for s in all_summaries:
        print(f"  Ep{s['episode']} | reward={s['reward']:8.1f} | "
              f"steps={s['steps']:4d} | end={s['term_reason']:<12} | "
              f"speed={s['avg_speed']:.1f}km/h")
    print("-"*55)
    avg_r = np.mean([s["reward"] for s in all_summaries])
    avg_s = np.mean([s["steps"]  for s in all_summaries])
    cols  = sum(s["collision"]  for s in all_summaries)
    offs  = sum(s["offroad"]    for s in all_summaries)
    comps = sum(s["completion"] for s in all_summaries)
    print(f"  Avg Reward    : {avg_r:.2f}")
    print(f"  Avg Steps     : {avg_s:.1f}")
    print(f"  Collisions    : {cols}/{args.episodes}")
    print(f"  Offroads      : {offs}/{args.episodes}")
    print(f"  Completions   : {comps}/{args.episodes}")
    print("="*55)
    print(f"\nData saved to: {OUT_DIR}/")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
