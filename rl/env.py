"""
AssaultEnv — 与 game.js 物理完全对齐的 gymnasium 环境。

只模拟 assault 姿态下 ally 的决策：给定当前帧的世界状态，
输出 9 个离散移动方向之一（含静止），攻击逻辑不变（始终朝最近敌人开火）。

物理常量全部来自 game.js，禁止随意修改。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 从 game.js 精确对齐的常量 ─────────────────────────────────────────────────

CANVAS_W: float = 900.0
CANVAS_H: float = 540.0

# ally 参数（game.js state.ally）
ALLY_RADIUS: float = 13.0
ALLY_MAX_HP: float = 160.0
ALLY_BASE_SPEED: float = 200.0

# assault 配置（game.js allyConfig() assault 分支）
ASSAULT_ATTACK_RANGE: float = 110.0  # 停止前进距离
ASSAULT_KITE_RANGE: float = 55.0     # 开始后撤距离
ASSAULT_INTERVAL: float = 0.45       # 攻击冷却
ASSAULT_SPEED_MUL: float = 1.2
ASSAULT_DAMAGE: float = 13.0
ASSAULT_STRAFE_AMP: float = 0.3
ASSAULT_BULLET_SPEED: float = 400.0

# 敌人参数（game.js spawnEnemies）
MOB_RADIUS: float = 12.0
MOB_BASE_HP: float = 30.0
MOB_BASE_SPEED: float = 42.0
MOB_SHOOT_CD_MIN: float = 0.8
MOB_SHOOT_CD_MAX: float = 1.8
MOB_BULLET_SPEED: float = 170.0
MOB_BULLET_DAMAGE: float = 5.0
MOB_SHOOT_CD_RESET: float = 1.6

BOSS_RADIUS: float = 20.0
BOSS_BASE_HP: float = 200.0
BOSS_BASE_SPEED: float = 32.0
BOSS_BULLET_SPEED: float = 200.0
BOSS_BULLET_DAMAGE: float = 9.0
BOSS_SHOOT_CD_RESET: float = 1.2

# 子弹参数
BULLET_RADIUS: float = 4.0
BULLET_TTL: float = 2.2
ALLY_BULLET_SPEED: float = 400.0

# 仿真步长：匹配浏览器 60fps，每步 dt = 1/60
DT: float = 1.0 / 60.0
# 每 episode 最多步数：60 fps × 40 秒
MAX_STEPS: int = 60 * 40

# floor scale 乘数范围（训练覆盖 floor 1-6）
FLOOR_RANGE = (1, 6)

# 障碍物布局，精确复刻 game.js OBSTACLE_LAYOUTS（x, y, w, h）
OBSTACLE_LAYOUTS: list[list[dict]] = [
    [
        {"x": 410, "y": 220, "w": 100, "h": 100},
        {"x": 270, "y": 160, "w": 70,  "h": 40},
        {"x": 620, "y": 320, "w": 70,  "h": 40},
    ],
    [
        {"x": 370, "y": 140, "w": 90, "h": 50},
        {"x": 370, "y": 350, "w": 90, "h": 50},
        {"x": 265, "y": 235, "w": 55, "h": 70},
        {"x": 640, "y": 235, "w": 55, "h": 70},
    ],
    [
        {"x": 290, "y": 155, "w": 85, "h": 45},
        {"x": 540, "y": 195, "w": 85, "h": 45},
        {"x": 290, "y": 340, "w": 85, "h": 45},
        {"x": 540, "y": 310, "w": 85, "h": 45},
    ],
]

# 敌人出生位置（game.js MOB_BASE_POSITIONS）
MOB_BASE_POSITIONS = [
    (520, 100), (650, 150), (780, 100),
    (560, 380), (700, 430), (820, 360),
    (700, 270), (820, 200), (760, 430),
    (850, 130),
]

# 9 个离散动作对应的方向向量（含静止），与 WASD 8方向对齐
# 0=静止 1=上 2=右上 3=右 4=右下 5=下 6=左下 7=左 8=左上
_SQ2 = math.sqrt(2) / 2
ACTION_VECTORS: list[tuple[float, float]] = [
    (0.0,  0.0),   # 0 静止
    (0.0, -1.0),   # 1 上
    (_SQ2, -_SQ2), # 2 右上
    (1.0,  0.0),   # 3 右
    (_SQ2,  _SQ2), # 4 右下
    (0.0,  1.0),   # 5 下
    (-_SQ2, _SQ2), # 6 左下
    (-1.0, 0.0),   # 7 左
    (-_SQ2,-_SQ2), # 8 左上
]

# obs 维度常量
MAX_ENEMIES = 5       # 最多观测 5 个敌人（按距离排序）
MAX_BULLETS = 8       # 最多观测 8 颗敌方子弹（按距离排序）
MAX_OBSTACLES = 4     # 最多 4 个障碍物（pad 到固定长度）

# obs 各段长度：
#   自身状态:       6
#   主目标敌人:     6
#   其余敌人:       (MAX_ENEMIES-1) × 5 = 20
#   敌方子弹:       MAX_BULLETS × 5    = 40
#   障碍物:         MAX_OBSTACLES × 6  = 24
#   边界距离:       4
#   攻击冷却比例:   1
#   ─────────────
#   合计:           101
OBS_DIM = 6 + 6 + (MAX_ENEMIES - 1) * 5 + MAX_BULLETS * 5 + MAX_OBSTACLES * 6 + 4 + 1


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    x: float
    y: float
    radius: float
    hp: float
    max_hp: float
    speed: float

    def dist(self, other: "Entity | Bullet") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Enemy(Entity):
    kind: str = "mob"          # "mob" | "boss"
    shoot_cd: float = 0.0


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    damage: float
    ttl: float


# ── 物理工具函数（精确复刻 game.js）─────────────────────────────────────────

def _normalize(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def _collides_with_obstacle(cx: float, cy: float, radius: float,
                             obstacles: list[dict]) -> bool:
    for o in obstacles:
        near_x = max(o["x"], min(cx, o["x"] + o["w"]))
        near_y = max(o["y"], min(cy, o["y"] + o["h"]))
        if math.hypot(cx - near_x, cy - near_y) < radius:
            return True
    return False


def _bullet_hits_obstacle(bx: float, by: float, obstacles: list[dict]) -> bool:
    for o in obstacles:
        if o["x"] <= bx <= o["x"] + o["w"] and o["y"] <= by <= o["y"] + o["h"]:
            return True
    return False


def _move_with_collision(entity: Entity, dx: float, dy: float,
                          obstacles: list[dict]) -> None:
    """精确复刻 game.js moveWithCollision：分量滑动。"""
    r = entity.radius
    new_x = max(r, min(entity.x + dx, CANVAS_W - r))
    new_y = max(r, min(entity.y + dy, CANVAS_H - r))

    if not _collides_with_obstacle(new_x, new_y, r, obstacles):
        entity.x = new_x
        entity.y = new_y
    elif not _collides_with_obstacle(new_x, entity.y, r, obstacles):
        entity.x = new_x
    elif not _collides_with_obstacle(entity.x, new_y, r, obstacles):
        entity.y = new_y


def _create_bullet(owner_x: float, owner_y: float,
                   target_x: float, target_y: float,
                   speed: float, damage: float) -> Bullet:
    nx, ny = _normalize(target_x - owner_x, target_y - owner_y)
    return Bullet(x=owner_x, y=owner_y,
                  vx=nx * speed, vy=ny * speed,
                  radius=BULLET_RADIUS, damage=damage, ttl=BULLET_TTL)


# ── 环境主体 ──────────────────────────────────────────────────────────────────

class AssaultEnv(gym.Env):
    """
    模拟 assault 姿态下 ally 的单步决策。

    观测空间（101 维 float32，全部归一化到 [-1, 1] 或 [0, 1]）：
        [0:6]   自身状态
        [6:12]  主目标敌人（最近）详细信息
        [12:32] 其余最多 4 个敌人简要信息
        [32:72] 最多 8 颗最近敌方子弹信息
        [72:96] 最多 4 个障碍物信息
        [96:100] 到四壁的归一化距离
        [100]   攻击冷却比例

    动作空间：Discrete(9)，0=静止，1-8=8方向移动
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        # 运行时状态（reset 初始化）
        self._ally: Entity = None        # type: ignore[assignment]
        self._enemies: list[Enemy] = []
        self._ally_bullets: list[Bullet] = []
        self._enemy_bullets: list[Bullet] = []
        self._obstacles: list[dict] = []
        self._attack_cd: float = 0.0
        self._step_count: int = 0
        self._floor: int = 1
        self._hp_mul: float = 1.0
        self._speed_mul: float = 1.0

    # ── 公开接口 ────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        floor = random.randint(*FLOOR_RANGE)
        self._floor = floor
        f = floor - 1
        self._hp_mul = 1.0 + f * 0.30
        self._speed_mul = 1.0 + f * 0.06
        mob_count = min(3 + int(f * 1.2), 10)

        # 随机选障碍物布局
        self._obstacles = OBSTACLE_LAYOUTS[(floor - 1) % len(OBSTACLE_LAYOUTS)]

        # 生成 ally，出生在左侧安全区随机位置
        self._ally = Entity(
            x=random.uniform(160, 240),
            y=random.uniform(220, 320),
            radius=ALLY_RADIUS,
            hp=ALLY_MAX_HP,
            max_hp=ALLY_MAX_HP,
            speed=ALLY_BASE_SPEED,
        )

        # 生成敌人
        self._enemies = []
        for i in range(mob_count):
            bx, by = MOB_BASE_POSITIONS[i % len(MOB_BASE_POSITIONS)]
            self._enemies.append(Enemy(
                x=bx + random.uniform(-20, 20),
                y=by + random.uniform(-20, 20),
                radius=MOB_RADIUS,
                hp=round(MOB_BASE_HP * self._hp_mul),
                max_hp=round(MOB_BASE_HP * self._hp_mul),
                speed=MOB_BASE_SPEED * self._speed_mul,
                kind="mob",
                shoot_cd=random.uniform(MOB_SHOOT_CD_MIN, MOB_SHOOT_CD_MAX),
            ))
        self._enemies.append(Enemy(
            x=820.0, y=270.0,
            radius=BOSS_RADIUS,
            hp=round(BOSS_BASE_HP * self._hp_mul),
            max_hp=round(BOSS_BASE_HP * self._hp_mul),
            speed=BOSS_BASE_SPEED * self._speed_mul,
            kind="boss",
            shoot_cd=0.8,
        ))

        self._ally_bullets = []
        self._enemy_bullets = []
        self._attack_cd = 0.0
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ally is not None, "call reset() first"
        self._step_count += 1

        ally_hp_before = self._ally.hp
        damage_dealt = 0.0

        # 1. ally 移动
        speed = self._ally.speed * ASSAULT_SPEED_MUL
        mx, my = ACTION_VECTORS[action]
        _move_with_collision(self._ally, mx * speed * DT, my * speed * DT,
                             self._obstacles)

        # 2. ally 攻击（自动朝最近敌人，不由 action 控制）
        self._attack_cd = max(0.0, self._attack_cd - DT)
        target = self._nearest_enemy()
        if target is not None and self._attack_cd <= 0.0:
            self._ally_bullets.append(
                _create_bullet(self._ally.x, self._ally.y,
                               target.x, target.y,
                               ALLY_BULLET_SPEED, ASSAULT_DAMAGE)
            )
            self._attack_cd = ASSAULT_INTERVAL

        # 3. 更新敌人移动 + 开火
        for enemy in self._enemies:
            enemy.shoot_cd = max(0.0, enemy.shoot_cd - DT)
            nx, ny = _normalize(self._ally.x - enemy.x, self._ally.y - enemy.y)
            _move_with_collision(enemy, nx * enemy.speed * DT, ny * enemy.speed * DT,
                                 self._obstacles)
            if enemy.shoot_cd <= 0.0:
                spd = BOSS_BULLET_SPEED if enemy.kind == "boss" else MOB_BULLET_SPEED
                dmg = BOSS_BULLET_DAMAGE if enemy.kind == "boss" else MOB_BULLET_DAMAGE
                cd  = BOSS_SHOOT_CD_RESET if enemy.kind == "boss" else MOB_SHOOT_CD_RESET
                self._enemy_bullets.append(
                    _create_bullet(enemy.x, enemy.y,
                                   self._ally.x, self._ally.y, spd, dmg)
                )
                enemy.shoot_cd = cd

        # 4. 移动子弹 + 碰撞检测
        damage_dealt = self._update_bullets()

        # 5. 清除死亡敌人
        self._enemies = [e for e in self._enemies if e.hp > 0]

        # 6. 计算奖励
        hp_lost = max(0.0, ally_hp_before - self._ally.hp)
        reward = self._compute_reward(damage_dealt, hp_lost, target)

        # 7. 终止条件
        terminated = self._ally.hp <= 0 or len(self._enemies) == 0
        truncated = self._step_count >= MAX_STEPS

        return self._get_obs(), reward, terminated, truncated, {
            "damage_dealt": damage_dealt,
            "hp_lost": hp_lost,
            "enemies_left": len(self._enemies),
        }

    # ── 观测构建 ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        构建 101 维归一化观测向量。

        归一化约定：
          - 位置：/ CANVAS_W 或 / CANVAS_H  → [0, 1]
          - 相对位置（dx/dy）：/ CANVAS_W|H，已有符号 → [-1, 1]
          - 距离：/ 对角线长度（sqrt(900²+540²)≈1051）→ [0, 1]
          - hp：/ max_hp → [0, 1]
          - 速度：/ 500（最大可能速度）→ [0, 1]
          - 冷却：直接比例 [0, 1]
          - 障碍物尺寸：/ CANVAS_W|H → [0, 1]
        """
        DIAG = math.hypot(CANVAS_W, CANVAS_H)  # ~1051
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        ally = self._ally

        # ── 段1：自身状态 (6维) ─────────────────────────────────────────────
        # [0] 归一化 x 位置
        # [1] 归一化 y 位置
        # [2] hp 比例
        # [3] 到最近障碍物的归一化距离（方位感知）
        # [4] 当前攻击冷却比例
        # [5] 当前敌人数量归一化（/ 11，最多 10mob+1boss）
        obs[idx]   = ally.x / CANVAS_W
        obs[idx+1] = ally.y / CANVAS_H
        obs[idx+2] = ally.hp / ally.max_hp
        obs[idx+3] = self._dist_to_nearest_obstacle() / DIAG
        obs[idx+4] = self._attack_cd / ASSAULT_INTERVAL
        obs[idx+5] = len(self._enemies) / 11.0
        idx += 6

        # ── 段2：主目标敌人（最近）(6维) ────────────────────────────────────
        # [0] 相对 dx（有符号，/ CANVAS_W）
        # [1] 相对 dy（有符号，/ CANVAS_H）
        # [2] 归一化距离
        # [3] hp 比例
        # [4] 是否 boss（0/1）
        # [5] 敌人射击冷却比例（越小越快要开枪）
        target = self._nearest_enemy()
        if target is not None:
            dx = target.x - ally.x
            dy = target.y - ally.y
            dist = math.hypot(dx, dy)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = dist / DIAG
            obs[idx+3] = target.hp / target.max_hp
            obs[idx+4] = 1.0 if target.kind == "boss" else 0.0
            obs[idx+5] = target.shoot_cd / (BOSS_SHOOT_CD_RESET if target.kind == "boss"
                                             else MOB_SHOOT_CD_RESET)
        idx += 6

        # ── 段3：其余最多 (MAX_ENEMIES-1) 个敌人 (5维/敌) ───────────────────
        # 按距离排序，跳过主目标
        others = sorted(
            [e for e in self._enemies if e is not target],
            key=lambda e: math.hypot(e.x - ally.x, e.y - ally.y)
        )[: MAX_ENEMIES - 1]
        for e in others:
            dx = e.x - ally.x
            dy = e.y - ally.y
            dist = math.hypot(dx, dy)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = dist / DIAG
            obs[idx+3] = e.hp / e.max_hp
            obs[idx+4] = 1.0 if e.kind == "boss" else 0.0
            idx += 5
        idx += (MAX_ENEMIES - 1 - len(others)) * 5  # padding（保持为零）

        # ── 段4：最多 MAX_BULLETS 颗最近敌方子弹 (5维/颗) ──────────────────
        # 排序依据：预测碰撞威胁度 = 子弹到 ally 的距离 / 子弹速度（越小越危险）
        # 5维：相对位置 dx/dy，速度方向 vx/vy（归一化），距离
        MAX_BULLET_DIST = 200.0  # 超出此范围的子弹视为无威胁，obs 填零
        threat_bullets = sorted(
            [b for b in self._enemy_bullets
             if math.hypot(b.x - ally.x, b.y - ally.y) < MAX_BULLET_DIST],
            key=lambda b: math.hypot(b.x - ally.x, b.y - ally.y)
        )[: MAX_BULLETS]
        for b in threat_bullets:
            dx = b.x - ally.x
            dy = b.y - ally.y
            dist = math.hypot(dx, dy)
            bspd = math.hypot(b.vx, b.vy) or 1.0
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = b.vx / bspd          # 归一化速度方向 x
            obs[idx+3] = b.vy / bspd          # 归一化速度方向 y
            obs[idx+4] = dist / MAX_BULLET_DIST
            idx += 5
        idx += (MAX_BULLETS - len(threat_bullets)) * 5  # padding

        # ── 段5：障碍物 (6维/块，pad 到 MAX_OBSTACLES) ──────────────────────
        # 按到 ally 中心的距离排序
        # 6维：中心相对 dx/dy，宽/高归一化，ally 到障碍物最近点的距离，方向余弦（dx/dist）
        sorted_obs = sorted(
            self._obstacles,
            key=lambda o: math.hypot(
                (o["x"] + o["w"] / 2) - ally.x,
                (o["y"] + o["h"] / 2) - ally.y
            )
        )[: MAX_OBSTACLES]
        for o in sorted_obs:
            cx = o["x"] + o["w"] / 2
            cy = o["y"] + o["h"] / 2
            dx = cx - ally.x
            dy = cy - ally.y
            near_x = max(o["x"], min(ally.x, o["x"] + o["w"]))
            near_y = max(o["y"], min(ally.y, o["y"] + o["h"]))
            near_dist = math.hypot(ally.x - near_x, ally.y - near_y)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = o["w"] / CANVAS_W
            obs[idx+3] = o["h"] / CANVAS_H
            obs[idx+4] = near_dist / DIAG
            obs[idx+5] = (dx / (math.hypot(dx, dy) or 1.0))  # 方向余弦 x
            idx += 6
        idx += (MAX_OBSTACLES - len(sorted_obs)) * 6  # padding

        # ── 段6：到四壁的归一化距离 (4维) ───────────────────────────────────
        # 顺序：上/下/左/右
        obs[idx]   = ally.y / CANVAS_H
        obs[idx+1] = (CANVAS_H - ally.y) / CANVAS_H
        obs[idx+2] = ally.x / CANVAS_W
        obs[idx+3] = (CANVAS_W - ally.x) / CANVAS_W
        idx += 4

        # ── 段7：攻击冷却比例 (1维) ─────────────────────────────────────────
        obs[idx] = min(1.0, self._attack_cd / ASSAULT_INTERVAL)
        idx += 1

        assert idx == OBS_DIM, f"obs dim mismatch: {idx} != {OBS_DIM}"
        return obs

    # ── 奖励函数 ────────────────────────────────────────────────────────────

    def _compute_reward(self, damage_dealt: float, hp_lost: float,
                        target: Enemy | None) -> float:
        """
        奖励设计原则：
          1. 核心驱动：对敌造成伤害 → 正奖励
          2. 生存压力：被打中 → 负奖励（比伤害正奖励权重更高，迫使躲避）
          3. 位置引导：在攻击射程内但不过近 → 小正奖励（减少无效游荡）
          4. 接近引导：距目标过远时轻微惩罚（防止一直逃）
          5. 死亡惩罚：终止大惩罚
          6. 胜利奖励：消灭所有敌人
          7. 超时惩罚：鼓励速攻

        权重经验值，可通过 train.py 的 reward_scale 参数调整。
        """
        reward = 0.0
        ally = self._ally

        # 1. 伤害奖励：每点伤害 +0.08（一颗子弹打 mob 约 +1.0）
        reward += damage_dealt * 0.08

        # 2. 受伤惩罚：权重为伤害奖励的 2 倍，迫使学习躲避
        #    mob 子弹 5 伤 → -0.5，boss 子弹 9 伤 → -0.9
        reward -= hp_lost * 0.10

        # 3. 位置质量奖励：在有效攻击射程内（kite_range < dist < attack_range）
        if target is not None:
            dist = ally.dist(target)
            if ASSAULT_KITE_RANGE < dist < ASSAULT_ATTACK_RANGE:
                # 在射程内持续给小正奖励，鼓励维持攻击位置
                reward += 0.004
            elif dist > ASSAULT_ATTACK_RANGE * 2.0:
                # 距目标过远，轻微惩罚（不要一直逃）
                reward -= 0.003
            elif dist < ASSAULT_KITE_RANGE * 0.5:
                # 过近，会被近战打死，轻微惩罚
                reward -= 0.002

        # 4. 障碍物感知奖励：贴墙站立时轻微惩罚（鼓励利用掩体而非卡在角落）
        near_obs_dist = self._dist_to_nearest_obstacle()
        if near_obs_dist < ally.radius + 2:
            # 真正卡进障碍物边缘，惩罚（通常不会发生，但有分量滑动时可能贴墙站）
            reward -= 0.002

        # 5. 死亡大惩罚
        if ally.hp <= 0:
            reward -= 30.0

        # 6. 消灭所有敌人的胜利奖励
        if len(self._enemies) == 0:
            reward += 50.0

        # 7. 时间效率惩罚：每步 -0.001，鼓励速攻而不是无限消耗
        reward -= 0.001

        return float(reward)

    # ── 内部辅助 ────────────────────────────────────────────────────────────

    def _nearest_enemy(self) -> Enemy | None:
        if not self._enemies:
            return None
        return min(self._enemies, key=lambda e: self._ally.dist(e))

    def _dist_to_nearest_obstacle(self) -> float:
        if not self._obstacles:
            return float(math.hypot(CANVAS_W, CANVAS_H))
        ally = self._ally
        min_dist = float("inf")
        for o in self._obstacles:
            near_x = max(o["x"], min(ally.x, o["x"] + o["w"]))
            near_y = max(o["y"], min(ally.y, o["y"] + o["h"]))
            d = math.hypot(ally.x - near_x, ally.y - near_y)
            min_dist = min(min_dist, d)
        return min_dist

    def _update_bullets(self) -> float:
        """移动所有子弹，处理碰撞，返回 ally 对敌造成的总伤害。"""
        damage_dealt = 0.0

        # ally 子弹移动 + 命中检测
        new_ally_bullets = []
        for b in self._ally_bullets:
            b.x += b.vx * DT
            b.y += b.vy * DT
            b.ttl -= DT
            if (b.ttl <= 0
                    or b.x < -10 or b.x > CANVAS_W + 10
                    or b.y < -10 or b.y > CANVAS_H + 10
                    or _bullet_hits_obstacle(b.x, b.y, self._obstacles)):
                continue
            hit = False
            for e in self._enemies:
                if math.hypot(b.x - e.x, b.y - e.y) <= b.radius + e.radius:
                    e.hp -= b.damage
                    damage_dealt += b.damage
                    hit = True
                    break
            if not hit:
                new_ally_bullets.append(b)
        self._ally_bullets = new_ally_bullets

        # 敌方子弹移动 + 命中检测
        new_enemy_bullets = []
        for b in self._enemy_bullets:
            b.x += b.vx * DT
            b.y += b.vy * DT
            b.ttl -= DT
            if (b.ttl <= 0
                    or b.x < -10 or b.x > CANVAS_W + 10
                    or b.y < -10 or b.y > CANVAS_H + 10
                    or _bullet_hits_obstacle(b.x, b.y, self._obstacles)):
                continue
            if math.hypot(b.x - self._ally.x, b.y - self._ally.y) <= b.radius + self._ally.radius:
                self._ally.hp -= b.damage
            else:
                new_enemy_bullets.append(b)
        self._enemy_bullets = new_enemy_bullets

        return damage_dealt
