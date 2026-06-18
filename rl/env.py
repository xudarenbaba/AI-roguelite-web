"""
AssaultEnv — 与 game.js 物理完全对齐的 gymnasium 环境。

只模拟 assault 姿态下 ally 的决策：给定当前帧的世界状态，
输出 9 个离散移动方向之一（含静止），攻击逻辑不变（始终朝最近敌人开火）。

物理常量全部来自 game.js，禁止随意修改。

v2 变更：
  - OBS 新增：LOS 标志位、上帧动作 one-hot、子弹预测碰撞时间、到目标距离变化量
  - 奖励：动作平滑惩罚、LOS 遮挡开火惩罚、收紧距离引导、接近奖励塑形
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 从 game.js 精确对齐的常量 ─────────────────────────────────────────────────

CANVAS_W: float = 900.0
CANVAS_H: float = 540.0

ALLY_RADIUS: float = 13.0
ALLY_MAX_HP: float = 160.0
ALLY_BASE_SPEED: float = 200.0

ASSAULT_ATTACK_RANGE: float = 110.0
ASSAULT_KITE_RANGE: float   = 65.0   # 从 55 调整到 65，保证 ally 与敌人的视觉间距
ASSAULT_INTERVAL: float     = 0.45
ASSAULT_SPEED_MUL: float    = 1.2
ASSAULT_DAMAGE: float       = 13.0
ASSAULT_BULLET_SPEED: float = 400.0

MOB_RADIUS: float        = 12.0
MOB_BASE_HP: float       = 30.0
MOB_BASE_SPEED: float    = 42.0
MOB_SHOOT_CD_MIN: float  = 0.8
MOB_SHOOT_CD_MAX: float  = 1.8
MOB_BULLET_SPEED: float  = 170.0
MOB_BULLET_DAMAGE: float = 5.0
MOB_SHOOT_CD_RESET: float = 1.6

BOSS_RADIUS: float        = 20.0
BOSS_BASE_HP: float       = 200.0
BOSS_BASE_SPEED: float    = 32.0
BOSS_BULLET_SPEED: float  = 200.0
BOSS_BULLET_DAMAGE: float = 9.0
BOSS_SHOOT_CD_RESET: float = 1.2

BULLET_RADIUS: float = 4.0
BULLET_TTL: float    = 2.2
ALLY_BULLET_SPEED: float = 400.0

DT: float       = 1.0 / 60.0
MAX_STEPS: int  = 60 * 40
FLOOR_RANGE     = (1, 6)

OBSTACLE_LAYOUTS: list[list[dict]] = [
    # 布局 0：中央横墙 + 两侧竖柱 + 斜角掩体
    [
        {"x": 360, "y": 255, "w": 180, "h": 22},
        {"x": 240, "y": 170, "w": 22,  "h": 130},
        {"x": 660, "y": 220, "w": 22,  "h": 130},
        {"x": 480, "y": 140, "w": 140, "h": 20},
        {"x": 420, "y": 360, "w": 140, "h": 20},
        {"x": 300, "y": 360, "w": 80,  "h": 20},
    ],
    # 布局 1：走廊型（上下各一道长墙，中间留缺口）
    [
        {"x": 280, "y": 145, "w": 200, "h": 20},
        {"x": 560, "y": 145, "w": 160, "h": 20},
        {"x": 280, "y": 375, "w": 160, "h": 20},
        {"x": 520, "y": 375, "w": 200, "h": 20},
        {"x": 235, "y": 220, "w": 20,  "h": 110},
        {"x": 660, "y": 210, "w": 20,  "h": 110},
        {"x": 410, "y": 245, "w": 100, "h": 20},
    ],
    # 布局 2：分散长条（斜向交错）
    [
        {"x": 270, "y": 160, "w": 160, "h": 20},
        {"x": 580, "y": 200, "w": 20,  "h": 150},
        {"x": 340, "y": 340, "w": 160, "h": 20},
        {"x": 630, "y": 330, "w": 140, "h": 20},
        {"x": 240, "y": 280, "w": 20,  "h": 100},
        {"x": 450, "y": 150, "w": 20,  "h": 120},
    ],
    # 布局 3：十字形 + 外围长条
    [
        {"x": 390, "y": 230, "w": 120, "h": 20},
        {"x": 445, "y": 165, "w": 20,  "h": 140},
        {"x": 240, "y": 155, "w": 130, "h": 20},
        {"x": 620, "y": 155, "w": 130, "h": 20},
        {"x": 240, "y": 365, "w": 130, "h": 20},
        {"x": 620, "y": 365, "w": 130, "h": 20},
    ],
]

MOB_BASE_POSITIONS = [
    (520, 100), (650, 150), (780, 100),
    (560, 380), (700, 430), (820, 360),
    (700, 270), (820, 200), (760, 430),
    (850, 130),
]

_SQ2 = math.sqrt(2) / 2
ACTION_VECTORS: list[tuple[float, float]] = [
    (0.0,   0.0),    # 0 静止
    (0.0,  -1.0),    # 1 上
    ( _SQ2, -_SQ2),  # 2 右上
    (1.0,   0.0),    # 3 右
    ( _SQ2,  _SQ2),  # 4 右下
    (0.0,   1.0),    # 5 下
    (-_SQ2,  _SQ2),  # 6 左下
    (-1.0,  0.0),    # 7 左
    (-_SQ2, -_SQ2),  # 8 左上
]
N_ACTIONS = len(ACTION_VECTORS)  # 9

# ── OBS 维度 ──────────────────────────────────────────────────────────────────
#
# 段1  自身状态              7  (+1 hp比例)
# 段2  主目标敌人           8  (+1 LOS标志, +1 预测命中距离)
# 段3  其余最多4个敌人      4×5 = 20
# 段4  最多8颗威胁子弹      8×6 = 48  (+1 预测碰撞时间/TTL归一化)
# 段5  最多4个障碍物        4×6 = 24
# 段6  到四壁距离            4
# 段7  上帧动作 one-hot      9
# 段8  到目标距离变化量      1  (归一化，>0=靠近，<0=远离)
# ─────────────────────────────────────────────────────
# 合计                      121

MAX_ENEMIES   = 5
MAX_BULLETS   = 8
MAX_OBSTACLES = 7  # 新布局最多 7 个障碍物

_SEG1 = 7
_SEG2 = 8
_SEG3 = (MAX_ENEMIES - 1) * 5   # 20
_SEG4 = MAX_BULLETS * 6          # 48
_SEG5 = MAX_OBSTACLES * 6        # 42
_SEG6 = 4
_SEG7 = N_ACTIONS                # 9
_SEG8 = 1
OBS_DIM = _SEG1 + _SEG2 + _SEG3 + _SEG4 + _SEG5 + _SEG6 + _SEG7 + _SEG8  # 139

MAX_BULLET_DIST = 200.0  # 超出此范围的子弹视为无威胁


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
    kind: str      = "mob"
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


def _has_line_of_sight(ax: float, ay: float,
                        bx: float, by: float,
                        obstacles: list[dict],
                        steps: int = 16) -> bool:
    """
    射线步进检测两点间是否有障碍物遮挡。
    steps=16 在精度和性能间取得平衡（每帧调用一次，约 16 次 AABB 检测）。
    返回 True 表示视线通畅，False 表示被障碍物遮挡。
    """
    for i in range(1, steps):
        t = i / steps
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        for o in obstacles:
            if o["x"] <= px <= o["x"] + o["w"] and o["y"] <= py <= o["y"] + o["h"]:
                return False
    return True


def _bullet_time_to_ally(b: Bullet, ally: Entity) -> float:
    """
    估算子弹到达 ally 的时间（秒）。
    用子弹速度方向的投影距离计算，负值表示子弹已飞过或背向。
    返回归一化到 [0,1] 的值：= max(0, tta) / BULLET_TTL。
    """
    dx = ally.x - b.x
    dy = ally.y - b.y
    spd = math.hypot(b.vx, b.vy)
    if spd < 1e-6:
        return 1.0
    # 子弹速度方向上的投影距离
    proj = (dx * b.vx + dy * b.vy) / spd
    if proj <= 0:
        return 1.0  # 子弹背向 ally
    tta = proj / spd
    return min(1.0, max(0.0, tta / BULLET_TTL))


# ── 环境主体 ──────────────────────────────────────────────────────────────────

class AssaultEnv(gym.Env):
    """
    观测空间（121 维 float32）：
        [0:7]    自身状态
        [7:15]   主目标敌人（含 LOS 标志）
        [15:35]  其余最多 4 个敌人
        [35:83]  最多 8 颗威胁子弹（含预测碰撞时间）
        [83:107] 最多 4 个障碍物
        [107:111] 到四壁距离
        [111:120] 上帧动作 one-hot
        [120]    到目标距离变化量

    动作空间：Discrete(9)，0=静止，1-8=8方向移动
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._ally: Entity = None          # type: ignore[assignment]
        self._enemies: list[Enemy] = []
        self._ally_bullets: list[Bullet] = []
        self._enemy_bullets: list[Bullet] = []
        self._obstacles: list[dict] = []
        self._attack_cd: float = 0.0
        self._step_count: int = 0
        self._floor: int = 1
        self._hp_mul: float = 1.0
        self._speed_mul: float = 1.0
        self._prev_action: int = 0
        self._prev_dist_to_target: float = 0.0
        self._prev_ally_pos: tuple[float, float] = (0.0, 0.0)
        self._still_frames: int = 0        # 连续静止帧计数（方案四）

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        floor = random.randint(*FLOOR_RANGE)
        self._floor = floor
        f = floor - 1
        self._hp_mul   = 1.0 + f * 0.30
        self._speed_mul = 1.0 + f * 0.06
        mob_count = min(3 + int(f * 1.2), 10)

        self._obstacles = OBSTACLE_LAYOUTS[(floor - 1) % len(OBSTACLE_LAYOUTS)]

        self._ally = Entity(
            x=random.uniform(160, 240),
            y=random.uniform(220, 320),
            radius=ALLY_RADIUS,
            hp=ALLY_MAX_HP,
            max_hp=ALLY_MAX_HP,
            speed=ALLY_BASE_SPEED,
        )

        self._enemies = []
        for i in range(mob_count):
            bx, by = MOB_BASE_POSITIONS[i % len(MOB_BASE_POSITIONS)]
            ex, ey = self._safe_spawn(bx, by, MOB_RADIUS, fallback_x=750.0, fallback_y=270.0)
            self._enemies.append(Enemy(
                x=ex, y=ey,
                radius=MOB_RADIUS,
                hp=round(MOB_BASE_HP * self._hp_mul),
                max_hp=round(MOB_BASE_HP * self._hp_mul),
                speed=MOB_BASE_SPEED * self._speed_mul,
                kind="mob",
                shoot_cd=random.uniform(MOB_SHOOT_CD_MIN, MOB_SHOOT_CD_MAX),
            ))
        bx, by = self._safe_spawn(820.0, 270.0, BOSS_RADIUS, fallback_x=830.0, fallback_y=400.0)
        self._enemies.append(Enemy(
            x=bx, y=by,
            radius=BOSS_RADIUS,
            hp=round(BOSS_BASE_HP * self._hp_mul),
            max_hp=round(BOSS_BASE_HP * self._hp_mul),
            speed=BOSS_BASE_SPEED * self._speed_mul,
            kind="boss",
            shoot_cd=0.8,
        ))

        self._ally_bullets  = []
        self._enemy_bullets = []
        self._attack_cd     = 0.0
        self._step_count    = 0
        self._prev_action   = 0
        self._still_frames  = 0
        self._prev_ally_pos = (self._ally.x, self._ally.y)

        target = self._nearest_enemy()
        self._prev_dist_to_target = self._ally.dist(target) if target else 0.0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ally is not None, "call reset() first"
        self._step_count += 1

        ally_hp_before = self._ally.hp
        target_before  = self._nearest_enemy()
        dist_before    = self._ally.dist(target_before) if target_before else 0.0

        # 1. ally 移动
        speed = self._ally.speed * ASSAULT_SPEED_MUL
        mx, my = ACTION_VECTORS[action]
        _move_with_collision(self._ally, mx * speed * DT, my * speed * DT,
                             self._obstacles)

        # 2. ally 攻击
        self._attack_cd = max(0.0, self._attack_cd - DT)
        target = self._nearest_enemy()
        fired_this_step = False
        if target is not None and self._attack_cd <= 0.0:
            self._ally_bullets.append(
                _create_bullet(self._ally.x, self._ally.y,
                               target.x, target.y,
                               ALLY_BULLET_SPEED, ASSAULT_DAMAGE)
            )
            self._attack_cd = ASSAULT_INTERVAL
            fired_this_step = True

        # 3. 敌人移动 + 开火
        for enemy in self._enemies:
            enemy.shoot_cd = max(0.0, enemy.shoot_cd - DT)
            nx, ny = _normalize(self._ally.x - enemy.x, self._ally.y - enemy.y)
            _move_with_collision(enemy, nx * enemy.speed * DT, ny * enemy.speed * DT,
                                 self._obstacles)
            if enemy.shoot_cd <= 0.0:
                spd = BOSS_BULLET_SPEED  if enemy.kind == "boss" else MOB_BULLET_SPEED
                dmg = BOSS_BULLET_DAMAGE if enemy.kind == "boss" else MOB_BULLET_DAMAGE
                cd  = BOSS_SHOOT_CD_RESET if enemy.kind == "boss" else MOB_SHOOT_CD_RESET
                self._enemy_bullets.append(
                    _create_bullet(enemy.x, enemy.y,
                                   self._ally.x, self._ally.y, spd, dmg)
                )
                enemy.shoot_cd = cd

        # 4. 子弹物理
        damage_dealt = self._update_bullets()

        # 5. 清除死亡敌人
        self._enemies = [e for e in self._enemies if e.hp > 0]

        # 6. 判断本步开火是否有 LOS（用开火前的目标位置）
        fired_without_los = False
        if fired_this_step and target_before is not None:
            los = _has_line_of_sight(
                self._ally.x, self._ally.y,
                target_before.x, target_before.y,
                self._obstacles,
            )
            fired_without_los = not los

        # 7. 计算当前帧到目标的距离变化量
        target_now = self._nearest_enemy()
        dist_now   = self._ally.dist(target_now) if target_now else dist_before
        dist_delta = dist_before - dist_now   # >0 表示靠近，<0 表示远离

        # 8. 奖励
        hp_lost = max(0.0, ally_hp_before - self._ally.hp)
        reward = self._compute_reward(
            damage_dealt, hp_lost, target_now,
            action, dist_now, dist_delta, fired_without_los,
            self._still_frames,
        )

        # 9. 更新帧间状态
        # 静止检测：位移 < 0.5px 视为静止
        moved = math.hypot(
            self._ally.x - self._prev_ally_pos[0],
            self._ally.y - self._prev_ally_pos[1],
        )
        if moved < 0.5:
            self._still_frames += 1
        else:
            self._still_frames = 0
        self._prev_ally_pos       = (self._ally.x, self._ally.y)
        self._prev_action         = action
        self._prev_dist_to_target = dist_now

        terminated = self._ally.hp <= 0 or len(self._enemies) == 0
        truncated  = self._step_count >= MAX_STEPS

        return self._get_obs(), reward, terminated, truncated, {
            "damage_dealt": damage_dealt,
            "hp_lost": hp_lost,
            "enemies_left": len(self._enemies),
        }

    # ── 观测构建 ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        构建 121 维归一化观测向量。

        归一化约定：
          位置            / CANVAS_W|H           → [0,1]
          相对位置 dx/dy  / CANVAS_W|H，有符号   → [-1,1]
          距离            / DIAG(≈1051)           → [0,1]
          hp              / max_hp                → [0,1]
          冷却            / 最大冷却时间           → [0,1]
          速度方向        单位向量                → [-1,1]
          one-hot         0/1
        """
        DIAG = math.hypot(CANVAS_W, CANVAS_H)
        obs  = np.zeros(OBS_DIM, dtype=np.float32)
        idx  = 0
        ally = self._ally
        target = self._nearest_enemy()

        # ── 段1：自身状态 (7维) ───────────────────────────────────────────────
        # x, y, hp比例, 到最近障碍物距离, 攻击冷却比例, 敌人数量归一化, 速度归一化
        obs[idx]   = ally.x / CANVAS_W
        obs[idx+1] = ally.y / CANVAS_H
        obs[idx+2] = ally.hp / ally.max_hp
        obs[idx+3] = self._dist_to_nearest_obstacle() / DIAG
        obs[idx+4] = min(1.0, self._attack_cd / ASSAULT_INTERVAL)
        obs[idx+5] = len(self._enemies) / 11.0
        # 当前 hp 损失速率感知：最近帧受伤程度（用 1 - hp比例 代替，越低越危险）
        obs[idx+6] = 1.0 - (ally.hp / ally.max_hp)
        idx += _SEG1

        # ── 段2：主目标敌人（最近，8维）──────────────────────────────────────
        # dx, dy, 距离, hp比例, is_boss, 射击冷却比例, LOS标志, 到目标的有效射程标志
        if target is not None:
            dx   = target.x - ally.x
            dy   = target.y - ally.y
            dist = math.hypot(dx, dy)
            shoot_cd_max = BOSS_SHOOT_CD_RESET if target.kind == "boss" else MOB_SHOOT_CD_RESET
            los  = _has_line_of_sight(ally.x, ally.y, target.x, target.y, self._obstacles)
            # 有效射程标志：在 [kite_range, attack_range] 内为 1.0，线性衰减
            in_range = 1.0 if ASSAULT_KITE_RANGE < dist < ASSAULT_ATTACK_RANGE else 0.0
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = dist / DIAG
            obs[idx+3] = target.hp / target.max_hp
            obs[idx+4] = 1.0 if target.kind == "boss" else 0.0
            obs[idx+5] = target.shoot_cd / shoot_cd_max
            obs[idx+6] = 1.0 if los else 0.0   # LOS 标志：1=视线通畅
            obs[idx+7] = in_range
        idx += _SEG2

        # ── 段3：其余最多 4 个敌人（5维/敌）────────────────────────────────────
        others = sorted(
            [e for e in self._enemies if e is not target],
            key=lambda e: math.hypot(e.x - ally.x, e.y - ally.y)
        )[: MAX_ENEMIES - 1]
        for e in others:
            dx   = e.x - ally.x
            dy   = e.y - ally.y
            dist = math.hypot(dx, dy)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = dist / DIAG
            obs[idx+3] = e.hp / e.max_hp
            obs[idx+4] = 1.0 if e.kind == "boss" else 0.0
            idx += 5
        idx += (MAX_ENEMIES - 1 - len(others)) * 5

        # ── 段4：最多 8 颗威胁子弹（6维/颗）─────────────────────────────────
        # dx, dy, 速度方向vx, 速度方向vy, 距离归一化, 预测碰撞时间归一化
        # 排序：优先按预测碰撞时间（越短越危险）
        threat_bullets = [
            b for b in self._enemy_bullets
            if math.hypot(b.x - ally.x, b.y - ally.y) < MAX_BULLET_DIST
        ]
        threat_bullets.sort(
            key=lambda b: _bullet_time_to_ally(b, ally)
        )
        threat_bullets = threat_bullets[: MAX_BULLETS]
        for b in threat_bullets:
            dx   = b.x - ally.x
            dy   = b.y - ally.y
            dist = math.hypot(dx, dy)
            bspd = math.hypot(b.vx, b.vy) or 1.0
            tta  = _bullet_time_to_ally(b, ally)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = b.vx / bspd
            obs[idx+3] = b.vy / bspd
            obs[idx+4] = dist / MAX_BULLET_DIST
            obs[idx+5] = tta                    # 预测碰撞时间（0=即将命中）
            idx += 6
        idx += (MAX_BULLETS - len(threat_bullets)) * 6

        # ── 段5：障碍物（6维/块，pad 到 MAX_OBSTACLES）───────────────────────
        sorted_obs = sorted(
            self._obstacles,
            key=lambda o: math.hypot(
                (o["x"] + o["w"] / 2) - ally.x,
                (o["y"] + o["h"] / 2) - ally.y,
            )
        )[: MAX_OBSTACLES]
        for o in sorted_obs:
            cx = o["x"] + o["w"] / 2
            cy = o["y"] + o["h"] / 2
            dx = cx - ally.x
            dy = cy - ally.y
            near_x    = max(o["x"], min(ally.x, o["x"] + o["w"]))
            near_y    = max(o["y"], min(ally.y, o["y"] + o["h"]))
            near_dist = math.hypot(ally.x - near_x, ally.y - near_y)
            cdx = dx / (math.hypot(dx, dy) or 1.0)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = o["w"] / CANVAS_W
            obs[idx+3] = o["h"] / CANVAS_H
            obs[idx+4] = near_dist / DIAG
            obs[idx+5] = cdx
            idx += 6
        idx += (MAX_OBSTACLES - len(sorted_obs)) * 6

        # ── 段6：到四壁距离（4维）────────────────────────────────────────────
        obs[idx]   = ally.y / CANVAS_H
        obs[idx+1] = (CANVAS_H - ally.y) / CANVAS_H
        obs[idx+2] = ally.x / CANVAS_W
        obs[idx+3] = (CANVAS_W - ally.x) / CANVAS_W
        idx += _SEG6

        # ── 段7：上帧动作 one-hot（9维）──────────────────────────────────────
        obs[idx + self._prev_action] = 1.0
        idx += _SEG7

        # ── 段8：到目标距离变化量（1维）──────────────────────────────────────
        # 归一化：变化量 / CANVAS_W，靠近为正，远离为负
        dist_delta = self._prev_dist_to_target - (
            self._ally.dist(target) if target is not None else self._prev_dist_to_target
        )
        obs[idx] = max(-1.0, min(1.0, dist_delta / CANVAS_W * 10.0))
        idx += _SEG8

        assert idx == OBS_DIM, f"obs dim mismatch: {idx} != {OBS_DIM}"
        return obs

    # ── 奖励函数 ──────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        damage_dealt: float,
        hp_lost: float,
        target: Enemy | None,
        action: int,
        dist_now: float,
        dist_delta: float,
        fired_without_los: bool,
        still_frames: int,
    ) -> float:
        """
        v4 奖励函数：

        1. 伤害奖励
        2. 受伤惩罚
        3. 势能场接近奖励（方案一）：用距离本身持续惩罚，绕路时不受干扰
        4. 距离+LOS 联合区间奖励
           - 过近（<kiteRange）：加重惩罚（方案C）
           - 射程内 LOS 通畅：最高正奖励
           - 射程内 LOS 遮挡：持续帧惩罚
           - 稍远（110~120px）LOS 通畅：小正奖励
           - 过远（>120px，方案三）：随距离加重惩罚
        5. 动作平滑惩罚
        6. LOS 遮挡开火惩罚
        7. 连续静止惩罚（方案四）：>45 帧起累积加重
        8. 死亡惩罚
        9. 胜利奖励
        10. 时间惩罚
        """
        DIAG  = math.hypot(CANVAS_W, CANVAS_H)
        reward = 0.0
        ally   = self._ally

        # 1. 伤害奖励
        reward += damage_dealt * 0.08

        # 2. 受伤惩罚
        reward -= hp_lost * 0.12

        # 3. 势能场接近奖励（方案一）
        # 用当前距离本身做持续惩罚，不依赖 delta，绕路时仍有稳定的"靠近"引力
        # 距离 500px → -0.007/帧；距离 100px → -0.0014/帧
        if target is not None:
            reward -= (dist_now / DIAG) * 0.015

        # 4. 距离 + LOS 联合区间奖励
        if target is not None:
            los = _has_line_of_sight(
                ally.x, ally.y, target.x, target.y, self._obstacles
            )

            if dist_now < ASSAULT_KITE_RANGE:
                # 过近：加重惩罚（方案C，从 -0.004 → -0.012）
                reward -= 0.012

            elif ASSAULT_KITE_RANGE <= dist_now < ASSAULT_ATTACK_RANGE:
                if los:
                    reward += 0.010   # 最优攻击位
                else:
                    reward -= 0.008   # LOS 遮挡 = 每帧亏钱

            elif ASSAULT_ATTACK_RANGE <= dist_now < 120.0:
                # 方案三：阈值从 150 收紧到 120
                if los:
                    reward += 0.003
                # LOS 遮挡时不给正奖励（保持中性）

            else:
                # 过远（>120px），方案三：惩罚系数从 0.02 → 0.04
                excess = (dist_now - 120.0) / CANVAS_W
                reward -= 0.008 + excess * 0.04

        # 5. 动作平滑惩罚
        prev_vec = ACTION_VECTORS[self._prev_action]
        curr_vec = ACTION_VECTORS[action]
        dot = prev_vec[0] * curr_vec[0] + prev_vec[1] * curr_vec[1]
        if action != 0 and self._prev_action != 0 and dot < -0.5:
            reward -= 0.005

        # 6. LOS 遮挡开火惩罚
        if fired_without_los:
            reward -= 0.04

        # 7. 连续静止惩罚（方案四）
        # 连续 45 帧（0.75s）不动开始惩罚，每帧 -0.008，随时间线性加重到 -0.020
        if still_frames > 45:
            # 超出 45 帧后每帧从 -0.008 线性增大，最高 -0.020
            overage = min(still_frames - 45, 75)   # 最多再加 75 帧的加重量
            penalty = 0.008 + overage * (0.012 / 75)
            reward -= penalty

        # 8. 死亡惩罚
        if ally.hp <= 0:
            reward -= 30.0

        # 9. 胜利奖励
        if len(self._enemies) == 0:
            reward += 50.0

        # 10. 时间惩罚
        reward -= 0.001

        return float(reward)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _safe_spawn(self, base_x: float, base_y: float,
                    radius: float,
                    fallback_x: float, fallback_y: float,
                    max_tries: int = 20,
                    jitter: float = 20.0) -> tuple[float, float]:
        """
        在 base 附近随机采样一个不与障碍物碰撞的坐标。
        最多尝试 max_tries 次，失败后返回 fallback 坐标。
        """
        for _ in range(max_tries):
            x = base_x + random.uniform(-jitter, jitter)
            y = base_y + random.uniform(-jitter, jitter)
            x = max(radius, min(x, CANVAS_W - radius))
            y = max(radius, min(y, CANVAS_H - radius))
            if not _collides_with_obstacle(x, y, radius, self._obstacles):
                return x, y
        return fallback_x, fallback_y

    def _nearest_enemy(self) -> Enemy | None:
        if not self._enemies:
            return None
        return min(self._enemies, key=lambda e: self._ally.dist(e))

    def _dist_to_nearest_obstacle(self) -> float:
        if not self._obstacles:
            return float(math.hypot(CANVAS_W, CANVAS_H))
        ally    = self._ally
        min_dist = float("inf")
        for o in self._obstacles:
            near_x = max(o["x"], min(ally.x, o["x"] + o["w"]))
            near_y = max(o["y"], min(ally.y, o["y"] + o["h"]))
            d = math.hypot(ally.x - near_x, ally.y - near_y)
            min_dist = min(min_dist, d)
        return min_dist

    def _update_bullets(self) -> float:
        damage_dealt = 0.0

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
