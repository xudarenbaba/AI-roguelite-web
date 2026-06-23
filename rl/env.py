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

# ── OBS 维度（v9：射线检测 + 历史最小距离，配合 LSTM）─────────────────────────
#
# 段1  自身状态              4   (x, y, 攻击冷却, 敌人数量)
# 段2  主目标敌人            8   (dx, dy, 距离, hp, is_boss, 射击CD, LOS, 射程标志)
# 段3  其余最多4个敌人       4×5 = 20
# 段4  最多8颗威胁子弹       8×6 = 48
# 段5  射线检测              N_RAYS = 16  (替换原矩形障碍物 obs，泛化性更强)
# 段6  到四壁距离            4
# 段7  上帧动作 one-hot      9
# 段8  到目标距离变化量      1
# 段9  最危险子弹 TTA        1
# 段10 历史最小距离          1   (episode 内到目标的历史最近距离，配合 LSTM)
# ─────────────────────────────────────────────────────
# 合计                      112

MAX_ENEMIES   = 5
MAX_BULLETS   = 8

# 射线检测：从 ally 向 16 个均匀方向发射，返回到障碍物的归一化距离
N_RAYS       = 16
RAY_MAX_DIST = 260.0   # 射线最大检测距离，超出视为无障碍（归一化为 1.0）
RAY_STEPS    = 26      # 每条射线步进采样次数（步长 = RAY_MAX_DIST / RAY_STEPS ≈ 10px）

_SEG1 = 4
_SEG2 = 8
_SEG3 = (MAX_ENEMIES - 1) * 5   # 20
_SEG4 = MAX_BULLETS * 6          # 48
_SEG5 = N_RAYS                   # 16（射线检测）
_SEG6 = 4
_SEG7 = N_ACTIONS                # 9
_SEG8 = 1
_SEG9 = 1                        # 最危险子弹 TTA
_SEG10 = 1                       # 历史最小距离（非马尔可夫信号，由 LSTM 利用）
OBS_DIM = _SEG1 + _SEG2 + _SEG3 + _SEG4 + _SEG5 + _SEG6 + _SEG7 + _SEG8 + _SEG9 + _SEG10  # 112

# 16 方向射线的单位向量（每 22.5°）
_RAY_DIRS = [
    (math.cos(2 * math.pi * i / N_RAYS), math.sin(2 * math.pi * i / N_RAYS))
    for i in range(N_RAYS)
]

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


def _raycast_obstacle(ox: float, oy: float,
                      dx: float, dy: float,
                      obstacles: list[dict],
                      max_dist: float = RAY_MAX_DIST,
                      steps: int = RAY_STEPS) -> float:
    """
    从 (ox, oy) 沿单位方向 (dx, dy) 步进，检测第一个障碍物碰撞点的距离。
    返回归一化距离 [0, 1]：0 = 紧贴障碍物，1 = max_dist 内无障碍。
    也把画布边界当作障碍（射线打到墙壁也返回距离）。
    """
    step_len = max_dist / steps
    for i in range(1, steps + 1):
        d = i * step_len
        px = ox + dx * d
        py = oy + dy * d
        # 画布边界
        if px < 0 or px > CANVAS_W or py < 0 or py > CANVAS_H:
            return min(1.0, d / max_dist)
        # 障碍物 AABB
        for o in obstacles:
            if o["x"] <= px <= o["x"] + o["w"] and o["y"] <= py <= o["y"] + o["h"]:
                return d / max_dist
    return 1.0


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
    观测空间：112 维（射线检测 + 历史最小距离，配合 RecurrentPPO 的 LSTM）
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
        self._still_frames: int = 0
        self._prev_los: bool = False
        self._scraping_frames: int = 0     # 连续蹭墙帧计数
        self._curriculum_ratio: float = 0.0  # 0.0~1.0，由 train.py callback 更新
        self._min_dist_so_far: float = float("inf")  # episode 内到目标的历史最近距离

    def set_curriculum_ratio(self, ratio: float) -> None:
        """由训练 callback 调用，更新课程学习进度（0.0 = 起点，1.0 = 终点）。"""
        self._curriculum_ratio = max(0.0, min(1.0, ratio))

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        stage = self._curriculum_stage()

        floor = random.randint(*FLOOR_RANGE)
        self._floor = floor
        f = floor - 1
        self._hp_mul   = 1.0 + f * 0.30
        self._speed_mul = 1.0 + f * 0.06

        # 阶段0：只放少量 mob，先学找人；后续阶段恢复正常数量
        if stage == 0:
            mob_count = 2
        elif stage == 1:
            mob_count = 3
        else:
            mob_count = min(3 + int(f * 1.2), 10)

        self._obstacles = self._curriculum_obstacles()

        self._ally = Entity(
            x=random.uniform(160, 240),
            y=random.uniform(220, 320),
            radius=ALLY_RADIUS,
            hp=ALLY_MAX_HP,
            max_hp=ALLY_MAX_HP,
            speed=ALLY_BASE_SPEED,
        )

        # 敌人生成距离课程：早期阶段敌人靠近 ally，后期全图随机
        # stage 0: 敌人在 ally 附近 150px；stage 1: 300px 内；stage 2+: 全图
        self._enemies = []
        for i in range(mob_count):
            if stage == 0:
                bx, by = self._spawn_near_ally(150.0)
            elif stage == 1:
                bx, by = self._spawn_near_ally(300.0)
            else:
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
        # boss 只在阶段 2+ 出现（早期阶段专注学基础导航）
        if stage >= 2:
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
        self._scraping_frames = 0
        self._prev_ally_pos = (self._ally.x, self._ally.y)

        target = self._nearest_enemy()
        self._prev_dist_to_target = self._ally.dist(target) if target else 0.0
        self._min_dist_so_far = self._prev_dist_to_target if target else float("inf")
        self._prev_los = (
            _has_line_of_sight(
                self._ally.x, self._ally.y, target.x, target.y, self._obstacles
            ) if target else False
        )

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

        # 8. 计算本帧 LOS 状态，检测是否刚刚绕路成功（上帧遮挡→本帧通畅）
        los_now = (
            _has_line_of_sight(
                self._ally.x, self._ally.y,
                target_now.x, target_now.y,
                self._obstacles,
            ) if target_now else False
        )
        los_improved = (not self._prev_los) and los_now   # 刚从遮挡变为通畅

        # 9. 进度奖励（历史最小距离）：只有创下 episode 内历史新低才给奖励
        # 这是 IJCAI 论文的核心设计——绕路时距离暂时拉远不惩罚，
        # 只奖励"真正的前进"（突破历史最近距离），彻底消除绕路的结构性惩罚。
        progress = 0.0
        if target_now is not None:
            if dist_now < self._min_dist_so_far:
                progress = self._min_dist_so_far - dist_now  # 创新低的幅度
                self._min_dist_so_far = dist_now

        # 9b. 奖励
        hp_lost = max(0.0, ally_hp_before - self._ally.hp)
        ally_ref = self._ally
        min_bullet_tta = (
            min((_bullet_time_to_ally(b, ally_ref) for b in self._enemy_bullets), default=1.0)
            if self._enemy_bullets else 1.0
        )
        reward = self._compute_reward(
            damage_dealt, hp_lost, target_now,
            action, dist_now, dist_delta, fired_without_los,
            self._still_frames, min_bullet_tta, los_improved,
            self._scraping_frames, progress,
        )

        # 10. 更新帧间状态
        if action == 0:
            self._still_frames += 1
        else:
            self._still_frames = 0

        # 蹭墙检测：选了非静止动作，但实际移动方向与期望方向严重偏差
        # 说明被墙卡住、在做分量滑动（蹭墙）
        actual_dx = self._ally.x - self._prev_ally_pos[0]
        actual_dy = self._ally.y - self._prev_ally_pos[1]
        actual_len = math.hypot(actual_dx, actual_dy)
        if action != 0 and actual_len > 1e-4:
            expected_dx, expected_dy = ACTION_VECTORS[action]
            # 实际移动方向与期望方向的点积
            dot_scrape = actual_dx / actual_len * expected_dx + actual_dy / actual_len * expected_dy
            if dot_scrape < 0.3:   # 方向偏差超过 ~72°，认为在蹭墙
                self._scraping_frames += 1
            else:
                self._scraping_frames = 0
        else:
            self._scraping_frames = 0

        self._prev_los            = los_now
        self._prev_ally_pos       = (self._ally.x, self._ally.y)
        self._prev_action         = action
        self._prev_dist_to_target = dist_now

        terminated = self._ally.hp <= 0 or len(self._enemies) == 0
        truncated  = self._step_count >= MAX_STEPS

        # 超时巨额惩罚：耗尽步数仍未清空敌人，给一次性大惩罚
        # 打击"兜圈磨时间"行为，强迫 agent 在时限内完成击杀
        if truncated and len(self._enemies) > 0:
            reward -= 25.0

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

        # ── 段1：自身状态 (4维) ───────────────────────────────────────────────
        # 去掉 hp 和"到最近障碍物距离"：
        #   - hp 不感知 → 低血时不保守
        #   - 障碍物距离 → 会让 agent 隐式学到"远离障碍物=安全"，导致不敢进入中间区域
        #   障碍物的位置信息在段5里已有完整描述，这里不需要冗余的距离标量
        obs[idx]   = ally.x / CANVAS_W
        obs[idx+1] = ally.y / CANVAS_H
        obs[idx+2] = min(1.0, self._attack_cd / ASSAULT_INTERVAL)
        obs[idx+3] = len(self._enemies) / 11.0
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

        # ── 段5：射线检测（16 方向，替换矩形障碍物 obs）─────────────────────
        # 从 ally 向 16 个均匀方向发射射线，返回到障碍物/边界的归一化距离
        # 0 = 紧贴障碍，1 = 该方向 RAY_MAX_DIST 内无障碍
        for rdx, rdy in _RAY_DIRS:
            obs[idx] = _raycast_obstacle(ally.x, ally.y, rdx, rdy, self._obstacles)
            idx += 1

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
        dist_delta = self._prev_dist_to_target - (
            self._ally.dist(target) if target is not None else self._prev_dist_to_target
        )
        obs[idx] = max(-1.0, min(1.0, dist_delta / CANVAS_W * 10.0))
        idx += _SEG8

        # ── 段9：最危险子弹 TTA（1维）────────────────────────────────────────
        if threat_bullets:
            min_tta = min(_bullet_time_to_ally(b, ally) for b in threat_bullets)
        else:
            min_tta = 1.0
        obs[idx] = min_tta
        idx += _SEG9

        # ── 段10：历史最小距离（1维）─────────────────────────────────────────
        # episode 内到目标的历史最近距离（归一化）。配合 LSTM 实现"进度奖励"，
        # 让 agent 感知"我离目标最近时有多近"，绕路时不会因为暂时拉远而迷失。
        if target is not None and self._min_dist_so_far != float("inf"):
            obs[idx] = min(1.0, self._min_dist_so_far / DIAG)
        else:
            obs[idx] = 1.0
        idx += _SEG10

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
        min_bullet_tta: float,
        los_improved: bool,
        scraping_frames: int,
        progress: float,
    ) -> float:
        """
        v9 奖励函数（配合 LSTM + 射线检测）：

        核心改动（解决隔墙问题）：
          - 用"历史最小距离进度奖励"替换势能场：只奖励真正前进，绕路不惩罚
          - 非对称远离惩罚：主动远离目标比靠近代价更高
          - LOS 奖励扩展到全距离：任何位置有视线都给奖励
          - 去掉角落惩罚：角落问题交给进度奖励处理（待角落不能创新低）

        1.  伤害奖励
        2.  受伤惩罚
        3.  进度奖励（历史最小距离创新低）—— 替换势能场
        4.  非对称远离惩罚
        5.  LOS 全距离奖励
        6.  距离区间奖励（射程内攻击位）
        7.  绕路成功奖励（LOS 遮挡→通畅）
        8.  移动方向奖励
        9.  子弹危险时主动移动奖励
        10. 动作平滑惩罚
        11. LOS 遮挡开火惩罚
        12. 连续静止惩罚（最优攻击位豁免）
        13. 蹭墙惩罚
        14. 死亡惩罚
        15. 胜利奖励
        16. 时间惩罚
        （超时巨额惩罚在 step() 末尾处理）
        """
        reward = 0.0
        ally   = self._ally

        # 1. 伤害奖励
        reward += damage_dealt * 0.08

        # 2. 受伤惩罚（修复：从 0.24 降回 0.12，避免绕路受伤风险压过靠近收益）
        reward -= hp_lost * 0.12

        # 3. 进度奖励：创下历史最近距离时按幅度奖励（奖励"真正前进"）
        reward += progress * 0.02

        # 3b. 势能场（修复：恢复持续靠近压力，与进度奖励共存）
        # 进度奖励只在"创新低"那一帧给，agent 冲到某距离后会失去动力；
        # 势能场每帧都按当前距离施压，确保 agent 持续有"再靠近"的拉力。
        # dist > 150px 强信号，≤150px 弱信号（近距离绕路时不过度干扰）。
        if target is not None:
            DIAG = math.hypot(CANVAS_W, CANVAS_H)
            if dist_now > 150.0:
                reward -= (dist_now / DIAG) * 0.020
            else:
                reward -= (dist_now / DIAG) * 0.008

        # 4. 非对称远离惩罚：主动远离目标时额外惩罚（远离代价 > 不动）
        if target is not None and dist_delta < 0:
            reward -= abs(dist_delta) * 0.008

        # 5. LOS 全距离奖励：任何距离视线通畅给小正奖励，遮挡给小负奖励
        in_optimal_pos = False
        if target is not None:
            los = _has_line_of_sight(
                ally.x, ally.y, target.x, target.y, self._obstacles
            )
            if los:
                reward += 0.002
            else:
                reward -= 0.002   # 修复：遮挡惩罚加倍（原 -0.001）

            # 6. 距离区间奖励（射程内攻击位）
            if dist_now < ASSAULT_KITE_RANGE:
                reward -= 0.010   # 过近
            elif ASSAULT_KITE_RANGE <= dist_now < ASSAULT_ATTACK_RANGE:
                if los:
                    reward += 0.020   # 修复：最优攻击位奖励提高（0.012→0.020）
                    in_optimal_pos = True
                else:
                    reward -= 0.015   # 修复：射程内遮挡=隔墙苟着，明确重罚（原来无惩罚）
            elif ASSAULT_ATTACK_RANGE <= dist_now < 130.0:
                if los:
                    reward += 0.006   # 稍远但可接受（0.004→0.006）
                else:
                    reward -= 0.008   # 修复：稍远遮挡也惩罚

        # 7. 绕路成功奖励：LOS 从遮挡变为通畅
        if los_improved:
            reward += 0.05

        # 8. 移动方向奖励
        if target is not None and action != 0:
            move_dx, move_dy = ACTION_VECTORS[action]
            to_dx = target.x - ally.x
            to_dy = target.y - ally.y
            to_len = math.hypot(to_dx, to_dy) or 1.0
            move_dot = move_dx * (to_dx / to_len) + move_dy * (to_dy / to_len)
            if move_dot > 0.5:
                reward += 0.002

        # 9. 子弹危险时主动移动奖励
        if min_bullet_tta < 0.2 and action != 0:
            reward += 0.003

        # 10. 动作平滑惩罚（阈值 -0.7，只惩罚近乎完全反向）
        prev_vec = ACTION_VECTORS[self._prev_action]
        curr_vec = ACTION_VECTORS[action]
        dot = prev_vec[0] * curr_vec[0] + prev_vec[1] * curr_vec[1]
        if action != 0 and self._prev_action != 0 and dot < -0.7:
            reward -= 0.005

        # 11. LOS 遮挡开火惩罚
        if fired_without_los:
            reward -= 0.04

        # 12. 连续静止惩罚（最优攻击位豁免）
        if still_frames > 45 and not in_optimal_pos:
            overage = min(still_frames - 45, 75)
            reward -= 0.008 + overage * (0.012 / 75)

        # 13. 蹭墙惩罚
        if scraping_frames > 10:
            overage = min(scraping_frames - 10, 50)
            reward -= 0.004 + overage * (0.004 / 50)

        # 14. 死亡惩罚
        if ally.hp <= 0:
            reward -= 30.0

        # 15. 胜利奖励
        if len(self._enemies) == 0:
            reward += 50.0

        # 16. 时间惩罚
        reward -= 0.003

        return float(reward)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _curriculum_stage(self) -> int:
        """
        根据训练进度返回当前课程阶段 0~4。

        5 阶段课程（距离 + 障碍物双维度）：
          阶段0（0.00~0.20）：无障碍，敌人在 ally 附近（先学"找人攻击"）
          阶段1（0.20~0.40）：1 块简单障碍，敌人 300px 内
          阶段2（0.40~0.60）：布局0（最简单），敌人全图随机
          阶段3（0.60~0.80）：布局0-1 随机，敌人全图随机
          阶段4（0.80~1.00）：全部布局随机，敌人全图随机
        """
        r = self._curriculum_ratio
        if r < 0.20:
            return 0
        if r < 0.40:
            return 1
        if r < 0.60:
            return 2
        if r < 0.80:
            return 3
        return 4

    def _curriculum_obstacles(self) -> list[dict]:
        """根据课程阶段返回障碍物布局。"""
        stage = self._curriculum_stage()
        if stage == 0:
            return []   # 无障碍
        if stage == 1:
            # 单块简单障碍（中央一道横墙）
            return [{"x": 380, "y": 255, "w": 140, "h": 22}]
        if stage == 2:
            return OBSTACLE_LAYOUTS[0]
        if stage == 3:
            return OBSTACLE_LAYOUTS[random.randint(0, 1)]
        return OBSTACLE_LAYOUTS[random.randint(0, len(OBSTACLE_LAYOUTS) - 1)]

    def _spawn_near_ally(self, max_radius: float) -> tuple[float, float]:
        """
        在 ally 周围 [80, max_radius] 距离的随机位置生成敌人基准坐标。
        用于课程学习早期阶段，让敌人靠近 ally 以稳定训练信号。
        """
        ally = self._ally
        for _ in range(20):
            ang = random.uniform(0, 2 * math.pi)
            r   = random.uniform(80.0, max_radius)
            x   = ally.x + math.cos(ang) * r
            y   = ally.y + math.sin(ang) * r
            x   = max(40.0, min(x, CANVAS_W - 40.0))
            y   = max(40.0, min(y, CANVAS_H - 40.0))
            return x, y
        return CANVAS_W * 0.6, CANVAS_H * 0.5

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
        """
        LOS 加权评分选取目标。
        评分 = 直线距离 × LOS惩罚系数（LOS通畅=1.0，遮挡=1.3）
        系数从 1.8 降到 1.3：轻度偏好有 LOS 的目标，
        但不过度放大遮挡目标的"有效距离"，避免 agent 因为全部目标都被遮挡时
        不知道该往哪走、倾向于在原地等待。
        """
        if not self._enemies:
            return None
        ally = self._ally
        best = None
        best_score = float("inf")
        for e in self._enemies:
            dist = ally.dist(e)
            los = _has_line_of_sight(ally.x, ally.y, e.x, e.y, self._obstacles)
            score = dist * (1.0 if los else 1.3)
            if score < best_score:
                best_score = score
                best = e
        return best

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
