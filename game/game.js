const NPC_API = "http://127.0.0.1:5100";

// ── RL 推理模块（assault 姿态）────────────────────────────────────────────────
// onnxruntime-web 从 CDN 加载；如需离线部署可改为本地路径。
// 模型文件 assault_policy.onnx 放在 game/ 目录下。
// 在模型加载完成前，assault 姿态降级为规则 AI。

const _RL_MODEL_PATH = "assault_policy.onnx";

// 与 rl/env.py ACTION_VECTORS 严格对齐（索引 0-8）
const _RL_ACTION_VECTORS = [
  [ 0.0,  0.0],          // 0 静止
  [ 0.0, -1.0],          // 1 上
  [ 0.7071, -0.7071],    // 2 右上
  [ 1.0,  0.0],          // 3 右
  [ 0.7071,  0.7071],    // 4 右下
  [ 0.0,  1.0],          // 5 下
  [-0.7071,  0.7071],    // 6 左下
  [-1.0,  0.0],          // 7 左
  [-0.7071, -0.7071],    // 8 左上
];

// 与 rl/env.py 常量对齐（v9：射线检测 + LSTM，OBS_DIM=112）
const _RL_CANVAS_W        = 900.0;
const _RL_CANVAS_H        = 540.0;
const _RL_DIAG            = Math.hypot(_RL_CANVAS_W, _RL_CANVAS_H);  // ~1051
const _RL_MAX_ENEMIES     = 5;
const _RL_MAX_BULLETS     = 8;
const _RL_MAX_BULLET_DIST = 200.0;
const _RL_ASSAULT_INTERVAL = 0.45;
const _RL_N_ACTIONS       = 9;

// 射线检测（与 env.py N_RAYS / RAY_MAX_DIST / RAY_STEPS 严格对齐）
const _RL_N_RAYS       = 16;
const _RL_RAY_MAX_DIST = 260.0;
const _RL_RAY_STEPS    = 26;
// 16 方向单位向量（每 22.5°）
const _RL_RAY_DIRS = [];
for (let i = 0; i < _RL_N_RAYS; i++) {
  const a = 2 * Math.PI * i / _RL_N_RAYS;
  _RL_RAY_DIRS.push([Math.cos(a), Math.sin(a)]);
}

// 段长与 env.py 严格对齐
const _RL_SEG1 = 4;
const _RL_SEG2 = 8;
const _RL_SEG3 = (_RL_MAX_ENEMIES - 1) * 5;   // 20
const _RL_SEG4 = _RL_MAX_BULLETS * 6;          // 48
const _RL_SEG5 = _RL_N_RAYS;                   // 16（射线检测）
const _RL_SEG6 = 4;
const _RL_SEG7 = _RL_N_ACTIONS;               // 9
const _RL_SEG8 = 1;
const _RL_SEG9 = 1;                            // 最危险子弹 TTA
const _RL_SEG10 = 1;                           // 历史最小距离
const _RL_OBS_DIM = _RL_SEG1 + _RL_SEG2 + _RL_SEG3 + _RL_SEG4 + _RL_SEG5 + _RL_SEG6 + _RL_SEG7 + _RL_SEG8 + _RL_SEG9 + _RL_SEG10; // 112

// LSTM hidden state 维度（与训练模型对齐，导出后从 meta json 也可读取）
const _RL_LSTM_HIDDEN = 128;
const _RL_LSTM_LAYERS = 1;

// 帧间状态
let _rlPrevAction = 0;
let _rlPrevDistToTarget = 0.0;
let _rlMinDistSoFar = Infinity;   // episode 内到目标的历史最近距离
// LSTM hidden state（Float32Array，形状 [layers, 1, hidden]）
let _rlHidden = null;   // h state
let _rlCell   = null;   // c state

function _rlResetLstmState() {
  const size = _RL_LSTM_LAYERS * 1 * _RL_LSTM_HIDDEN;
  _rlHidden = new Float32Array(size);   // 全零
  _rlCell   = new Float32Array(size);
  _rlMinDistSoFar = Infinity;
  _rlPrevAction = 0;
}

// 状态：null = 未加载，"loading" = 加载中，InferenceSession = 就绪，"error" = 失败
let _rlSession = null;
let _rlLoadState = "idle";  // "idle" | "loading" | "ready" | "error"

async function _rlLoadModel() {
  if (_rlLoadState !== "idle") return;
  _rlLoadState = "loading";
  try {
    if (typeof ort === "undefined") {
      throw new Error("onnxruntime-web not loaded — check network or CDN");
    }
    ort.env.wasm.numThreads = 1;  // 避免 SharedArrayBuffer 跨域限制
    _rlSession = await ort.InferenceSession.create(_RL_MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    _rlResetLstmState();   // 初始化 LSTM hidden state 为全零
    _rlLoadState = "ready";
    console.log("[RL] assault_policy.onnx (LSTM) loaded, RL mode active.");
  } catch (e) {
    _rlLoadState = "error";
    console.error("[RL] Failed to load model:", e);
  }
}

// LOS 射线步进检测（与 env.py _has_line_of_sight 对齐，steps=16）
function _rlHasLOS(ax, ay, bx, by) {
  const steps = 16;
  for (let i = 1; i < steps; i++) {
    const t  = i / steps;
    const px = ax + (bx - ax) * t;
    const py = ay + (by - ay) * t;
    for (const o of state.obstacles) {
      if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) return false;
    }
  }
  return true;
}

// 子弹预测碰撞时间（与 env.py _bullet_time_to_ally 对齐）
function _rlBulletTTA(b, ally) {
  const dx  = ally.x - b.x;
  const dy  = ally.y - b.y;
  const spd = Math.hypot(b.vx, b.vy);
  if (spd < 1e-6) return 1.0;
  const proj = (dx * b.vx + dy * b.vy) / spd;
  if (proj <= 0) return 1.0;
  const tta = proj / spd;
  return Math.min(1.0, Math.max(0.0, tta / 2.2));  // 2.2 = BULLET_TTL
}

// 射线检测：从 (ox,oy) 沿 (dx,dy) 步进，返回到障碍物/边界的归一化距离
// 与 env.py _raycast_obstacle 严格对齐
function _rlRaycast(ox, oy, dx, dy) {
  const stepLen = _RL_RAY_MAX_DIST / _RL_RAY_STEPS;
  for (let i = 1; i <= _RL_RAY_STEPS; i++) {
    const d  = i * stepLen;
    const px = ox + dx * d;
    const py = oy + dy * d;
    if (px < 0 || px > _RL_CANVAS_W || py < 0 || py > _RL_CANVAS_H) {
      return Math.min(1.0, d / _RL_RAY_MAX_DIST);
    }
    for (const o of state.obstacles) {
      if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) {
        return d / _RL_RAY_MAX_DIST;
      }
    }
  }
  return 1.0;
}

// 构建 121 维观测向量，与 rl/env.py _get_obs() 严格对齐（v2）
function _rlBuildObs(attackCd) {
  const obs  = new Float32Array(_RL_OBS_DIM);
  const ally = state.ally;
  let idx    = 0;

  // ── 段1：自身状态 (4维) ─────────────────────────────────────────────────
  // 去掉 hp 和障碍物距离：障碍物距离会让 agent 隐式学到"远离障碍物=安全"，
  // 导致不敢进入地图中间区域；障碍物位置信息段5里已有完整描述
  obs[idx++] = ally.x / _RL_CANVAS_W;
  obs[idx++] = ally.y / _RL_CANVAS_H;
  obs[idx++] = Math.min(1.0, attackCd / _RL_ASSAULT_INTERVAL);
  obs[idx++] = state.enemies.length / 11.0;

  // ── 段2：主目标敌人（最近，8维）──────────────────────────────────────────
  const target = _rlNearestEnemy();
  if (target !== null) {
    const dx   = target.x - ally.x;
    const dy   = target.y - ally.y;
    const dist = Math.hypot(dx, dy);
    const shootCdMax = target.kind === "boss" ? 1.2 : 1.6;
    const los  = _rlHasLOS(ally.x, ally.y, target.x, target.y);
    const inRange = (dist > 55 && dist < 110) ? 1.0 : 0.0;
    obs[idx++] = dx / _RL_CANVAS_W;
    obs[idx++] = dy / _RL_CANVAS_H;
    obs[idx++] = dist / _RL_DIAG;
    obs[idx++] = target.hp / target.maxHp;
    obs[idx++] = target.kind === "boss" ? 1.0 : 0.0;
    obs[idx++] = target.shootCd / shootCdMax;
    obs[idx++] = los ? 1.0 : 0.0;   // LOS 标志
    obs[idx++] = inRange;            // 有效射程标志
  } else {
    idx += 8;
  }

  // ── 段3：其余最多 4 个敌人 (5维/敌) ────────────────────────────────────
  const others = state.enemies
    .filter(e => e !== target)
    .map(e => ({ e, d: Math.hypot(e.x - ally.x, e.y - ally.y) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, _RL_MAX_ENEMIES - 1);
  for (const { e } of others) {
    const dx = e.x - ally.x;
    const dy = e.y - ally.y;
    obs[idx++] = dx / _RL_CANVAS_W;
    obs[idx++] = dy / _RL_CANVAS_H;
    obs[idx++] = Math.hypot(dx, dy) / _RL_DIAG;
    obs[idx++] = e.hp / e.maxHp;
    obs[idx++] = e.kind === "boss" ? 1.0 : 0.0;
  }
  idx += (_RL_MAX_ENEMIES - 1 - others.length) * 5;

  // ── 段4：最多 8 颗威胁子弹（6维/颗，按 TTA 排序）────────────────────────
  const threatBullets = state.enemyBullets
    .filter(b => Math.hypot(b.x - ally.x, b.y - ally.y) < _RL_MAX_BULLET_DIST)
    .map(b => ({ b, tta: _rlBulletTTA(b, ally) }))
    .sort((a, b) => a.tta - b.tta)   // TTA 越小越危险，排在前面
    .slice(0, _RL_MAX_BULLETS);
  for (const { b, tta } of threatBullets) {
    const dx   = b.x - ally.x;
    const dy   = b.y - ally.y;
    const dist = Math.hypot(dx, dy);
    const bspd = Math.hypot(b.vx, b.vy) || 1.0;
    obs[idx++] = dx / _RL_CANVAS_W;
    obs[idx++] = dy / _RL_CANVAS_H;
    obs[idx++] = b.vx / bspd;
    obs[idx++] = b.vy / bspd;
    obs[idx++] = dist / _RL_MAX_BULLET_DIST;
    obs[idx++] = tta;                 // 预测碰撞时间（0=即将命中）
  }
  idx += (_RL_MAX_BULLETS - threatBullets.length) * 6;

  // ── 段5：射线检测（16 方向，替换矩形障碍物 obs）─────────────────────────
  for (const [rdx, rdy] of _RL_RAY_DIRS) {
    obs[idx++] = _rlRaycast(ally.x, ally.y, rdx, rdy);
  }

  // ── 段6：到四壁距离 (4维) ────────────────────────────────────────────────
  obs[idx++] = ally.y / _RL_CANVAS_H;
  obs[idx++] = (_RL_CANVAS_H - ally.y) / _RL_CANVAS_H;
  obs[idx++] = ally.x / _RL_CANVAS_W;
  obs[idx++] = (_RL_CANVAS_W - ally.x) / _RL_CANVAS_W;

  // ── 段7：上帧动作 one-hot (9维) ─────────────────────────────────────────
  obs[idx + _rlPrevAction] = 1.0;
  idx += _RL_N_ACTIONS;

  // ── 段8：到目标距离变化量 (1维) ─────────────────────────────────────────
  const distNow   = target ? Math.hypot(target.x - ally.x, target.y - ally.y) : _rlPrevDistToTarget;
  const distDelta = _rlPrevDistToTarget - distNow;  // >0 靠近
  obs[idx++] = Math.max(-1.0, Math.min(1.0, distDelta / _RL_CANVAS_W * 10.0));

  // ── 段9：最危险子弹 TTA (1维) ────────────────────────────────────────────
  // 所有威胁子弹中 TTA 最小值，0=即将命中，1=无威胁
  let minTTA = 1.0;
  for (const b of state.enemyBullets) {
    if (Math.hypot(b.x - ally.x, b.y - ally.y) < _RL_MAX_BULLET_DIST) {
      const tta = _rlBulletTTA(b, ally);
      if (tta < minTTA) minTTA = tta;
    }
  }
  obs[idx++] = minTTA;

  // ── 段10：历史最小距离 (1维) ─────────────────────────────────────────────
  // 与 env.py 对齐：先更新历史最近距离，再写入 obs
  if (target) {
    if (distNow < _rlMinDistSoFar) _rlMinDistSoFar = distNow;
    obs[idx++] = Math.min(1.0, _rlMinDistSoFar / _RL_DIAG);
  } else {
    obs[idx++] = 1.0;
  }

  return obs;
}

function _rlNearestEnemy() {
  // LOS 加权评分，与 rl/env.py _nearest_enemy 严格对齐
  // 评分 = 距离 × (LOS通畅 ? 1.0 : 1.8)，优先选有视线且近的敌人
  if (state.enemies.length === 0) return null;
  const ally = state.ally;
  let best = null;
  let bestScore = Infinity;
  for (const e of state.enemies) {
    const d     = Math.hypot(e.x - ally.x, e.y - ally.y);
    const los   = _rlHasLOS(ally.x, ally.y, e.x, e.y);
    const score = d * (los ? 1.0 : 1.8);
    if (score < bestScore) { bestScore = score; best = e; }
  }
  return best;
}

function _rlNearestObstacleDist() {
  if (state.obstacles.length === 0) return _RL_DIAG;
  let minD = Infinity;
  const ally = state.ally;
  for (const o of state.obstacles) {
    const nx = Math.max(o.x, Math.min(ally.x, o.x + o.w));
    const ny = Math.max(o.y, Math.min(ally.y, o.y + o.h));
    const d  = Math.hypot(ally.x - nx, ally.y - ny);
    if (d < minD) minD = d;
  }
  return minD;
}

// 同步推理：每帧调用，返回动作索引 0-8
// onnxruntime-web 的 run() 是 async，但 MLP 推理耗时 < 1ms，
// 使用缓存结果：每帧提交推理任务，下一帧使用上一帧的结果（滞后 1 帧，完全可接受）
let _rlLastAction = 3;  // 默认向右（朝敌人方向）
let _rlInferring  = false;

function _rlInferAsync(attackCd) {
  if (_rlLoadState !== "ready" || _rlInferring) return;
  if (!_rlHidden || !_rlCell) _rlResetLstmState();
  _rlInferring = true;

  // 更新帧间距离（供下帧 obs 段8 使用）
  const target = _rlNearestEnemy();
  const ally   = state.ally;
  _rlPrevDistToTarget = target
    ? Math.hypot(target.x - ally.x, target.y - ally.y)
    : _rlPrevDistToTarget;

  const obs = _rlBuildObs(attackCd);
  // LSTM 输入：obs[1,D]、h_in/c_in[layers,1,hidden]
  const hShape = [_RL_LSTM_LAYERS, 1, _RL_LSTM_HIDDEN];
  const feeds = {
    obs:  new ort.Tensor("float32", obs, [1, _RL_OBS_DIM]),
    h_in: new ort.Tensor("float32", _rlHidden, hShape),
    c_in: new ort.Tensor("float32", _rlCell,   hShape),
  };
  _rlSession.run(feeds).then(output => {
    const logits = output.logits.data;  // Float32Array[9]
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    _rlLastAction = best;
    _rlPrevAction = best;
    // 递推 LSTM hidden state：本帧输出作为下帧输入（关键）
    _rlHidden = output.h_out.data;
    _rlCell   = output.c_out.data;
    _rlInferring = false;
  }).catch((e) => {
    console.error("[RL] inference error:", e);
    _rlInferring = false;
  });
}

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const hudStats = document.getElementById("hudStats");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");

const FLOOR_META = [
  { name: "拔舌狱", tone: "#542127", accent: "#c65f52", haze: "#2a0f12", hint: "妄言会在这里变成锁链，少站直线位。" },
  { name: "剪刀狱", tone: "#4a202b", accent: "#ce7a6e", haze: "#1f1016", hint: "敌群会连线增伤，优先切断牵引目标。" },
  { name: "铁树狱", tone: "#40272c", accent: "#ad7c5a", haze: "#171115", hint: "冲刺别贪，刺林会惩罚错误位移。" },
  { name: "孽镜狱", tone: "#2f2638", accent: "#8f7fe0", haze: "#120f1d", hint: "镜像会复刻行动，先辨认真身再集火。" },
  { name: "蒸笼狱", tone: "#5b2a1d", accent: "#de8a58", haze: "#24120f", hint: "热压会叠层，别在闷区久留。" },
  { name: "铜柱狱", tone: "#5a2e19", accent: "#d88946", haze: "#28150a", hint: "火线扩散很快，卡掩体边缘走。" },
  { name: "刀山狱", tone: "#402826", accent: "#ba8f88", haze: "#1b1314", hint: "地形比怪更危险，先抢安全落脚点。" },
  { name: "冰山狱", tone: "#1d2a3f", accent: "#7ab8d1", haze: "#101722", hint: "冻结后伤害翻倍，别硬吃连续弹道。" },
  { name: "油锅狱", tone: "#52301a", accent: "#d48e4f", haze: "#23160b", hint: "油溅有抛物轨迹，保持横向机动。" },
  { name: "牛坑狱", tone: "#3f281f", accent: "#be7c57", haze: "#190f0a", hint: "冲锋波次会连段，留一个位移保命。" },
  { name: "石压狱", tone: "#3f3132", accent: "#a6978d", haze: "#161314", hint: "塌陷有读秒，宁可少打也别贪刀。" },
  { name: "舂臼狱", tone: "#49352b", accent: "#b89271", haze: "#19120f", hint: "节拍是活路，跟着环境节律移动。" },
  { name: "血池狱", tone: "#5a1f29", accent: "#d66b73", haze: "#2a0e14", hint: "侵蚀会滚雪球，别在血潭硬站桩。" },
  { name: "枉死狱", tone: "#30212d", accent: "#a88db9", haze: "#18101d", hint: "怨魂会扰乱目标，先解控再输出。" },
  { name: "磔刑狱", tone: "#4f2623", accent: "#d37669", haze: "#220f0f", hint: "分部位破坏更高效，别均摊伤害。" },
  { name: "火山狱", tone: "#66261c", accent: "#ee7b42", haze: "#2e120a", hint: "喷发前有前兆，提前换位。" },
  { name: "石磨狱", tone: "#41332a", accent: "#b8a27d", haze: "#1c1410", hint: "碾压周期固定，记住三拍后撤。" },
  { name: "无间边狱", tone: "#2a182b", accent: "#b070da", haze: "#140b16", hint: "终层按因果结算，稳住比抢速更重要。" },
];

const state = {
  player: { x: 180, y: 260, radius: 14, hp: 160, maxHp: 160, speed: 220, attackCd: 0, shieldCd: 0 },
  ally: {
    x: 220,
    y: 290,
    radius: 13,
    hp: 160,
    maxHp: 160,
    speed: 200,
    attackCd: 0,
    rescueCd: 0,
    stance: "guard",
    bubble: "",
    bubbleUntil: 0,
    dead: false,   // 死亡 flag，防止 checkDefeat 每帧重置气泡
  },
  enemies: [],
  playerBullets: [],
  allyBullets: [],
  enemyBullets: [],
  bossAlive: true,
  obstacles: [],
  keys: {},
  result: "",
  playerId: "player_web_demo",
  floor: 1,
  floorState: "playing", // "playing" | "clear"
  transitionTimer: 0,
  cinders: [],
};

// ── 关卡配置 ──────────────────────────────────────────────────────────────────

function floorScale() {
  const f = state.floor - 1;
  return {
    hpMul:    1 + f * 0.30,
    speedMul: 1 + f * 0.06,
    mobCount: Math.min(3 + Math.floor(f * 1.2), 10),
  };
}

const MOB_BASE_POSITIONS = [
  { x: 520, y: 100 }, { x: 650, y: 150 }, { x: 780, y: 100 },
  { x: 560, y: 380 }, { x: 700, y: 430 }, { x: 820, y: 360 },
  { x: 700, y: 270 }, { x: 820, y: 200 }, { x: 760, y: 430 },
  { x: 850, y: 130 },
];

// 在障碍物附近随机采样安全出生点，与 rl/env.py _safe_spawn 逻辑对齐
function safeSpawnPos(baseX, baseY, radius, fallbackX, fallbackY) {
  for (let i = 0; i < 20; i += 1) {
    const x = baseX + (Math.random() - 0.5) * 40;
    const y = baseY + (Math.random() - 0.5) * 40;
    if (!collidesWithObstacle(x, y, radius)) return { x, y };
  }
  return { x: fallbackX, y: fallbackY };
}

function spawnEnemies() {
  const { hpMul, speedMul, mobCount } = floorScale();
  const mobs = [];
  for (let i = 0; i < mobCount; i += 1) {
    const base = MOB_BASE_POSITIONS[i % MOB_BASE_POSITIONS.length];
    const pos  = safeSpawnPos(base.x, base.y, 12, 750, 270);
    mobs.push({
      kind: "mob",
      x: pos.x, y: pos.y,
      hp: Math.round(30 * hpMul),
      maxHp: Math.round(30 * hpMul),
      radius: 12,
      speed: 42 * speedMul,
      shootCd: Math.random() * 1.0 + 0.8,
    });
  }
  const bossPos = safeSpawnPos(820, 270, 20, 830, 400);
  mobs.push({
    kind: "boss",
    x: bossPos.x, y: bossPos.y,
    hp: Math.round(200 * hpMul),
    maxHp: Math.round(200 * hpMul),
    radius: 20,
    speed: 32 * speedMul,
    shootCd: 0.8,
  });
  state.enemies = mobs;
  state.bossAlive = true;
}

// ── 障碍物 ────────────────────────────────────────────────────────────────────

const OBSTACLE_LAYOUTS = [
  // 布局 0：中央横墙 + 两侧竖柱 + 斜角掩体
  [
    { x: 360, y: 255, w: 180, h: 22 },  // 中央横向长条
    { x: 240, y: 170, w: 22,  h: 130 }, // 左侧竖柱
    { x: 660, y: 220, w: 22,  h: 130 }, // 右侧竖柱
    { x: 480, y: 140, w: 140, h: 20 },  // 上方横条
    { x: 420, y: 360, w: 140, h: 20 },  // 下方横条
    { x: 300, y: 360, w: 80,  h: 20 },  // 左下短横
  ],
  // 布局 1：走廊型（上下各一道长墙，中间留缺口）
  [
    { x: 280, y: 145, w: 200, h: 20 },  // 上横墙左段
    { x: 560, y: 145, w: 160, h: 20 },  // 上横墙右段
    { x: 280, y: 375, w: 160, h: 20 },  // 下横墙左段
    { x: 520, y: 375, w: 200, h: 20 },  // 下横墙右段
    { x: 235, y: 220, w: 20,  h: 110 }, // 左竖柱
    { x: 660, y: 210, w: 20,  h: 110 }, // 右竖柱
    { x: 410, y: 245, w: 100, h: 20 },  // 中央横短条
  ],
  // 布局 2：分散长条（斜向交错）
  [
    { x: 270, y: 160, w: 160, h: 20 },  // 左上横条
    { x: 580, y: 200, w: 20,  h: 150 }, // 右侧竖条
    { x: 340, y: 340, w: 160, h: 20 },  // 左下横条
    { x: 630, y: 330, w: 140, h: 20 },  // 右下横条
    { x: 240, y: 280, w: 20,  h: 100 }, // 左竖柱
    { x: 450, y: 150, w: 20,  h: 120 }, // 中竖柱
  ],
  // 布局 3：十字形 + 外围长条
  [
    { x: 390, y: 230, w: 120, h: 20 },  // 横臂
    { x: 445, y: 165, w: 20,  h: 140 }, // 竖臂
    { x: 240, y: 155, w: 130, h: 20 },  // 左上横
    { x: 620, y: 155, w: 130, h: 20 },  // 右上横
    { x: 240, y: 365, w: 130, h: 20 },  // 左下横
    { x: 620, y: 365, w: 130, h: 20 },  // 右下横
  ],
];

function generateObstacles() {
  const idx = (state.floor - 1) % OBSTACLE_LAYOUTS.length;
  state.obstacles = OBSTACLE_LAYOUTS[idx];
}

// 圆形实体与矩形障碍物碰撞检测
function collidesWithObstacle(cx, cy, radius) {
  return state.obstacles.some((o) => {
    const nearX = Math.max(o.x, Math.min(cx, o.x + o.w));
    const nearY = Math.max(o.y, Math.min(cy, o.y + o.h));
    return Math.hypot(cx - nearX, cy - nearY) < radius;
  });
}

// 子弹（点）与障碍物碰撞
function bulletHitsObstacle(bx, by) {
  return state.obstacles.some(
    (o) => bx >= o.x && bx <= o.x + o.w && by >= o.y && by <= o.y + o.h
  );
}

// 带障碍物分量滑动的移动辅助
function moveWithCollision(entity, dx, dy) {
  const r = entity.radius;
  const newX = clampUnit(entity.x + dx, r, canvas.width - r);
  const newY = clampUnit(entity.y + dy, r, canvas.height - r);
  if (!collidesWithObstacle(newX, newY, r)) {
    entity.x = newX;
    entity.y = newY;
  } else if (!collidesWithObstacle(newX, entity.y, r)) {
    entity.x = newX;
  } else if (!collidesWithObstacle(entity.x, newY, r)) {
    entity.y = newY;
  }
}

// ── 关卡过渡 ──────────────────────────────────────────────────────────────────

function nextFloor() {
  state.playerBullets = [];
  state.allyBullets = [];
  state.enemyBullets = [];
  state.player.x = 180; state.player.y = 260;
  state.ally.x = 220;   state.ally.y = 290;
  state.player.hp = Math.min(state.player.maxHp, state.player.hp + 40);
  state.ally.hp   = Math.min(state.ally.maxHp,   state.ally.hp   + 40);
  state.ally.dead = false;   // 进入下一层时复活
  // 关卡切换 = 新 episode，重置 LSTM hidden state
  if (_rlLoadState === "ready") _rlResetLstmState();
  generateObstacles();
  spawnEnemies();
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  setAllyBubble(`第 ${state.floor} 层 ${meta.name}。${meta.hint}`);
}

function updateFloorTransition(dt) {
  if (state.floorState !== "clear") return;
  state.transitionTimer -= dt;
  if (state.transitionTimer <= 0) {
    state.floor += 1;
    nextFloor();
    state.floorState = "playing";
  }
}

// ── 初始化第一关 ──────────────────────────────────────────────────────────────

generateObstacles();
spawnEnemies();

// ── 工具函数 ──────────────────────────────────────────────────────────────────

function nowSeconds() {
  return performance.now() / 1000;
}

function setAllyBubble(text) {
  state.ally.bubble = text;
  const duration = Math.min(12, Math.max(3, (text || "").length * 0.12));
  state.ally.bubbleUntil = nowSeconds() + duration;
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = role === "player" ? `你：${text}` : `${NPC_DISPLAY_NAME}：${text}`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

const EMOTION_KAOMOJI = {
  neutral:   "( ・_・)",
  focused:   "(•̀ᴗ•́)و",
  annoyed:   "(╯°□°）╯",
  worried:   "(；ω；)",
  happy:     "(＾▽＾)",
  tense:     "(°ロ°!)",
  sarcastic: "(¬‿¬)",
};

function emotionKaomoji(emotion) {
  return EMOTION_KAOMOJI[emotion] || EMOTION_KAOMOJI.neutral;
}

function appendStreamingNpcMessage() {
  const el = document.createElement("div");
  el.className = "msg npc";
  el.textContent = `${NPC_DISPLAY_NAME}：`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return {
    append(delta) {
      el.textContent += delta;
      chatLog.scrollTop = chatLog.scrollHeight;
    },
    finish(finalText, emotion) {
      const kaomoji = emotionKaomoji(emotion);
      el.textContent = `${NPC_DISPLAY_NAME} ${kaomoji}：${finalText}`;
      chatLog.scrollTop = chatLog.scrollHeight;
    },
  };
}

function normalize(dx, dy) {
  const len = Math.hypot(dx, dy);
  if (len < 0.0001) return [0, 0];
  return [dx / len, dy / len];
}

function clampUnit(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function findNearestEnemy(from) {
  let nearest = null;
  let minDist = Number.POSITIVE_INFINITY;
  state.enemies.forEach((enemy) => {
    const d = distance(from, enemy);
    if (d < minDist) { minDist = d; nearest = enemy; }
  });
  return [nearest, minDist];
}

// ── 游戏逻辑更新 ──────────────────────────────────────────────────────────────

function removeDeadEnemies() {
  state.enemies = state.enemies.filter((e) => e.hp > 0);
  state.bossAlive = state.enemies.some((e) => e.kind === "boss");
  if (state.enemies.length === 0 && state.floorState === "playing" && !state.result) {
    state.floorState = "clear";
    state.transitionTimer = 3.0;
    setAllyBubble(`第 ${state.floor} 层清除！稍作准备，下一层马上来。`);
  }
}

function createBullet(owner, from, to, speed, damage) {
  const [nx, ny] = normalize(to.x - from.x, to.y - from.y);
  return { owner, x: from.x, y: from.y, vx: nx * speed, vy: ny * speed, radius: 4, damage, ttl: 2.2 };
}

function playerAttack() {
  if (state.player.attackCd > 0 || state.result || state.floorState === "clear") return;
  const [target] = findNearestEnemy(state.player);
  if (!target) return;
  state.player.attackCd = 0.3;
  state.playerBullets.push(createBullet("player", state.player, target, 430, 18));
}

function updatePlayer(dt) {
  if (state.result) return;
  let dx = 0;
  let dy = 0;
  if (state.keys.ArrowUp    || state.keys.KeyW) dy -= 1;
  if (state.keys.ArrowDown  || state.keys.KeyS) dy += 1;
  if (state.keys.ArrowLeft  || state.keys.KeyA) dx -= 1;
  if (state.keys.ArrowRight || state.keys.KeyD) dx += 1;

  const [nx, ny] = normalize(dx, dy);
  moveWithCollision(state.player, nx * state.player.speed * dt, ny * state.player.speed * dt);
  state.player.attackCd = Math.max(0, state.player.attackCd - dt);
  state.player.shieldCd = Math.max(0, state.player.shieldCd - dt);
}

function allyConfig() {
  if (state.ally.stance === "assault") {
    return { attackRange: 110, kiteRange: 65, interval: 0.45, speedMul: 1.2, damage: 13, strafeAmp: 0.3 };
  }
  // guard（默认）
  return { attackRange: 0, kiteRange: 0, interval: 0.75, speedMul: 1.0, damage: 10, strafeAmp: 0 };
}

function strafeVector(from, target) {
  const dx = target.x - from.x;
  const dy = target.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  return [-dy / len, dx / len];
}

function updateAlly(dt) {
  const cfg = allyConfig();
  state.ally.attackCd = Math.max(0, state.ally.attackCd - dt);
  state.ally.rescueCd = Math.max(0, state.ally.rescueCd - dt);
  const speed = state.ally.speed * cfg.speedMul;

  if (state.ally.stance === "guard" || state.player.hp <= 40) {
    const d = distance(state.ally, state.player);
    if (d > 50) {
      const [nx, ny] = normalize(state.player.x - state.ally.x, state.player.y - state.ally.y);
      moveWithCollision(state.ally, nx * speed * dt, ny * speed * dt);
    }
    const [target] = findNearestEnemy(state.ally);
    if (target && state.ally.attackCd <= 0) {
      state.allyBullets.push(createBullet("ally", state.ally, target, 380, cfg.damage));
      state.ally.attackCd = cfg.interval;
    }

  } else if (state.ally.stance === "assault") {
    // 触发懒加载（首次进入 assault 时）
    if (_rlLoadState === "idle") _rlLoadModel();

    const [target] = findNearestEnemy(state.ally);
    if (target) {
      if (_rlLoadState === "ready") {
        // ── RL 控制移动 ────────────────────────────────────────────────────
        // 提交本帧观测，下帧使用（滞后 1 帧，< 16ms，完全可接受）
        _rlInferAsync(state.ally.attackCd);
        const [mx, my] = _RL_ACTION_VECTORS[_rlLastAction];
        moveWithCollision(state.ally, mx * speed * dt, my * speed * dt);
      } else {
        // ── 降级：规则 AI（模型未就绪时）──────────────────────────────────
        const d = distance(state.ally, target);
        if (d > cfg.attackRange) {
          const [nx, ny] = normalize(target.x - state.ally.x, target.y - state.ally.y);
          moveWithCollision(state.ally, nx * speed * dt, ny * speed * dt);
        } else if (d < cfg.kiteRange) {
          const [nx, ny] = normalize(state.ally.x - target.x, state.ally.y - target.y);
          moveWithCollision(state.ally, nx * speed * 0.6 * dt, ny * speed * 0.6 * dt);
        } else {
          const [sx, sy] = strafeVector(state.ally, target);
          moveWithCollision(state.ally, sx * speed * cfg.strafeAmp * dt, sy * speed * cfg.strafeAmp * dt);
        }
      }
      // 攻击逻辑不变：始终自动朝最近敌人开火
      if (state.ally.attackCd <= 0) {
        state.allyBullets.push(createBullet("ally", state.ally, target, 400, cfg.damage));
        state.ally.attackCd = cfg.interval;
      }
    }

  }

  // 边界夹紧
  state.ally.x = clampUnit(state.ally.x, 10, canvas.width - 10);
  state.ally.y = clampUnit(state.ally.y, 10, canvas.height - 10);

  if (state.player.hp <= 45 && state.ally.rescueCd <= 0 && state.ally.stance !== "assault") {
    state.player.hp = Math.min(state.player.maxHp, state.player.hp + 28);
    state.player.shieldCd = 2.2;
    state.ally.rescueCd = 9.0;
    setAllyBubble("先后撤，我给你护盾。");
  }
}

function updateEnemies(dt) {
  if (state.floorState === "clear") return;
  state.enemies.forEach((enemy) => {
    enemy.shootCd = Math.max(0, enemy.shootCd - dt);
    if (state.result) return;

    const distToPlayer = distance(enemy, state.player);
    const distToAlly = state.ally.hp > 0 ? distance(enemy, state.ally) : Number.POSITIVE_INFINITY;
    const primary = distToPlayer <= distToAlly ? state.player : state.ally;
    const [nx, ny] = normalize(primary.x - enemy.x, primary.y - enemy.y);
    moveWithCollision(enemy, nx * enemy.speed * dt, ny * enemy.speed * dt);

    if (enemy.shootCd <= 0) {
      const bulletSpeed  = enemy.kind === "boss" ? 200 : 170;
      const bulletDamage = enemy.kind === "boss" ? 9   : 5;
      state.enemyBullets.push(createBullet("enemy", enemy, primary, bulletSpeed, bulletDamage));
      enemy.shootCd = enemy.kind === "boss" ? 1.2 : 1.6;
    }
  });
}

function updateBullets(dt) {
  const moveBullets = (arr) => {
    for (let i = arr.length - 1; i >= 0; i -= 1) {
      const b = arr[i];
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.ttl -= dt;
      if (
        b.ttl <= 0 ||
        b.x < -10 || b.x > canvas.width + 10 ||
        b.y < -10 || b.y > canvas.height + 10 ||
        bulletHitsObstacle(b.x, b.y)
      ) {
        arr.splice(i, 1);
      }
    }
  };
  moveBullets(state.playerBullets);
  moveBullets(state.allyBullets);
  moveBullets(state.enemyBullets);

  for (let i = state.playerBullets.length - 1; i >= 0; i -= 1) {
    const b = state.playerBullets[i];
    let hit = false;
    for (let j = 0; j < state.enemies.length; j += 1) {
      const e = state.enemies[j];
      if (distance(b, e) <= b.radius + e.radius) { e.hp -= b.damage; hit = true; break; }
    }
    if (hit) state.playerBullets.splice(i, 1);
  }

  for (let i = state.allyBullets.length - 1; i >= 0; i -= 1) {
    const b = state.allyBullets[i];
    let hit = false;
    for (let j = 0; j < state.enemies.length; j += 1) {
      const e = state.enemies[j];
      if (distance(b, e) <= b.radius + e.radius) { e.hp -= b.damage; hit = true; break; }
    }
    if (hit) state.allyBullets.splice(i, 1);
  }

  for (let i = state.enemyBullets.length - 1; i >= 0; i -= 1) {
    const b = state.enemyBullets[i];
    let consumed = false;
    if (distance(b, state.player) <= b.radius + state.player.radius) {
      const raw = b.damage;
      const final = state.player.shieldCd > 0 ? Math.max(2, Math.floor(raw * 0.3)) : raw;
      state.player.hp -= final;
      consumed = true;
    } else if (state.ally.hp > 0 && distance(b, state.ally) <= b.radius + state.ally.radius) {
      state.ally.hp -= b.damage;
      consumed = true;
    }
    if (consumed) state.enemyBullets.splice(i, 1);
  }
}

function checkDefeat() {
  if (state.result || state.floorState === "clear") return;
  if (state.player.hp <= 0) {
    state.player.hp = 0;
    state.result = `战败。坚持到了第 ${state.floor} 层。`;
    setAllyBubble("这次没守住，下轮我们换打法。");
  }
  if (state.ally.hp <= 0 && !state.ally.dead) {
    state.ally.hp    = 0;
    state.ally.dead  = true;
    state.ally.stance = "guard";   // 自动切回守护
    setAllyBubble("灵核失稳...你先继续前进。");
    // 气泡只设一次（3秒），后续不再重置
  }
}

// ── 渲染 ──────────────────────────────────────────────────────────────────────

function ensureCinders() {
  if (state.cinders.length > 0) return;
  for (let i = 0; i < 90; i += 1) {
    state.cinders.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vy: -(18 + Math.random() * 20),
      size: Math.random() < 0.2 ? 2 : 1,
      flicker: Math.random() * Math.PI * 2,
    });
  }
}

function drawBackground(dt) {
  ensureCinders();
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  grad.addColorStop(0, meta.tone);
  grad.addColorStop(1, meta.haze);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255, 191, 160, 0.07)";
  for (let i = 0; i < canvas.width; i += 40) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, canvas.height);
    ctx.stroke();
  }
  for (let j = 0; j < canvas.height; j += 40) {
    ctx.beginPath();
    ctx.moveTo(0, j);
    ctx.lineTo(canvas.width, j);
    ctx.stroke();
  }

  const sigilX = canvas.width * 0.78;
  const sigilY = canvas.height * 0.5;
  ctx.strokeStyle = `${meta.accent}55`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(sigilX, sigilY, 74, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(sigilX, sigilY, 46, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1;

  state.cinders.forEach((c) => {
    c.y += c.vy * dt;
    c.x += Math.sin((performance.now() / 900) + c.flicker) * 0.2;
    if (c.y < -4) {
      c.y = canvas.height + Math.random() * 20;
      c.x = Math.random() * canvas.width;
    }
    ctx.fillStyle = Math.sin((performance.now() / 500) + c.flicker) > 0 ? `${meta.accent}bb` : `${meta.accent}66`;
    ctx.fillRect(c.x, c.y, c.size, c.size);
  });
}

function drawObstacles() {
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  state.obstacles.forEach((o) => {
    ctx.fillStyle = `${meta.haze}dd`;
    ctx.fillRect(o.x, o.y, o.w, o.h);
    ctx.strokeStyle = `${meta.accent}aa`;
    ctx.lineWidth = 2;
    ctx.strokeRect(o.x, o.y, o.w, o.h);
    ctx.fillStyle = `${meta.accent}22`;
    ctx.fillRect(o.x + 6, o.y + 6, o.w - 12, 6);
    ctx.lineWidth = 1;
  });
}

function drawPixelSprite(entity, palette, dir = 1) {
  const px = 4;
  const originX = Math.floor(entity.x - 6 * px);
  const originY = Math.floor(entity.y - 8 * px);
  const head = [
    "0011111100",
    "0112222210",
    "0123333321",
    "0123333321",
    "0012332100",
  ];
  const body = [
    "0001441000",
    "0014444100",
    "0144544410",
    "0144444410",
    "0014545100",
    "0015005100",
  ];
  const rows = [...head, ...body];
  const map = { "1": palette.outline, "2": palette.skin, "3": palette.eye, "4": palette.cloth, "5": palette.trim };

  // 先画一个高可见底座，避免深色关卡里角色不可辨认
  ctx.beginPath();
  ctx.fillStyle = palette.base || "rgba(245, 220, 200, 0.28)";
  ctx.arc(entity.x, entity.y, entity.radius + 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.strokeStyle = palette.baseStroke || "rgba(255, 230, 200, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.arc(entity.x, entity.y, entity.radius + 2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1;

  rows.forEach((row, rowIdx) => {
    [...row].forEach((cell, colIdx) => {
      if (cell === "0") return;
      const drawCol = dir > 0 ? colIdx : row.length - colIdx - 1;
      ctx.fillStyle = map[cell];
      ctx.fillRect(originX + drawCol * px, originY + rowIdx * px, px, px);
    });
  });

  const w = entity.radius * 2;
  const hpRatio = clampUnit(entity.hp / (entity.maxHp || 100), 0, 1);
  ctx.fillStyle = "#101317";
  ctx.fillRect(entity.x - entity.radius, entity.y - entity.radius - 10, w, 4);
  ctx.fillStyle = palette.hp || "#9fe4ff";
  ctx.fillRect(entity.x - entity.radius, entity.y - entity.radius - 10, w * hpRatio, 4);
}

function drawBullets() {
  const drawSet = (arr, color) => {
    ctx.fillStyle = color;
    arr.forEach((b) => {
      const x = Math.floor(b.x);
      const y = Math.floor(b.y);
      ctx.fillRect(x - 1, y - 1, 3, 3);
      ctx.fillRect(x - 2, y, 1, 1);
      ctx.fillRect(x + 2, y, 1, 1);
    });
  };
  drawSet(state.playerBullets, "#6bc8ff");
  drawSet(state.allyBullets,   "#9af19b");
  drawSet(state.enemyBullets,  "#ff9f83");
}

function drawAllyBubble() {
  if (!state.ally.bubble || nowSeconds() > state.ally.bubbleUntil) return;
  const text = state.ally.bubble.slice(0, 90);
  ctx.font = "13px Segoe UI";
  const pad = 8;
  const width  = ctx.measureText(text).width + pad * 2;
  const height = 26;
  const x = clampUnit(state.ally.x - width / 2, 6, canvas.width - width - 6);
  const y = state.ally.y - state.ally.radius - 40;
  ctx.fillStyle = "rgba(20, 24, 38, 0.92)";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = "#8ea3ff";
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = "#e7ecfb";
  ctx.fillText(text, x + pad, y + 17);
}

function drawOverlay() {
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  ctx.fillStyle = "#f8d1b4";
  ctx.font = "bold 16px Segoe UI";
  ctx.fillText(`第 ${state.floor} 层：${meta.name}`, 16, 26);
  ctx.fillStyle = "#f2b48f";
  ctx.font = "13px Segoe UI";
  ctx.fillText(`狱律提示：${meta.hint}`, 16, 46);

  if (state.floorState === "clear") {
    ctx.fillStyle = "rgba(6, 8, 12, 0.70)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ecf1ff";
    ctx.font = "bold 26px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(
      `第 ${state.floor} 层清除！${Math.ceil(state.transitionTimer)} 秒后进入下一层…`,
      canvas.width / 2,
      canvas.height / 2,
    );
    ctx.textAlign = "left";
  }
  if (state.result) {
    ctx.fillStyle = "rgba(6, 8, 12, 0.65)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ecf1ff";
    ctx.font = "bold 24px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(state.result, canvas.width / 2, canvas.height / 2);
    ctx.textAlign = "left";
  }
}

function render(dt) {
  drawBackground(dt);
  drawObstacles();
  drawPixelSprite(state.player, { outline: "#1f2f39", skin: "#e1bfa1", eye: "#6ec4ff", cloth: "#3f86c2", trim: "#9fe4ff", hp: "#8dd0ff" }, 1);
  drawPixelSprite(state.ally, { outline: "#2f1d14", skin: "#d7b399", eye: "#ffb77f", cloth: "#6e3d2a", trim: "#d08c55", base: "rgba(255, 188, 132, 0.30)", baseStroke: "rgba(255, 202, 160, 0.60)", hp: "#b8f7b6" }, 1);
  state.enemies.forEach((enemy) => {
    drawPixelSprite(
      enemy,
      enemy.kind === "boss"
        ? { outline: "#320d0d", skin: "#d7a299", eye: "#ffd1d1", cloth: "#a72f2f", trim: "#ff7d62", base: "rgba(255, 110, 94, 0.32)", baseStroke: "rgba(255, 138, 116, 0.60)", hp: "#ffd7d2" }
        : { outline: "#321a13", skin: "#c79f86", eye: "#ffcc9c", cloth: "#8f4c35", trim: "#d98b5a", base: "rgba(240, 148, 112, 0.26)", baseStroke: "rgba(255, 178, 140, 0.5)", hp: "#ffd7a3" },
      enemy.x < state.player.x ? -1 : 1,
    );
  });
  drawBullets();
  drawAllyBubble();
  drawOverlay();
}

function updateHud() {
  const stanceLabel = STANCE_LABELS[state.ally.stance] || state.ally.stance;
  const floorName = FLOOR_META[(state.floor - 1) % FLOOR_META.length].name;

  // assault 姿态时显示 RL 模型加载状态
  let rlTag = "";
  if (state.ally.stance === "assault") {
    if      (_rlLoadState === "ready")   rlTag = " [RL]";
    else if (_rlLoadState === "loading") rlTag = " [RL 加载中…]";
    else if (_rlLoadState === "error")   rlTag = " [RL 失败·规则]";
    else                                 rlTag = " [规则]";
  }

  hudStats.textContent =
    `${floorName}（第 ${state.floor} 层） | 玩家 HP ${Math.floor(state.player.hp)} | 乌枭 HP ${Math.floor(state.ally.hp)} | 姿态 ${stanceLabel}${rlTag} | 敌人 ${state.enemies.length}`;
}

// ── 主循环 ────────────────────────────────────────────────────────────────────

let lastTs = performance.now();
function loop(ts) {
  const dt = Math.min(0.033, (ts - lastTs) / 1000);
  lastTs = ts;

  updatePlayer(dt);
  updateAlly(dt);
  updateEnemies(dt);
  updateBullets(dt);
  removeDeadEnemies();
  checkDefeat();
  updateFloorTransition(dt);
  render(dt);
  updateHud();
  requestAnimationFrame(loop);
}

// ── 输入 ──────────────────────────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyA", "KeyS", "KeyD"].includes(e.code)) {
    e.preventDefault();
  }
  state.keys[e.code] = true;
  if (e.code === "Space") { e.preventDefault(); playerAttack(); }
});
document.addEventListener("keyup", (e) => { state.keys[e.code] = false; });

// ── NPC 对话 ──────────────────────────────────────────────────────────────────

const NPC_ID   = "wuxiao_01";
const NPC_NAME = "乌枭";
const NPC_DISPLAY_NAME = "乌枭";
const STANCE_LABELS = { assault: "突击", guard: "守护" };

function buildSceneInfo() {
  return {
    mode:        "battle",
    floor:       state.floor,
    ally_stance: state.ally.stance,
    player_hp:   Math.floor(state.player.hp),
    ally_hp:     Math.floor(state.ally.hp),
    enemy_count: state.enemies.length,
    boss_alive:  state.bossAlive,
  };
}

function applyStance(stance, reply) {
  if (!stance) return;
  // 客户端保底：NPC 无血时不允许切突击（正常情况由后端拦截并给出回复）
  if (stance === "assault" && state.ally.dead) return;
  // 切到突击 = 新的决策 episode，重置 LSTM hidden state，避免上一段记忆污染
  if (stance === "assault" && _rlLoadState === "ready") _rlResetLstmState();
  state.ally.stance = stance;
  const label  = STANCE_LABELS[stance] || stance;
  const bubble = reply || `姿态切换：${label}。`;
  setAllyBubble(bubble);
  appendMessage("npc", bubble);
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  appendMessage("player", message);

  const payload = {
    player_id:  state.playerId,
    npc_id:     NPC_ID,
    npc_name:   NPC_NAME,
    message,
    scene_info: buildSceneInfo(),
  };

  try {
    const resp = await fetch(`${NPC_API}/api/chat/stream`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();

    let msgNode       = null;
    let accumulated   = "";
    let buffer        = "";
    let bubbleThrottle = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let evt;
        try { evt = JSON.parse(trimmed); } catch { continue; }

        if (evt.type === "command") {
          applyStance(evt.stance, evt.reply);
        } else if (evt.type === "delta") {
          if (!msgNode) msgNode = appendStreamingNpcMessage();
          accumulated += evt.text;
          msgNode.append(evt.text);
          const now = Date.now();
          if (now - bubbleThrottle > 100) { setAllyBubble(accumulated); bubbleThrottle = now; }
        } else if (evt.type === "done") {
          const finalText = evt.action?.dialogue || accumulated || "收到，我会继续和你协同。";
          const emotion   = evt.action?.emotion  || "neutral";
          if (msgNode) msgNode.finish(finalText, emotion);
          setAllyBubble(`${emotionKaomoji(emotion)} ${finalText}`);
        } else if (evt.type === "error") {
          const fallbackText = evt.fallback?.dialogue || "连接中断。我会继续执行上一条指令。";
          if (msgNode) msgNode.finish(fallbackText, "neutral");
          else appendMessage("npc", fallbackText);
          setAllyBubble(fallbackText);
        }
      }
    }
  } catch (err) {
    const text = "连接中断。我会继续执行上一条指令。";
    appendMessage("npc", text);
    setAllyBubble(text);
  }
});

appendMessage("npc", "黑签鬼差乌枭到位。你别乱送，我就能把你带到无间边狱。");
requestAnimationFrame(loop);
