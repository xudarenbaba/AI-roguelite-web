const NPC_API = "http://127.0.0.1:5100";

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

function spawnEnemies() {
  const { hpMul, speedMul, mobCount } = floorScale();
  const mobs = [];
  for (let i = 0; i < mobCount; i += 1) {
    const base = MOB_BASE_POSITIONS[i % MOB_BASE_POSITIONS.length];
    mobs.push({
      kind: "mob",
      x: base.x + (Math.random() - 0.5) * 40,
      y: base.y + (Math.random() - 0.5) * 40,
      hp: Math.round(30 * hpMul),
      maxHp: Math.round(30 * hpMul),
      radius: 12,
      speed: 42 * speedMul,
      shootCd: Math.random() * 1.0 + 0.8,
    });
  }
  mobs.push({
    kind: "boss",
    x: 820,
    y: 270,
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
  // 布局 0：中央掩体 + 两侧掩护
  [
    { x: 410, y: 220, w: 100, h: 100 },
    { x: 270, y: 160, w: 70,  h: 40  },
    { x: 620, y: 320, w: 70,  h: 40  },
  ],
  // 布局 1：上下横墙 + 左右侧墙
  [
    { x: 370, y: 140, w: 90, h: 50 },
    { x: 370, y: 350, w: 90, h: 50 },
    { x: 265, y: 235, w: 55, h: 70 },
    { x: 640, y: 235, w: 55, h: 70 },
  ],
  // 布局 2：菱形分散掩体
  [
    { x: 290, y: 155, w: 85, h: 45 },
    { x: 540, y: 195, w: 85, h: 45 },
    { x: 290, y: 340, w: 85, h: 45 },
    { x: 540, y: 310, w: 85, h: 45 },
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
    return { attackRange: 110, kiteRange: 55, interval: 0.45, speedMul: 1.2, damage: 13, strafeAmp: 0.3 };
  }
  if (state.ally.stance === "guard") {
    return { attackRange: 0, kiteRange: 0, interval: 0.75, speedMul: 1.0, damage: 10, strafeAmp: 0 };
  }
  return { attackRange: 150, kiteRange: 90, interval: 0.55, speedMul: 1.3, damage: 11, strafeAmp: 1.0 };
}

function findWeakestEnemy() {
  const mobs = state.enemies.filter((e) => e.kind === "mob" && e.hp > 0);
  if (mobs.length > 0) return mobs.reduce((a, b) => (a.hp <= b.hp ? a : b));
  return findNearestEnemy(state.ally)[0];
}

function calcDodgeVector() {
  let ox = 0; let oy = 0;
  state.enemyBullets.forEach((b) => {
    const d = distance(b, state.ally);
    if (d < 90) {
      const len = Math.hypot(b.vx, b.vy) || 1;
      ox += -b.vy / len / Math.max(1, d * 0.04);
      oy +=  b.vx / len / Math.max(1, d * 0.04);
    }
  });
  const mag = Math.hypot(ox, oy);
  return mag > 0.01 ? [ox / mag, oy / mag] : [0, 0];
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
    const [target] = findNearestEnemy(state.ally);
    if (target) {
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
      if (state.ally.attackCd <= 0) {
        state.allyBullets.push(createBullet("ally", state.ally, target, 400, cfg.damage));
        state.ally.attackCd = cfg.interval;
      }
    }

  } else {
    const target = findWeakestEnemy();
    if (target) {
      const d = distance(state.ally, target);
      let mx = 0; let my = 0;
      if (d > cfg.attackRange) {
        const [nx, ny] = normalize(target.x - state.ally.x, target.y - state.ally.y);
        mx += nx; my += ny;
      } else if (d < cfg.kiteRange) {
        const [nx, ny] = normalize(state.ally.x - target.x, state.ally.y - target.y);
        mx += nx * 1.2; my += ny * 1.2;
      } else {
        const [sx, sy] = strafeVector(state.ally, target);
        mx += sx * cfg.strafeAmp; my += sy * cfg.strafeAmp;
      }
      const [dodgeX, dodgeY] = calcDodgeVector();
      mx += dodgeX * 0.7; my += dodgeY * 0.7;
      const [fnx, fny] = normalize(mx, my);
      moveWithCollision(state.ally, fnx * speed * dt, fny * speed * dt);
      if (state.ally.attackCd <= 0) {
        state.allyBullets.push(createBullet("ally", state.ally, target, 380, cfg.damage));
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
  if (state.ally.hp <= 0) {
    state.ally.hp = 0;
    setAllyBubble("灵核失稳...你先继续前进。");
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
  const stanceLabel = { assault: "突击", guard: "守护", skirmish: "游击" }[state.ally.stance] || state.ally.stance;
  const floorName = FLOOR_META[(state.floor - 1) % FLOOR_META.length].name;
  hudStats.textContent =
    `${floorName}（第 ${state.floor} 层） | 玩家 HP ${Math.floor(state.player.hp)} | 乌枭 HP ${Math.floor(state.ally.hp)} | 姿态 ${stanceLabel} | 敌人 ${state.enemies.length}`;
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
const STANCE_LABELS = { assault: "突击", guard: "守护", skirmish: "游击" };

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
