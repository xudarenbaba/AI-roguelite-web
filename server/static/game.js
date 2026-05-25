const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const hudStats = document.getElementById("hudStats");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");

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
  keys: {},
  result: "",
  playerId: "player_web_demo",
};

function spawnEnemies() {
  const mobs = [];
  // 6 个小兵，分两组错开分布
  const mobPositions = [
    { x: 520, y: 100 }, { x: 650, y: 150 }, { x: 780, y: 100 },
    { x: 560, y: 380 }, { x: 700, y: 430 }, { x: 820, y: 360 },
  ];
  mobPositions.forEach(({ x, y }) => {
    mobs.push({
      kind: "mob",
      x,
      y,
      hp: 30,
      maxHp: 30,
      radius: 12,
      speed: 42,
      shootCd: Math.random() * 1.0 + 0.8,
    });
  });
  mobs.push({
    kind: "boss",
    x: 820,
    y: 270,
    hp: 200,
    maxHp: 200,
    radius: 20,
    speed: 32,
    shootCd: 0.8,
  });
  state.enemies = mobs;
}

spawnEnemies();

function nowSeconds() {
  return performance.now() / 1000;
}

function setAllyBubble(text) {
  state.ally.bubble = text;
  // 按字数动态计算气泡停留时间：每字 0.12 秒，最短 3 秒，最长 12 秒
  const duration = Math.min(12, Math.max(3, (text || "").length * 0.12));
  state.ally.bubbleUntil = nowSeconds() + duration;
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = role === "player" ? `你：${text}` : `烬：${text}`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendStreamingNpcMessage() {
  const el = document.createElement("div");
  el.className = "msg npc";
  el.textContent = "烬：";
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return {
    append(delta) {
      el.textContent += delta;
      chatLog.scrollTop = chatLog.scrollHeight;
    },
    finish(finalText) {
      el.textContent = `烬：${finalText}`;
      chatLog.scrollTop = chatLog.scrollHeight;
    },
  };
}

function normalize(dx, dy) {
  const len = Math.hypot(dx, dy);
  if (len < 0.0001) {
    return [0, 0];
  }
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
    if (d < minDist) {
      minDist = d;
      nearest = enemy;
    }
  });
  return [nearest, minDist];
}

function removeDeadEnemies() {
  state.enemies = state.enemies.filter((e) => e.hp > 0);
  state.bossAlive = state.enemies.some((e) => e.kind === "boss");
  if (state.enemies.length === 0 && !state.result) {
    state.result = "胜利！这层高塔已被你们清空。";
    setAllyBubble("本层清理完成，协同效率很高。");
  }
}

function createBullet(owner, from, to, speed, damage) {
  const [nx, ny] = normalize(to.x - from.x, to.y - from.y);
  return {
    owner,
    x: from.x,
    y: from.y,
    vx: nx * speed,
    vy: ny * speed,
    radius: 4,
    damage,
    ttl: 2.2,
  };
}

function playerAttack() {
  if (state.player.attackCd > 0 || state.result) {
    return;
  }
  const [target] = findNearestEnemy(state.player);
  if (!target) return;
  state.player.attackCd = 0.3;
  state.playerBullets.push(createBullet("player", state.player, target, 430, 18));
}

function updatePlayer(dt) {
  if (state.result) {
    return;
  }
  let dx = 0;
  let dy = 0;
  if (state.keys.ArrowUp || state.keys.KeyW) dy -= 1;
  if (state.keys.ArrowDown || state.keys.KeyS) dy += 1;
  if (state.keys.ArrowLeft || state.keys.KeyA) dx -= 1;
  if (state.keys.ArrowRight || state.keys.KeyD) dx += 1;

  const [nx, ny] = normalize(dx, dy);
  state.player.x = clampUnit(state.player.x + nx * state.player.speed * dt, 12, canvas.width - 12);
  state.player.y = clampUnit(state.player.y + ny * state.player.speed * dt, 12, canvas.height - 12);
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
  // skirmish
  return { attackRange: 150, kiteRange: 90, interval: 0.55, speedMul: 1.3, damage: 11, strafeAmp: 1.0 };
}

// 找血量最低的存活小兵；若无小兵则返回最近敌人
function findWeakestEnemy() {
  const mobs = state.enemies.filter((e) => e.kind === "mob" && e.hp > 0);
  if (mobs.length > 0) {
    return mobs.reduce((a, b) => (a.hp <= b.hp ? a : b));
  }
  return findNearestEnemy(state.ally)[0];
}

// 计算来袭子弹的规避方向向量（返回 [dx, dy]，已标准化）
function calcDodgeVector() {
  let ox = 0;
  let oy = 0;
  state.enemyBullets.forEach((b) => {
    const d = distance(b, state.ally);
    if (d < 90) {
      // 垂直于子弹速度方向的偏转（取法向量）
      const len = Math.hypot(b.vx, b.vy) || 1;
      ox += -b.vy / len / Math.max(1, d * 0.04);
      oy += b.vx / len / Math.max(1, d * 0.04);
    }
  });
  const mag = Math.hypot(ox, oy);
  return mag > 0.01 ? [ox / mag, oy / mag] : [0, 0];
}

// 绕目标圆周运动的切向量（顺时针）
function strafeVector(from, target) {
  const dx = target.x - from.x;
  const dy = target.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  return [-dy / len, dx / len];   // 顺时针切线
}

function updateAlly(dt) {
  const cfg = allyConfig();
  state.ally.attackCd = Math.max(0, state.ally.attackCd - dt);
  state.ally.rescueCd = Math.max(0, state.ally.rescueCd - dt);

  const speed = state.ally.speed * cfg.speedMul;

  // ── guard：紧跟玩家，不主动追敌 ──────────────────────────────────────
  if (state.ally.stance === "guard" || state.player.hp <= 40) {
    const followDist = 50;
    const d = distance(state.ally, state.player);
    if (d > followDist) {
      const [nx, ny] = normalize(state.player.x - state.ally.x, state.player.y - state.ally.y);
      state.ally.x += nx * speed * dt;
      state.ally.y += ny * speed * dt;
    }
    const [target] = findNearestEnemy(state.ally);
    if (target && state.ally.attackCd <= 0) {
      state.allyBullets.push(createBullet("ally", state.ally, target, 380, cfg.damage));
      state.ally.attackCd = cfg.interval;
    }

  // ── assault：主动冲近，近身持续输出，轻微横向走位 ─────────────────────
  } else if (state.ally.stance === "assault") {
    const [target] = findNearestEnemy(state.ally);
    if (target) {
      const d = distance(state.ally, target);
      if (d > cfg.attackRange) {
        // 冲向目标
        const [nx, ny] = normalize(target.x - state.ally.x, target.y - state.ally.y);
        state.ally.x += nx * speed * dt;
        state.ally.y += ny * speed * dt;
      } else if (d < cfg.kiteRange) {
        // 太近了往后退一步
        const [nx, ny] = normalize(state.ally.x - target.x, state.ally.y - target.y);
        state.ally.x += nx * speed * 0.6 * dt;
        state.ally.y += ny * speed * 0.6 * dt;
      } else {
        // 在攻击范围内：横向走位
        const [sx, sy] = strafeVector(state.ally, target);
        state.ally.x += sx * speed * cfg.strafeAmp * dt;
        state.ally.y += sy * speed * cfg.strafeAmp * dt;
      }
      if (state.ally.attackCd <= 0) {
        state.allyBullets.push(createBullet("ally", state.ally, target, 400, cfg.damage));
        state.ally.attackCd = cfg.interval;
      }
    }

  // ── skirmish：优先清弱怪，绕圈走位 + 子弹规避 + 近身后撤 ─────────────
  } else {
    const target = findWeakestEnemy();
    if (target) {
      const d = distance(state.ally, target);
      let mx = 0;
      let my = 0;

      if (d > cfg.attackRange) {
        // 向目标靠近
        const [nx, ny] = normalize(target.x - state.ally.x, target.y - state.ally.y);
        mx += nx;
        my += ny;
      } else if (d < cfg.kiteRange) {
        // 太近：后撤
        const [nx, ny] = normalize(state.ally.x - target.x, state.ally.y - target.y);
        mx += nx * 1.2;
        my += ny * 1.2;
      } else {
        // 射程内：绕圈
        const [sx, sy] = strafeVector(state.ally, target);
        mx += sx * cfg.strafeAmp;
        my += sy * cfg.strafeAmp;
      }

      // 叠加子弹规避权重 0.7
      const [dodgeX, dodgeY] = calcDodgeVector();
      mx += dodgeX * 0.7;
      my += dodgeY * 0.7;

      const [fnx, fny] = normalize(mx, my);
      state.ally.x += fnx * speed * dt;
      state.ally.y += fny * speed * dt;

      if (state.ally.attackCd <= 0) {
        state.allyBullets.push(createBullet("ally", state.ally, target, 380, cfg.damage));
        state.ally.attackCd = cfg.interval;
      }
    }
  }

  state.ally.x = clampUnit(state.ally.x, 10, canvas.width - 10);
  state.ally.y = clampUnit(state.ally.y, 10, canvas.height - 10);

  // 救援：守护/游击姿态下玩家低血触发
  if (state.player.hp <= 45 && state.ally.rescueCd <= 0 && state.ally.stance !== "assault") {
    state.player.hp = Math.min(state.player.maxHp, state.player.hp + 28);
    state.player.shieldCd = 2.2;
    state.ally.rescueCd = 9.0;
    setAllyBubble("先后撤，我给你护盾。");
  }
}

function updateEnemies(dt) {
  state.enemies.forEach((enemy) => {
    enemy.shootCd = Math.max(0, enemy.shootCd - dt);
    if (state.result) return;

    const distToPlayer = distance(enemy, state.player);
    const distToAlly = state.ally.hp > 0 ? distance(enemy, state.ally) : Number.POSITIVE_INFINITY;
    const primary = distToPlayer <= distToAlly ? state.player : state.ally;
    const [nx, ny] = normalize(primary.x - enemy.x, primary.y - enemy.y);
    enemy.x += nx * enemy.speed * dt;
    enemy.y += ny * enemy.speed * dt;

    if (enemy.shootCd <= 0) {
      const bulletSpeed = enemy.kind === "boss" ? 200 : 170;
      const bulletDamage = enemy.kind === "boss" ? 9 : 5;
      state.enemyBullets.push(createBullet("enemy", enemy, primary, bulletSpeed, bulletDamage));
      enemy.shootCd = enemy.kind === "boss" ? 1.2 : 1.6;
    }
  });
}

function updateBullets(dt) {
  const updateOneSide = (arr) => {
    for (let i = arr.length - 1; i >= 0; i -= 1) {
      const b = arr[i];
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.ttl -= dt;
      if (b.ttl <= 0 || b.x < -10 || b.x > canvas.width + 10 || b.y < -10 || b.y > canvas.height + 10) {
        arr.splice(i, 1);
      }
    }
  };
  updateOneSide(state.playerBullets);
  updateOneSide(state.allyBullets);
  updateOneSide(state.enemyBullets);

  for (let i = state.playerBullets.length - 1; i >= 0; i -= 1) {
    const b = state.playerBullets[i];
    let hit = false;
    for (let j = 0; j < state.enemies.length; j += 1) {
      const e = state.enemies[j];
      if (distance(b, e) <= b.radius + e.radius) {
        e.hp -= b.damage;
        hit = true;
        break;
      }
    }
    if (hit) state.playerBullets.splice(i, 1);
  }

  for (let i = state.allyBullets.length - 1; i >= 0; i -= 1) {
    const b = state.allyBullets[i];
    let hit = false;
    for (let j = 0; j < state.enemies.length; j += 1) {
      const e = state.enemies[j];
      if (distance(b, e) <= b.radius + e.radius) {
        e.hp -= b.damage;
        hit = true;
        break;
      }
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
  if (state.result) {
    return;
  }
  if (state.player.hp <= 0) {
    state.player.hp = 0;
    state.result = "战败。你在塔底再次苏醒。";
    setAllyBubble("这次没守住，下轮我们换打法。");
  }
  if (state.ally.hp <= 0) {
    state.ally.hp = 0;
    setAllyBubble("灵核失稳...你先继续前进。");
  }
}

function drawCharacter(entity, color, hpColor) {
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.arc(entity.x, entity.y, entity.radius, 0, Math.PI * 2);
  ctx.fill();
  const w = entity.radius * 2;
  const hpRatio = clampUnit(entity.hp / (entity.maxHp || 100), 0, 1);
  ctx.fillStyle = "#101317";
  ctx.fillRect(entity.x - entity.radius, entity.y - entity.radius - 10, w, 4);
  ctx.fillStyle = hpColor;
  ctx.fillRect(entity.x - entity.radius, entity.y - entity.radius - 10, w * hpRatio, 4);
}

function drawBackground() {
  ctx.fillStyle = "#1a1f2b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < canvas.width; i += 48) {
    ctx.strokeStyle = "rgba(130, 145, 190, 0.10)";
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, canvas.height);
    ctx.stroke();
  }
  for (let j = 0; j < canvas.height; j += 48) {
    ctx.strokeStyle = "rgba(130, 145, 190, 0.10)";
    ctx.beginPath();
    ctx.moveTo(0, j);
    ctx.lineTo(canvas.width, j);
    ctx.stroke();
  }
}

function drawBullets() {
  const drawSet = (arr, color) => {
    ctx.fillStyle = color;
    arr.forEach((b) => {
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
      ctx.fill();
    });
  };
  drawSet(state.playerBullets, "#6bc8ff");
  drawSet(state.allyBullets, "#9af19b");
  drawSet(state.enemyBullets, "#ff9f83");
}

function drawAllyBubble() {
  if (!state.ally.bubble || nowSeconds() > state.ally.bubbleUntil) {
    return;
  }
  const text = state.ally.bubble.slice(0, 90);
  ctx.font = "13px Segoe UI";
  const pad = 8;
  const width = ctx.measureText(text).width + pad * 2;
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

function render() {
  drawBackground();

  drawCharacter(state.player, "#4ea7ff", "#8dd0ff");
  drawCharacter(state.ally, "#8fdb8a", "#b8f7b6");
  state.enemies.forEach((enemy) => {
    drawCharacter(enemy, enemy.kind === "boss" ? "#c45757" : "#d78a55", "#ffd7a3");
  });
  drawBullets();

  drawAllyBubble();

  if (state.result) {
    ctx.fillStyle = "rgba(6, 8, 12, 0.65)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ecf1ff";
    ctx.font = "24px Segoe UI";
    ctx.fillText(state.result, 180, 270);
  }
}

function updateHud() {
  const stanceLabel = {
    assault: "突击",
    guard: "守护",
    skirmish: "游击",
  }[state.ally.stance] || state.ally.stance;
  hudStats.textContent = `玩家 HP ${Math.floor(state.player.hp)} | 烬 HP ${Math.floor(
    state.ally.hp
  )} | 姿态 ${stanceLabel} | 敌人数 ${state.enemies.length}`;
}

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
  render();
  updateHud();
  requestAnimationFrame(loop);
}

document.addEventListener("keydown", (e) => {
  state.keys[e.code] = true;
  if (e.code === "Space") {
    e.preventDefault();
    playerAttack();
  }
});

document.addEventListener("keyup", (e) => {
  state.keys[e.code] = false;
});

const NPC_ID = "ember_01";
const NPC_NAME = "烬";

const STANCE_LABELS = { assault: "突击", guard: "守护", skirmish: "游击" };

function buildSceneInfo() {
  return {
    mode: "battle",
    ally_stance: state.ally.stance,
    player_hp: Math.floor(state.player.hp),
    ally_hp: Math.floor(state.ally.hp),
    enemy_count: state.enemies.length,
    boss_alive: state.bossAlive,
  };
}

function applyStance(stance, reply) {
  if (!stance) return;
  state.ally.stance = stance;
  const label = STANCE_LABELS[stance] || stance;
  const bubble = reply || `姿态切换：${label}。`;
  setAllyBubble(bubble);
  appendMessage("npc", bubble);
}

async function sendDialogue(message) {
  const payload = {
    player_id: state.playerId,
    npc_id: NPC_ID,
    npc_name: NPC_NAME,
    message,
    scene_info: buildSceneInfo(),
  };

  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  const msgNode = appendStreamingNpcMessage();

  let accumulated = "";
  let buffer = "";
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
      try {
        evt = JSON.parse(trimmed);
      } catch {
        continue;
      }

      if (evt.type === "delta") {
        accumulated += evt.text;
        msgNode.append(evt.text);
        const now = Date.now();
        if (now - bubbleThrottle > 100) {
          setAllyBubble(accumulated);
          bubbleThrottle = now;
        }
      } else if (evt.type === "done") {
        const finalText = evt.action?.dialogue || accumulated || "收到，我会继续和你协同。";
        msgNode.finish(finalText);
        setAllyBubble(finalText);
      } else if (evt.type === "error") {
        const fallbackText = evt.fallback?.dialogue || "连接中断。我会继续执行上一条指令。";
        msgNode.finish(fallbackText);
        setAllyBubble(fallbackText);
      }
    }
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  appendMessage("player", message);

  try {
    const cmdResp = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        npc_name: NPC_NAME,
        scene_info: buildSceneInfo(),
      }),
    });
    const cmdData = await cmdResp.json();

    if (cmdData.type === "command") {
      applyStance(cmdData.stance, cmdData.reply);
    } else {
      await sendDialogue(message);
    }
  } catch (err) {
    const text = "连接中断。我会继续执行上一条指令。";
    appendMessage("npc", text);
    setAllyBubble(text);
  }
});

appendMessage("npc", "系统在线。边战斗边下达指令。");
requestAnimationFrame(loop);
