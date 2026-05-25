# AI Roguelite Web Demo (MVP)

这是一个前后端分离的网页 Demo，包含：

- 浏览器端 2D 战斗玩法（移动、射击、障碍物掩体、无限关卡）
- 单 NPC 流式对话后端（`Flask streaming + ChromaDB`）
- 通过自然语言控制 NPC 战术姿态（守护 / 突击 / 游击）
- 记忆系统（短期记忆 + 长期记忆）
- 长期记忆上下文并行检索（世界观设定 + 角色设定 + 对话历史）
- NPC 情绪系统（7 种情绪 + 颜文字展示）

---

![模拟游戏界面-1](images/img1.png)
![模拟游戏界面-2](images/img2.png)

## 项目目录介绍

```text
ai-roguelite-web
├─ run.py                          # 启动 NPC API 服务（端口 5100）
├─ run_game.py                     # 启动游戏静态文件服务（端口 8080）
├─ requirements.txt                # Python 依赖
├─ config.yaml                     # 当前生效配置
├─ config.example.yaml             # 配置模板
├─ game                            # 游戏客户端（纯静态，独立于后端）
│  ├─ index.html
│  ├─ game.js                      # 游戏逻辑：无限关卡、障碍物、NPC 对话
│  └─ styles.css
├─ lore
│  ├─ world_setting.md             # 世界观设定文档（用于导入 Chroma）
│  └─ persona_setting.md           # 角色设定文档（用于导入 Chroma）
├─ scripts
│  ├─ import_world_setting.py      # 导入世界观设定到 Chroma
│  └─ import_persona_setting.py    # 导入角色设定到 Chroma（需传 --npc-id）
└─ server
   ├─ app.py                       # NPC API 路由（/api/chat/stream、/api/command）
   └─ npc_backend
      ├─ config.py                 # 配置加载
      ├─ schemas.py                # 请求/响应结构（ChatRequest / CommandResponse）
      ├─ short_term.py             # 短期记忆（最近 N 轮）
      ├─ memory.py                 # 长期记忆（Chroma：world / persona / dialogue）
      ├─ prompts.py                # Prompt 组装
      ├─ llm.py                    # LLM 普通调用、流式调用、记忆分级、意图分类
      └─ graph.py                  # 流式 NPC 对话引擎（NpcConversationEngine）
```

---

## 启动命令（完整步骤）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 导入基础设定到 ChromaDB

```bash
python scripts/import_world_setting.py
python scripts/import_persona_setting.py --npc-id ember_01
```

导入策略说明：

- 世界观设定（`memory_type=world`）为**覆盖导入**：重新执行会替换旧的 world seed。
- 角色设定（`memory_type=persona`）为**覆盖导入**：重新执行会替换指定 `npc_id` 的 persona seed。
- 对话记忆（`memory_type=dialogue`）为**运行时追加**：不会被导入脚本清空。

### 3) 启动 NPC API 服务（终端 1）

```bash
python run.py
```

监听 `http://127.0.0.1:5100`，提供 `/api/command` 和 `/api/chat/stream`。

### 4) 启动游戏静态服务器（终端 2）

```bash
python run_game.py
```

监听 `http://127.0.0.1:8080`，托管 `game/` 目录。

### 5) 打开游戏页面

```text
http://127.0.0.1:8080
```

---

## 配置说明

- 项目已包含 `config.yaml`，可直接运行
- embedding 使用本地模式：`local_files_only: true`
- 可通过环境变量覆盖 LLM 配置：
  - `AI_NPC_LLM_API_KEY`
  - `AI_NPC_LLM_BASE_URL`
  - `AI_NPC_LLM_MODEL`

Windows PowerShell 示例：

```powershell
$env:AI_NPC_LLM_API_KEY="sk-xxxx"
$env:AI_NPC_LLM_BASE_URL="https://api.deepseek.com"
$env:AI_NPC_LLM_MODEL="deepseek-chat"
python run.py
```

---

## NPC 战术控制

NPC（烬）的战术姿态由玩家自然语言驱动，**无需按钮**。发送消息后后端先调用 `POST /api/command` 进行意图分类：

| 玩家输入示例 | 识别结果 | NPC 行为 |
|---|---|---|
| "回来保护我" / "别乱跑" | **守护（guard）** | 跟随玩家，攻击最近敌人 |
| "上去打" / "压制它" | **突击（assault）** | 主动追击敌人，近距离激进输出 |
| "先清小怪" / "游击打法" | **游击（skirmish）** | 优先清除最弱目标，灵活走位闪避 |
| 其他对话内容 | **对话（dialogue）** | 转入 HTTP 流式对话流程 |

NPC 初始姿态为**守护**。

---

## API 接口说明

### `POST /api/command`

意图分类接口，判断玩家输入是战术指令还是普通对话。

请求体：

```json
{
  "message": "回来保护我",
  "npc_name": "烬",
  "scene_info": {}
}
```

响应（指令）：

```json
{ "type": "command", "stance": "guard", "reply": "收到，我来护着你！" }
```

响应（对话）：

```json
{ "type": "dialogue" }
```

---

### `POST /api/chat/stream`

流式对话接口。后端会先并行读取短期记忆和长期记忆上下文，再调用 LLM 流式生成 NPC 回复。响应格式为 **NDJSON**，即每行一个 JSON 事件。

请求体：

```json
{
  "player_id": "p1",
  "npc_id": "ember_01",
  "npc_name": "烬",
  "message": "这里有多少敌人？",
  "scene_info": { "enemy_count": 3 }
}
```

响应流示例：

```text
{"type":"meta","npc_id":"ember_01"}
{"type":"delta","text":"三个，"}
{"type":"delta","text":"别废话，跟上！"}
{"type":"done","action":{"action_type":"dialogue","dialogue":"三个，别废话，跟上！","emotion":"focused"}}
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `player_id` | 是 | 玩家唯一 ID，用于记忆检索隔离 |
| `npc_id` | 是 | NPC 唯一 ID，作为记忆检索主键 |
| `message` | 是 | 玩家输入内容 |
| `scene_info` | 否 | 当前场景信息（敌人数量、位置等） |
| `npc_name` | 否 | NPC 显示名，不传则使用 `npc_id` |

---

## 对话链路

普通对话的主链路：

```text
game.js 表单提交
  ↓
POST /api/command
  ↓
type=dialogue 时调用 POST /api/chat/stream
  ↓
NpcConversationEngine.stream_chat()
  ↓
并行读取短期记忆 + 长期记忆上下文
  ↓
build_messages()
  ↓
chat_completion_stream()
  ↓
后端逐行返回 NDJSON delta
  ↓
前端 TextDecoder 读取并逐块更新 NPC 消息
  ↓
流结束后写入短期记忆和长期记忆
```

长期记忆检索由 `MemoryStore.search_context()` 统一完成。它会对当前 query 只计算一次 embedding，然后并行查询四类 Chroma 记忆：

- `world_chunks`：世界观设定，`memory_type=world`
- `persona_chunks`：角色设定，`memory_type=persona`
- `dialogue_daily_chunks`：日常对话记忆，`memory_type=dialogue` + `dialogue_tier=daily`
- `dialogue_important_chunks`：重要对话记忆，`memory_type=dialogue` + `dialogue_tier=important`

---

## 当前实现范围

**游戏端（`game/`）**

- 2D 俯视战斗，WASD 移动、空格射击
- 无限关卡：每层清空后自动进入下一层，敌人数量/血量/速度随层数递增
- 三种障碍物布局按层循环，玩家/NPC/敌人/子弹均有碰撞
- 场景信息传递当前层数（`floor`）给 NPC 后端
- NPC 情绪颜文字展示（7 种情绪）

**NPC 后端（`server/`）**

- 前后端分离，Flask 仅提供 API，支持跨域（CORS）
- HTTP 流式对话输出（`/api/chat/stream`，NDJSON 格式）
- NPC 回复携带情绪标签，后端解析后随 `done` 事件返回
- 流结束后后台异步写入短期记忆与长期记忆
- Chroma 长期记忆标签分层：
  - `memory_type=world`（世界观，全局共享）
  - `memory_type=persona`（角色设定，按 `npc_id` 隔离）
  - `memory_type=dialogue` + `dialogue_tier=daily|important`（对话历史）
- 长期记忆一次 embedding + 四路并行检索
- 短期记忆维持最近 N 轮上下文连续性
- LLM 意图分类驱动 NPC 战术姿态切换
- NPC 三种战术行为：守护 / 突击 / 游击
