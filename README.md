# AI Roguelite Web Demo (MVP)

这是一个可独立运行的网页 Demo，包含：

- 浏览器端 2D 战斗玩法（移动、射击、敌人与 NPC 同伴协作）
- 单 NPC 对话后端（`LangGraph + ChromaDB`）
- 通过自然语言控制 NPC 战术姿态（守护 / 突击 / 游击）
- 记忆系统（短期记忆 + 长期记忆）
- 长期记忆上下文拼装（世界观设定 + 角色设定 + 对话历史）

---

## 项目目录介绍

```text
ai-roguelite-web
├─ run.py                          # 启动入口（Flask）
├─ requirements.txt                # Python 依赖
├─ config.yaml                     # 当前生效配置（仅全局参数，不含 npc）
├─ config.example.yaml             # 配置模板
├─ lore
│  ├─ world_setting.md             # 世界观设定文档（用于导入 Chroma）
│  └─ persona_setting.md           # 角色设定文档（用于导入 Chroma）
├─ scripts
│  ├─ import_world_setting.py      # 导入世界观设定到 Chroma
│  └─ import_persona_setting.py    # 导入角色设定到 Chroma（需传 --npc-id）
└─ server
   ├─ app.py                       # Web 路由（/api/chat、/api/command）
   ├─ templates/index.html         # 页面模板
   ├─ static
   │  ├─ game.js                   # 前端战斗逻辑、对话调用
   │  └─ styles.css
   └─ npc_backend
      ├─ config.py                 # 配置加载
      ├─ schemas.py                # 请求/响应结构（ChatRequest / CommandResponse）
      ├─ short_term.py             # 短期记忆（最近 N 轮）
      ├─ memory.py                 # 长期记忆（Chroma：world / persona / dialogue）
      ├─ prompts.py                # Prompt 组装
      ├─ llm.py                    # LLM 调用、记忆分级、意图分类
      └─ graph.py                  # LangGraph 主流程
```

---

## 启动命令（完整步骤）

### 1) 进入项目目录并安装依赖

```bash
cd d:\otherwise\demo\ai-roguelite-web
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

### 3) 启动服务

```bash
python run.py
```

### 4) 打开页面

```text
http://127.0.0.1:5100
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
| 其他对话内容 | **对话（dialogue）** | 转入 LangGraph 完整对话流程 |

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

### `POST /api/chat`

完整对话接口，经由 LangGraph 生成 NPC 回复，并写入长期记忆。

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

响应：

```json
{
  "action_type": "dialogue",
  "dialogue": "三个，别废话，跟上！",
  "emotion": "neutral"
}
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

## 当前实现范围

- 单 NPC（`ember_01`）对话
- `LangGraph` 对话编排
- Chroma 长期记忆标签分层：
  - `memory_type=world`（世界观，全局共享）
  - `memory_type=persona`（角色设定，按 `npc_id` 隔离）
  - `memory_type=dialogue` + `dialogue_tier=daily|important`（对话历史）
- 短期记忆维持最近 N 轮上下文连续性
- LLM 意图分类驱动 NPC 战术姿态切换
- NPC 三种战术行为：守护 / 突击 / 游击
