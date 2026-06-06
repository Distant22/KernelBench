"""Export the GitHub Copilot Chat session-store turns to a Markdown transcript.

This produces the "raw data" (input prompts + agent outputs) appendix for the
Team 37 final report. Read-only access to the session-store SQLite DB.
"""
import os
import sqlite3

DB = os.path.expanduser(
    "~/.vscode-server/data/User/globalStorage/github.copilot-chat/session-store.db"
)
OUT = os.path.join(os.path.dirname(__file__), "agent_conversation_log.md")


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    sessions = con.execute(
        "SELECT id, summary, agent_name, created_at, updated_at FROM sessions "
        "ORDER BY created_at"
    ).fetchall()

    lines = []
    lines.append("# Team 37 — AI Agent 對話逐字稿 (原始資料)\n")
    lines.append("> 由 VS Code GitHub Copilot Chat session-store 自動匯出。")
    lines.append("> 內容為「input prompt（使用者訊息）」與「output（Agent 回應）」的原始紀錄。")
    lines.append("> 角色／系統提示見 `finalProject_260531/PROMPT.md`。\n")

    n_sessions = 0
    n_turns = 0
    for s in sessions:
        turns = con.execute(
            "SELECT turn_index, user_message, assistant_response, timestamp "
            "FROM turns WHERE session_id=? ORDER BY turn_index",
            (s["id"],),
        ).fetchall()
        # skip empty/placeholder sessions
        real = [t for t in turns if (t["user_message"] or t["assistant_response"])]
        if not real:
            continue
        n_sessions += 1
        lines.append("\n---\n")
        lines.append(f"## Session {n_sessions}  (`{s['id'][:8]}`)\n")
        lines.append(f"- 建立：{s['created_at']}  ・ 更新：{s['updated_at']}")
        lines.append(f"- Agent：{s['agent_name']}")
        if s["summary"]:
            summ = s["summary"].replace("\n", " ")[:120]
            lines.append(f"- 摘要：{summ}")
        lines.append("")

        last_user = None
        for t in real:
            um = (t["user_message"] or "").strip()
            ar = (t["assistant_response"] or "").strip()
            # collapse duplicated user turns (UI sometimes logs the prompt twice)
            if um and um != last_user:
                lines.append(f"\n### ▶ User (turn {t['turn_index']})\n")
                lines.append(um)
                last_user = um
            if ar:
                n_turns += 1
                lines.append(f"\n### ◀ Agent (turn {t['turn_index']})\n")
                lines.append(ar)
        lines.append("")

    header_stats = (
        f"\n> 統計：{n_sessions} 個有效 session、{n_turns} 則 Agent 回應。\n"
    )
    lines.insert(4, header_stats)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}: {n_sessions} sessions, {n_turns} agent responses, "
          f"{os.path.getsize(OUT)//1024} KB")


if __name__ == "__main__":
    main()
