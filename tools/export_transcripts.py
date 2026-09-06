"""Export Claude Code session transcripts (.jsonl) to readable Markdown.

Conversation text is kept in full; tool calls become one-liners and tool results
are truncated, because the point is to read the discussion, not replay the run.

Why this exists: Claude Code prunes its transcript folder on a timer, and the desktop app
has crashed mid-session more than once. Andrew wrote this on 2026-08-20 for his machine;
it came into the repository on 2026-09-06 when Douglas' machine turned out to have never
been exported at all. Paths come from the command line so one copy serves both machines:

    .venv/Scripts/python.exe tools/export_transcripts.py --dst "D:\\MEE2024 output\\MEE_transcripts"

--src defaults to the current user's ~/.claude/projects. Re-running is safe: a session
whose title has changed replaces its earlier export (the session id is the stable part
of the filename), and INDEX.md is rewritten each time. Each machine runs it on a daily
Windows scheduled task; see docs/README.md, "Not in this folder".
"""
import argparse
import json
import os
import re
import glob
import datetime

SRC = os.path.join(os.path.expanduser("~"), ".claude", "projects")   # overridden by --src
DST = None                                                           # required: --dst
RESULT_CHARS = 400          # per tool result
TOOL_ARG_CHARS = 160        # per tool call argument blob


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def strip_reminders(text):
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    return text.strip()


def fence(text, lang=""):
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def clip(text, n):
    text = text.rstrip()
    if len(text) <= n:
        return text, False
    return text[:n].rstrip(), True


def render_tool_use(block):
    name = block.get("name", "?")
    args = block.get("input", {}) or {}
    # Prefer the argument that identifies the target.
    for key in ("file_path", "path", "pattern", "command", "url", "prompt", "query"):
        if key in args:
            val = str(args[key]).replace("\n", " ")
            val, cut = clip(val, TOOL_ARG_CHARS)
            return f"- **{name}** `{val}{'…' if cut else ''}`"
    blob = json.dumps(args, ensure_ascii=False)
    blob, cut = clip(blob, TOOL_ARG_CHARS)
    return f"- **{name}** `{blob}{'…' if cut else ''}`"


def result_text(block):
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("[image]")
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def render(rows, meta):
    out = []
    stats = {"user": 0, "assistant": 0, "tools": 0, "thinking": 0, "images": 0}

    for row in rows:
        rtype = row.get("type")
        if rtype not in ("user", "assistant"):
            continue
        msg = row.get("message") or {}
        content = msg.get("content")
        stamp = (row.get("timestamp") or "")[:19].replace("T", " ")
        side = " *(subagent)*" if row.get("isSidechain") else ""

        blocks = []
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = [b for b in content if isinstance(b, dict)]

        texts, tools, results = [], [], []
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text", ""))
            elif btype == "thinking":
                stats["thinking"] += 1
            elif btype == "tool_use":
                tools.append(render_tool_use(block))
                stats["tools"] += 1
            elif btype == "tool_result":
                results.append(result_text(block))
            elif btype == "image":
                stats["images"] += 1
                texts.append("*[image attached]*")

        body = strip_reminders("\n\n".join(t for t in texts if t))

        if rtype == "user":
            # A turn carrying only tool results is machine chatter, not dialogue.
            if not body and results:
                for res in results:
                    res, cut = clip(res, RESULT_CHARS)
                    if res:
                        out.append("<details><summary>tool result</summary>\n\n"
                                   + fence(res + ("\n…[truncated]" if cut else ""))
                                   + "\n\n</details>\n")
                continue
            if not body:
                continue
            stats["user"] += 1
            out.append(f"\n---\n\n### 👤 User · {stamp}{side}\n\n{body}\n")
        else:
            if not body and not tools:
                continue
            stats["assistant"] += 1
            out.append(f"\n### 🤖 Claude · {stamp}{side}\n")
            if body:
                out.append(body + "\n")
            if tools:
                out.append("\n".join(tools) + "\n")

    header = [
        f"# {meta['title']}\n",
        f"*Session `{meta['sid']}` · {meta['start']} → {meta['end']} · "
        f"{stats['user']} prompts, {stats['assistant']} replies, {stats['tools']} tool calls*\n",
        f"*Working directory: `{meta['cwd']}`*\n",
        "> Exported from the Claude Code transcript. Conversation text is complete; "
        f"tool arguments are abbreviated and tool results truncated to {RESULT_CHARS} characters. "
        f"{stats['thinking']} internal reasoning blocks omitted.\n",
    ]
    return "\n".join(header) + "\n".join(out), stats


def slugify(text):
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:55] or "session"


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*", "*.jsonl")))
    files += sorted(glob.glob(os.path.join(SRC, "*", "*", "subagents", "*.jsonl")))
    index = []

    for path in files:
        rows = load(path)
        if not rows:
            continue
        sid = os.path.basename(path)[:-6]

        title = ""
        for row in rows:                       # last title wins
            if row.get("type") == "custom-title" and row.get("customTitle"):
                title = row["customTitle"]
            elif row.get("type") == "ai-title" and row.get("aiTitle") and not title:
                title = row["aiTitle"]
        if not title:
            for row in rows:                   # fall back to the opening prompt
                if row.get("type") == "user":
                    content = (row.get("message") or {}).get("content")
                    text = content if isinstance(content, str) else " ".join(
                        b.get("text", "") for b in content or [] if isinstance(b, dict))
                    text = strip_reminders(text or "")
                    if text:
                        title = text.split("\n")[0][:70]
                        break
        if not title:
            title = f"Session {sid[:8]}"
        if "subagents" in path:
            title = f"Subagent {sid[-8:]} — {title}"

        stamps = [r["timestamp"] for r in rows if r.get("timestamp")]
        start = min(stamps)[:10] if stamps else "unknown"
        end = max(stamps)[:10] if stamps else "unknown"
        cwd = next((r["cwd"] for r in rows if r.get("cwd")), "?")

        text, stats = render(rows, {"title": title, "sid": sid, "start": start,
                                    "end": end, "cwd": cwd})
        name = f"{start}_{slugify(title)}_{sid[:8]}.md"
        dest = os.path.join(DST, name)

        # A conversation's title is generated from its content and changes as the
        # conversation develops -- and the title is in the filename. Run this daily over a
        # session that spans a week and each new title would leave the previous file
        # behind, so the folder fills with near-duplicates of one conversation. The
        # session id is the stable part, so anything carrying this id under a different
        # name is a superseded copy of what is about to be written.
        for stale in glob.glob(os.path.join(DST, f"*_{sid[:8]}.md")):
            if os.path.basename(stale) != name:
                os.remove(stale)
                print(f"  superseded {os.path.basename(stale)}")
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)
        index.append((start, end, title, name, stats, os.path.getsize(dest)))
        print(f"{os.path.getsize(dest)/1000:8.0f} kB  {name}")

    index.sort()
    lines = [
        "# MEE2024 — Claude Code transcript archive\n",
        f"*Exported {datetime.date.today().isoformat()} from "
        f"`{SRC}`. {len(index)} sessions.*\n",
        "These are readable copies. The originals live in the Claude Code working",
        "directory, which prunes transcripts on a timer; this folder does not.\n",
        "| Period | Session | Prompts | Replies | Tools | File |",
        "|---|---|---|---|---:|---|",
    ]
    for start, end, title, name, stats, _ in index:
        period = start if start == end else f"{start} → {end}"
        safe = title.replace("|", "\\|")
        lines.append(f"| {period} | {safe} | {stats['user']} | {stats['assistant']} "
                     f"| {stats['tools']} | [{name}]({name.replace(' ', '%20')}) |")

    with open(os.path.join(DST, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(index)} transcripts + INDEX.md to {DST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export Claude Code transcripts to readable Markdown.")
    ap.add_argument("--src", default=SRC, help="the Claude Code projects folder (default: this user's)")
    ap.add_argument("--dst", required=True, help="where the Markdown copies and INDEX.md go")
    args = ap.parse_args()
    SRC, DST = args.src, args.dst
    main()
