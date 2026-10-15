#!/usr/bin/env python3
"""Create a portable HTML review page from dataset transfer diagnostic runs."""
import argparse
import csv
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    sections = []
    for file in sorted(args.input_dir.glob("*_full_body/summary.json")):
        report = json.loads(file.read_text())
        name = file.parent.name
        result = report["strategies"]["adaptive_whole_body"]
        with (file.parent / "adaptive_whole_body_frames.csv").open() as stream:
            frames = list(csv.DictReader(stream))
        metrics = []
        for side in ("left", "right"):
            values = [float(r[f"{side}_whole_body_isotropy_ratio"]) for r in frames
                      if r[f"{side}_reachable"] == "True"]
            low = sum(v < .04 for v in values)
            metrics.append(f"{side}: min η {min(values):.4f}, Low Manip. {low}/{len(values)} ({100*low/len(values):.2f}%)" if values else f"{side}: N/A")
        ranges = report["mobility"]["motion_ranges"]
        motion = "; ".join(f"{key}: {value['min']:.4f} to {value['max']:.4f}"
                           for key, value in ranges.items())
        videos = "".join(f'<video controls playsinline preload="metadata" src="{name}/{p.name}"></video>'
                         for p in sorted(file.parent.glob("*.mp4")))
        sections.append(f"""<section><h2>DM-SPARKY 全身 18 DoF：{html.escape(name)}</h2>
<p>逐帧连续求解，无抽帧插值。双臂可达 {result['both_reachable_frames']}/{result['frames']} = {100*result['both_reachable_rate']:.2f}%。
评估 episode：{report['episodes']}。下面视频播放完整 episode 0，按源数据 30 Hz 播放。</p>
<p>{'<br>'.join(metrics)}<br>此处 η 与 Low Manip. 使用全身 Jacobian，可达帧为分母。</p>
{videos}<p>启用双臂 14 + 膝/腰 3 + 底盘前后平移 1 自由度。底盘搜索范围 −0.50 到 −0.05 m。
轨迹点：左红、右蓝。状态点：绿色可达且 η ≥ 0.04、黄色低 η、红色不可达。
指标及运动范围统计覆盖本次选择的全部 episode。</p>
<p>实际运动范围（底盘单位 m，其余 rad）：{html.escape(motion)}</p>
<p class="warning">此结果为运动学上限。继承上一帧解不等于已施加速度、加速度或分支跳变约束。未验证碰撞、动力学或真机执行安全。并非全部硬件关节开放。</p>
<p><a href="{name}/summary.json">完整计算配置与结果</a></p></section>""")
    for file in sorted(args.input_dir.glob("*/validation.json")):
        report = json.loads(file.read_text())
        name = file.parent.name
        rows = []
        for robot in report["robots"]:
            samples = robot["sampled_independent_ik"]
            n = len(samples)
            cells = []
            for side in ("left", "right"):
                reachable = [r[side] for r in samples if r[side]["reachable"]]
                low = sum(r["isotropy_ratio"] < .04 for r in reachable)
                cells += [f"{len(reachable)}/{n}" if n else "N/A",
                          f"{low}/{len(reachable)} ({100*low/len(reachable):.2f}%)" if reachable else "N/A"]
            both = sum(r["left"]["reachable"] and r["right"]["reachable"] for r in samples)
            parity = robot["mujoco_parity"]
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in [
                html.escape(robot["robot"]), *cells,
                f"{both}/{n} ({100*both/n:.2f}%)" if n else "BLOCKED",
                f"{'PASS' if parity['pass'] else 'FAIL'} / {1e6*parity['max_position_difference_m']:.3f} µm",
            ]) + "</tr>")
        images = []
        for robot in ("tianji", "humanoid", "realman"):
            matches = sorted(file.parent.glob(f"{robot}_ep*.png"))
            if matches:
                p = matches[0]
                images.append(f'<figure><img src="{name}/{p.name}"><figcaption>{robot}: fixed-arm diagnostic</figcaption></figure>')
        # Show raw axes for unregistered new data, not an invented robot alignment.
        if not images:
            for p in sorted(file.parent.glob("source_ep0_frame*.png"), key=lambda p: int(p.stem.split("frame")[-1])):
                images.append(f'<figure><img src="{name}/{p.name}"><figcaption>{p.stem}</figcaption></figure>')
            camera = file.parent / "ego_ep0_mid.png"
            if camera.exists():
                images.append(f'<figure><img src="{name}/{camera.name}"><figcaption>Original Ego video, episode 0 midpoint. No re-undistortion.</figcaption></figure>')
        sections.append(f"""<section><h2>{html.escape(name)}</h2>
<p>Dataset: {html.escape(report['dataset']['dataset'])}<br>
World status: <strong>{html.escape(report['world_status'])}</strong><br>
Selected episodes: {', '.join(str(e['episode']) for e in report['episodes'])}</p>
<table><tr><th>Robot</th><th>Left reachable</th><th>Left low η / reachable</th><th>Right reachable</th><th>Right low η / reachable</th><th>Both reachable</th><th>FK parity / max Δp</th></tr>{''.join(rows)}</table>
<div class="images">{''.join(images)}</div><p><a href="{name}/validation.json">Numerical details and assumptions</a></p></section>""")
    page = """<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>跨数据集验证</title>
<style>body{font:17px/1.6 sans-serif;background:#111b24;color:#eef4fa;margin:32px auto;max-width:1500px;padding:0 24px}h1,h2{color:#69d9ce}section{margin:36px 0 60px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #536778;padding:10px;text-align:center}th{background:#213a4c}.images{display:flex;flex-wrap:wrap;gap:16px;margin-top:24px}figure{margin:0;width:calc(50% - 8px)}img{width:100%;display:block}video{width:100%;max-width:1280px;display:block;background:black}figcaption{color:#b9c8d5}a{color:#81c7ff}.warning{background:#443120;padding:18px}</style>
<h1>跨数据集可达性与可操纵性诊断</h1>
<p class="warning">顶部视频与全身结果来自新跑的 18 自由度逐帧连续求解。下方旧诊断仍保留固定躯干、每 episode 12 帧的抽样结果，仅作对照，两者统计分母不同。所有旧数据沿用此前确认的板系，真实场景配准没有重新实测。新数据坐标系不明时，结论仍为 BLOCKED，不视为不可达。</p>
<p>判据：位置误差 &lt; 5 mm，姿态误差 &lt; 3°，有效关节限位内。η = σmin([Jp; 0.30 Jr]) / σmax([Jp; 0.30 Jr])。Low Manip. 仅以可达帧为分母统计 η &lt; 0.04；无可达帧时为 N/A。未检查轨迹连续性、碰撞、速度和动力学。</p>
<p>新增防串用检查：录制 CSV 的帧序号、时间戳、目标位置必须与所选数据和场景一致。通过数值一致性检查不代表世界系已完成物理配准。</p>
""" + "".join(sections) + "</html>"
    (args.input_dir / "index.html").write_text(page, encoding="utf-8")
    print(args.input_dir / "index.html")


if __name__ == "__main__":
    main()
