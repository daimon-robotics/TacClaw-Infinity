"""Dependency-free interactive HTML visualization for arm-quality results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_quality_dashboard(
    path: str | Path,
    summary: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
) -> Path:
    """Write an offline dashboard with per-frame reachability and Jacobian metrics."""
    destination = Path(path).expanduser().resolve()
    compact: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for strategy, strategy_rows in rows.items():
        episodes: dict[str, list[dict[str, Any]]] = {}
        for row in strategy_rows:
            episode = str(int(row["episode_index"]))
            episodes.setdefault(episode, []).append({
                "f": int(row["frame_index"]),
                "t": float(row["timestamp_s"]),
                "lr": _required_bool(row["left_reachable"]),
                "rr": _required_bool(row["right_reachable"]),
                "li": _optional_float(row["left_isotropy_ratio"]),
                "ri": _optional_float(row["right_isotropy_ratio"]),
                "lm": _optional_float(row["left_joint_margin_fraction"]),
                "rm": _optional_float(row["right_joint_margin_fraction"]),
                "lp": float(row["left_position_error_m"]) * 1000.0,
                "rp": float(row["right_position_error_m"]) * 1000.0,
                "lo": float(row["left_orientation_error_deg"]),
                "ro": float(row["right_orientation_error_deg"]),
                "lx": float(row["left_target_x_m"]),
                "ly": float(row["left_target_y_m"]),
                "lz": float(row["left_target_z_m"]),
                "rx": float(row["right_target_x_m"]),
                "ry": float(row["right_target_y_m"]),
                "rz": float(row["right_target_z_m"]),
                "by": _optional_float(row.get("base_root_y_table_m")),
                "kn": _optional_float(row.get("knee_joint")),
                "hp": _optional_float(row.get("hip_pitch_joint")),
                "hy": _optional_float(row.get("hip_yaw_joint")),
            })
        compact[strategy] = episodes

    payload = {
        "robot": summary["robot_model"]["robot_name"],
        "comparison": summary["comparison_conclusion"],
        "thresholds": summary["thresholds"],
        "strategies": summary["strategies"],
        "frames": compact,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent an unusual string value from terminating the inline script.
    encoded = encoded.replace("</", "<\\/")
    html = _HTML.replace("__QUALITY_DATA__", encoded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _required_bool(value: Any) -> bool:
    """Accept native booleans and the CSV spellings used by this project."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return bool(value)


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>双臂轨迹质量可视化</title>
<style>
:root{color-scheme:light;--left:#d84a45;--right:#3478d4;--good:#2f9e62;--warn:#e39a21;--bad:#d84a45;--ink:#182230;--muted:#667085;--line:#d8dee8;--panel:#fff}
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
main{max-width:1240px;margin:auto;padding:28px}.title{display:flex;gap:18px;align-items:end;justify-content:space-between;flex-wrap:wrap}h1{font-size:26px;margin:0 0 4px}.muted{color:var(--muted)}
.controls,.cards,.legend{display:flex;gap:12px;flex-wrap:wrap}.controls label{font-weight:600}.controls select{margin-left:6px;padding:6px 10px;border:1px solid var(--line);border-radius:7px;background:#fff}
.cards{margin:18px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 9px #1822300b}.card{padding:13px 16px;min-width:180px}.card b{display:block;font-size:21px}.panel{padding:16px;margin:14px 0}.panel h2{font-size:16px;margin:0 0 8px}
canvas{display:block;width:100%;height:260px}.reach{display:grid;grid-template-columns:90px 1fr;gap:7px;align-items:center}.band{height:20px;display:flex;border-radius:4px;overflow:hidden;background:#e9edf2}.segment{height:100%;min-width:1px}.ok{background:var(--good)}.bad{background:var(--bad)}
.trajectory3d{height:440px;border:1px solid var(--line);border-radius:8px;cursor:grab;touch-action:none}.trajectory3d.dragging{cursor:grabbing}.trajectory-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.trajectory-grid canvas{height:330px;border:1px solid var(--line);border-radius:8px}@media(max-width:760px){.trajectory-grid{grid-template-columns:1fr}.trajectory3d{height:360px}}
.legend span:before{content:"";display:inline-block;width:11px;height:11px;margin-right:5px;border-radius:50%;background:var(--c)}.note{padding:10px 12px;background:#fff8e7;border-left:4px solid var(--warn);margin-top:14px}.left{--c:var(--left)}.right{--c:var(--right)}.threshold{--c:var(--warn)}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--line)}
</style>
</head>
<body><main>
<div class="title"><div><h1>双臂轨迹质量可视化</h1><div id="robot" class="muted"></div></div>
<div class="controls"><label>评估策略<select id="strategy"></select></label><label>Episode<select id="episode"></select></label></div></div>
<div class="cards"><div class="card">并列结论<b id="comparison"></b></div><div class="card">双臂同时可达率<b id="bothRate"></b></div><div class="card">当前帧数<b id="frameCount"></b></div></div>
<section class="panel"><h2>逐帧可达性</h2><div class="reach"><b>左臂</b><div id="leftReach" class="band"></div><b>右臂</b><div id="rightReach" class="band"></div></div><div class="legend" style="margin-top:9px"><span style="--c:var(--good)">满足位姿容差</span><span style="--c:var(--bad)">不满足/无有效解</span></div></section>
<section class="panel"><h2>TCP 目标轨迹点（base_link 坐标系）</h2><canvas id="trajectory3D" class="trajectory3d"></canvas><div class="muted">拖拽旋转，滚轮缩放，双击恢复视角。</div><div class="trajectory-grid"><canvas id="trajectoryXY"></canvas><canvas id="trajectoryXZ"></canvas></div><div class="legend" style="margin-top:9px"><span class="left">左手 TCP</span><span class="right">右手 TCP</span><span style="--c:#f0a51a">前 10 帧橙圈并标帧号</span><span style="--c:#111827">第 0 帧黑圈；叉号表示不可达</span></div></section>
<section id="mobilityPanel" class="panel" style="display:none"><h2>人形底盘与膝腰运动</h2><canvas id="basePosition"></canvas><canvas id="torsoAngles"></canvas><div class="legend"><span style="--c:#7c3aed">底盘 table-Y (m)</span><span style="--c:#2f9e62">膝 (rad)</span><span style="--c:#e39a21">腰俯仰 (rad)</span><span style="--c:#8b5cf6">腰偏航 (rad)</span></div></section>
<section class="panel"><h2>Jacobian isotropy ratio（越大越均衡）</h2><canvas id="isotropy"></canvas><div class="legend"><span class="left">左臂</span><span class="right">右臂</span><span class="threshold">低可操纵性阈值</span></div></section>
<section class="panel"><h2>最小归一化关节裕量（0 表示贴近限位）</h2><canvas id="margin"></canvas><div class="legend"><span class="left">左臂</span><span class="right">右臂</span></div></section>
<section class="panel"><h2>当前 Episode 汇总</h2><table><thead><tr><th></th><th>可达帧</th><th>可达率</th><th>最低 isotropy</th><th>最低关节裕量</th><th>最大位置误差</th><th>最大姿态误差</th></tr></thead><tbody id="episodeSummary"></tbody></table></section>
<div class="note">可操纵性只在满足位姿容差的帧上计算；不可达帧显示为红色且指标为空。本报告不包含碰撞、速度/加速度或真机安全结论。</div>
</main>
<script>
const DATA=__QUALITY_DATA__;
const $=id=>document.getElementById(id), strategy=$('strategy'), episode=$('episode');
$('trajectory3D').setAttribute('aria-label','可旋转的左右 TCP 三维轨迹');
$('robot').textContent=DATA.robot; $('comparison').textContent=DATA.comparison;
Object.keys(DATA.frames).forEach(name=>strategy.add(new Option(name,name)));
function episodes(){return Object.keys(DATA.frames[strategy.value]).sort((a,b)=>Number(a)-Number(b));}
function refillEpisodes(){const old=episode.value;episode.innerHTML='';episodes().forEach(id=>episode.add(new Option(id,id)));if(episodes().includes(old))episode.value=old;render();}
function pct(v){return (100*v).toFixed(2)+'%'}
function valueRange(values,threshold){const finite=values.filter(v=>v!==null&&Number.isFinite(v));return Math.max(threshold*1.25,...finite,1e-6)}
function draw(canvas,left,right,threshold){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);c.clearRect(0,0,w,h);const pad={l:52,r:16,t:12,b:28},iw=w-pad.l,ih=h-pad.t-pad.b,max=valueRange(left.concat(right),threshold||0);c.strokeStyle='#d8dee8';c.fillStyle='#667085';c.font='12px system-ui';for(let i=0;i<=4;i++){const y=pad.t+ih*i/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();const label=(max*(1-i/4)).toFixed(3);c.fillText(label,4,y+4)}
if(threshold!==null){const y=pad.t+ih*(1-threshold/max);c.setLineDash([6,4]);c.strokeStyle='#e39a21';c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.setLineDash([])}
function line(values,color){c.strokeStyle=color;c.lineWidth=1.7;c.beginPath();let open=false;values.forEach((v,i)=>{if(v===null||!Number.isFinite(v)){open=false;return}const x=pad.l+(values.length<=1?0:i/(values.length-1))*iw,y=pad.t+ih*(1-v/max);if(!open)c.moveTo(x,y);else c.lineTo(x,y);open=true});c.stroke()}line(left,'#d84a45');line(right,'#3478d4');c.fillStyle='#667085';c.fillText('frame',w-46,h-7)}
function drawSeries(canvas,series,title){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);c.clearRect(0,0,w,h);const finite=series.flatMap(s=>s.values.filter(v=>v!==null&&Number.isFinite(v)));if(!finite.length)return;let min=Math.min(...finite),max=Math.max(...finite);const span=Math.max(max-min,1e-5);min-=span*.08;max+=span*.08;const pad={l:58,r:16,t:26,b:28},iw=w-pad.l-pad.r,ih=h-pad.t-pad.b;c.strokeStyle='#d8dee8';c.fillStyle='#667085';c.font='11px system-ui';for(let i=0;i<=4;i++){const y=pad.t+ih*i/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText((max-(max-min)*i/4).toFixed(3),4,y+4)}for(const s of series){c.strokeStyle=s.color;c.lineWidth=1.6;c.beginPath();let open=false;s.values.forEach((v,i)=>{if(v===null||!Number.isFinite(v)){open=false;return}const x=pad.l+(s.values.length<=1?0:i/(s.values.length-1))*iw,y=pad.t+(max-v)/(max-min)*ih;if(open)c.lineTo(x,y);else c.moveTo(x,y);open=true});c.stroke()}c.fillStyle='#182230';c.font='600 12px system-ui';c.fillText(title,8,16);c.fillStyle='#667085';c.font='11px system-ui';c.fillText('frame',w-46,h-7)}
function reachBand(id,values){const el=$(id);el.innerHTML='';if(!values.length)return;let start=0,state=values[0];for(let i=1;i<=values.length;i++){if(i===values.length||values[i]!==state){const s=document.createElement('span');s.className='segment '+(state?'ok':'bad');s.style.width=(100*(i-start)/values.length)+'%';s.title=`frames ${start}-${i-1}: ${state?'reachable':'unreachable'}`;el.appendChild(s);start=i;state=values[i]}}}
function drawTrajectory(canvas,frames,a,b,title){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);c.clearRect(0,0,w,h);const pad={l:52,r:18,t:34,b:38},points=[];for(const [prefix,color] of [['l','#d84a45'],['r','#3478d4']])frames.forEach((f,i)=>points.push({i,prefix,color,x:1000*f[prefix+a],y:1000*f[prefix+b],ok:f[prefix+'r']}));if(!points.length)return;let xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x)),ymin=Math.min(...points.map(p=>p.y)),ymax=Math.max(...points.map(p=>p.y));const dx=Math.max(xmax-xmin,1),dy=Math.max(ymax-ymin,1);xmin-=dx*.08;xmax+=dx*.08;ymin-=dy*.08;ymax+=dy*.08;const px=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),py=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);c.strokeStyle='#d8dee8';c.fillStyle='#667085';c.font='11px system-ui';for(let i=0;i<=4;i++){const x=pad.l+(w-pad.l-pad.r)*i/4,y=pad.t+(h-pad.t-pad.b)*i/4;c.beginPath();c.moveTo(x,pad.t);c.lineTo(x,h-pad.b);c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText((xmin+(xmax-xmin)*i/4).toFixed(0),x-8,h-12);c.fillText((ymax-(ymax-ymin)*i/4).toFixed(0),4,y+4)}c.fillStyle='#182230';c.font='600 12px system-ui';c.fillText(title,10,18);c.fillText(a.toUpperCase()+' (mm)',w-56,h-12);c.save();c.translate(13,h/2+20);c.rotate(-Math.PI/2);c.fillText(b.toUpperCase()+' (mm)',0,0);c.restore();for(const prefix of ['l','r']){const arm=points.filter(p=>p.prefix===prefix);c.strokeStyle=arm[0]?.color||'#000';c.globalAlpha=.28;c.beginPath();arm.forEach((p,i)=>(i?c.lineTo(px(p.x),py(p.y)):c.moveTo(px(p.x),py(p.y))));c.stroke();c.globalAlpha=1;for(const p of arm){const x=px(p.x),y=py(p.y);c.fillStyle=p.color;c.beginPath();c.arc(x,y,2.4,0,Math.PI*2);c.fill();if(!p.ok){c.strokeStyle='#111827';c.lineWidth=1;c.beginPath();c.moveTo(x-3,y-3);c.lineTo(x+3,y+3);c.moveTo(x-3,y+3);c.lineTo(x+3,y-3);c.stroke()}if(p.i<10){c.strokeStyle=p.i===0?'#111827':'#f0a51a';c.lineWidth=p.i===0?2:1.3;c.beginPath();c.arc(x,y,p.i===0?6:4.5,0,Math.PI*2);c.stroke();c.fillStyle='#7a4c00';c.font='10px system-ui';c.fillText(String(p.i),x+5,y-4)}}}}
const orbit={yaw:-.72,pitch:.42,zoom:1,drag:false,x:0,y:0};
function drawTrajectory3D(canvas,frames){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);c.clearRect(0,0,w,h);if(!frames.length)return;const points=[];for(const [prefix,color] of [['l','#d84a45'],['r','#3478d4']])frames.forEach((f,i)=>points.push({i,prefix,color,x:1000*f[prefix+'x'],y:1000*f[prefix+'y'],z:1000*f[prefix+'z'],ok:f[prefix+'r']}));const center={x:(Math.min(...points.map(p=>p.x))+Math.max(...points.map(p=>p.x)))/2,y:(Math.min(...points.map(p=>p.y))+Math.max(...points.map(p=>p.y)))/2,z:(Math.min(...points.map(p=>p.z))+Math.max(...points.map(p=>p.z)))/2};const span=Math.max(...['x','y','z'].map(k=>Math.max(...points.map(p=>p[k]))-Math.min(...points.map(p=>p[k]))),1);const cy=Math.cos(orbit.yaw),sy=Math.sin(orbit.yaw),cp=Math.cos(orbit.pitch),sp=Math.sin(orbit.pitch),scale=.78*Math.min(w,h)/span*orbit.zoom;function project(p){const x=p.x-center.x,y=p.y-center.y,z=p.z-center.z,x1=cy*x-sy*y,y1=sy*x+cy*y;return {x:w/2+x1*scale,y:h/2-(z*cp-y1*sp)*scale,d:y1*cp+z*sp}}function line(a,b,color,width=1){const p=project(a),q=project(b);c.strokeStyle=color;c.lineWidth=width;c.beginPath();c.moveTo(p.x,p.y);c.lineTo(q.x,q.y);c.stroke()}const axisLength=span*.34,origin=center;line(origin,{...center,x:center.x+axisLength},'#d84a45',2);line(origin,{...center,y:center.y+axisLength},'#2f9e62',2);line(origin,{...center,z:center.z+axisLength},'#3478d4',2);c.font='600 12px system-ui';for(const [label,p,color] of [['+X',{...center,x:center.x+axisLength},'#d84a45'],['+Y',{...center,y:center.y+axisLength},'#2f9e62'],['+Z',{...center,z:center.z+axisLength},'#3478d4']]){const q=project(p);c.fillStyle=color;c.fillText(label,q.x+4,q.y-4)}for(const prefix of ['l','r']){const arm=points.filter(p=>p.prefix===prefix);c.strokeStyle=arm[0]?.color||'#000';c.globalAlpha=.25;c.lineWidth=1.2;c.beginPath();arm.forEach((p,i)=>{const q=project(p);i?c.lineTo(q.x,q.y):c.moveTo(q.x,q.y)});c.stroke();c.globalAlpha=1}points.sort((a,b)=>project(a).d-project(b).d);for(const p of points){const q=project(p);c.fillStyle=p.color;c.beginPath();c.arc(q.x,q.y,2.5,0,Math.PI*2);c.fill();if(!p.ok){c.strokeStyle='#111827';c.lineWidth=1;c.beginPath();c.moveTo(q.x-3.5,q.y-3.5);c.lineTo(q.x+3.5,q.y+3.5);c.moveTo(q.x-3.5,q.y+3.5);c.lineTo(q.x+3.5,q.y-3.5);c.stroke()}if(p.i<10){c.strokeStyle=p.i===0?'#111827':'#f0a51a';c.lineWidth=p.i===0?2:1.2;c.beginPath();c.arc(q.x,q.y,p.i===0?6:4.5,0,Math.PI*2);c.stroke()}}c.fillStyle='#667085';c.font='11px system-ui';c.fillText('base_link / orthographic',12,h-12)}
const view3d=$('trajectory3D');view3d.addEventListener('pointerdown',e=>{orbit.drag=true;orbit.x=e.clientX;orbit.y=e.clientY;view3d.classList.add('dragging');view3d.setPointerCapture(e.pointerId)});view3d.addEventListener('pointermove',e=>{if(!orbit.drag)return;orbit.yaw+=(e.clientX-orbit.x)*.008;orbit.pitch=Math.max(-1.45,Math.min(1.45,orbit.pitch+(e.clientY-orbit.y)*.008));orbit.x=e.clientX;orbit.y=e.clientY;render()});function stopDrag(){orbit.drag=false;view3d.classList.remove('dragging')}view3d.addEventListener('pointerup',stopDrag);view3d.addEventListener('pointercancel',stopDrag);view3d.addEventListener('wheel',e=>{e.preventDefault();orbit.zoom=Math.max(.35,Math.min(4,orbit.zoom*Math.exp(-e.deltaY*.001)));render()},{passive:false});view3d.addEventListener('dblclick',()=>{orbit.yaw=-.72;orbit.pitch=.42;orbit.zoom=1;render()});
function armStats(frames,prefix){const reach=frames.filter(x=>x[prefix+'r']);const iso=reach.map(x=>x[prefix+'i']).filter(x=>x!==null),margin=reach.map(x=>x[prefix+'m']).filter(x=>x!==null);return {n:reach.length,rate:reach.length/frames.length,iso:iso.length?Math.min(...iso):null,margin:margin.length?Math.min(...margin):null,p:Math.max(...frames.map(x=>x[prefix+'p'])),o:Math.max(...frames.map(x=>x[prefix+'o']))}}
function render(){const frames=DATA.frames[strategy.value][episode.value]||[],s=DATA.strategies[strategy.value];$('bothRate').textContent=pct(s.both_reachable_rate);$('frameCount').textContent=frames.length;reachBand('leftReach',frames.map(x=>x.lr));reachBand('rightReach',frames.map(x=>x.rr));drawTrajectory3D($('trajectory3D'),frames);drawTrajectory($('trajectoryXY'),frames,'x','y','俯视投影 X-Y');drawTrajectory($('trajectoryXZ'),frames,'x','z','侧视投影 X-Z');const mobile=frames.some(x=>x.by!==null);$('mobilityPanel').style.display=mobile?'block':'none';if(mobile){drawSeries($('basePosition'),[{values:frames.map(x=>x.by),color:'#7c3aed'}],'底盘前后位置 table-Y (m)');drawSeries($('torsoAngles'),[{values:frames.map(x=>x.kn),color:'#2f9e62'},{values:frames.map(x=>x.hp),color:'#e39a21'},{values:frames.map(x=>x.hy),color:'#8b5cf6'}],'膝与腰部关节角 (rad)')}draw($('isotropy'),frames.map(x=>x.li),frames.map(x=>x.ri),DATA.thresholds.isotropy_ratio_low);draw($('margin'),frames.map(x=>x.lm),frames.map(x=>x.rm),null);$('episodeSummary').innerHTML=[['左臂','l'],['右臂','r']].map(([label,p])=>{const a=armStats(frames,p);const fmt=v=>v===null?'N/A':v.toFixed(5);return `<tr><th>${label}</th><td>${a.n}/${frames.length}</td><td>${pct(a.rate)}</td><td>${fmt(a.iso)}</td><td>${fmt(a.margin)}</td><td>${a.p.toFixed(2)} mm</td><td>${a.o.toFixed(2)}°</td></tr>`}).join('')}
strategy.addEventListener('change',refillEpisodes);episode.addEventListener('change',render);window.addEventListener('resize',render);refillEpisodes();
</script></body></html>
"""
