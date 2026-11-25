"""Ganguly Video Generator - a deployable website. Prompt + attachments -> AI video, any length.

Needs no GPU: the server calls the fal.ai LTX-2.5 video models (native audio included), chains
clips (each starts from the previous clip's last frame) and joins them with ffmpeg.

Deploy (Render, Railway, Fly, any VPS):
  pip install -r requirements.txt
  env:  GEMINI_API_KEY=your_google_key (Google Veo)  or  FAL_KEY=your_fal_key,  plus SITE_PASSWORD=choose_a_password
  start: gunicorn app:app -w 1 --threads 16 -t 600 --bind 0.0.0.0:$PORT
Local test: FAL_KEY=... python app.py  ->  http://127.0.0.1:7860

requirements.txt:  flask  gunicorn  requests  pillow  imageio-ffmpeg
"""
import os, re, io, json, time, base64, threading, subprocess, uuid
import requests, imageio_ffmpeg
from PIL import Image
from flask import Flask, request, jsonify, send_file, abort, Response
from werkzeug.utils import secure_filename

KEY, PW = os.environ.get("FAL_KEY", ""), os.environ.get("SITE_PASSWORD", "")
GKEY = os.environ.get("GEMINI_API_KEY", "")                                 # Google Veo via the Gemini API
VEO = os.environ.get("GVG_VEO", "veo-3.1-generate-preview")
USE_G = bool(GKEY) and (os.environ.get("GVG_PROVIDER") == "gemini" or not KEY)  # Gemini if it is the only key
T2V = os.environ.get("GVG_T2V", "lightricks/ltx-2.5/text-to-video/fast")   # model ids change often:
I2V = os.environ.get("GVG_I2V", "lightricks/ltx-2.5/image-to-video/fast")   # override here, no code edit
EXTRA = json.loads(os.environ.get("GVG_EXTRA", "{}"))                       # e.g. {"resolution":"1080p"}
OUT = os.path.abspath(os.environ.get("VIDEOS_DIR", "ganguly_videos")); os.makedirs(OUT, exist_ok=True)
FF, ALLOWED = imageio_ffmpeg.get_ffmpeg_exe(), ((8,) if USE_G else (6, 8, 10))                    # clip lengths in seconds
EXT = dict(img={"png", "jpg", "jpeg", "webp"}, vid={"mp4", "mov", "webm", "mkv"},
           aud={"mp3", "wav", "m4a", "aac", "ogg"}, txt={"txt", "md"})
app, JOBS = Flask(__name__), {}
gate = threading.Semaphore(int(os.environ.get("GVG_PARALLEL", 4)))          # extra jobs wait, never refused


@app.before_request
def guard():                                       # protects your fal credit; unset SITE_PASSWORD to open the site
    if PW and (not request.authorization or request.authorization.password != PW):
        return Response("Password needed", 401, {"WWW-Authenticate": 'Basic realm="Ganguly Video Generator"'})


def ff(*a): subprocess.run([FF, "-y", "-loglevel", "error", *a], check=True)
def ok(r):
    if not r.ok: raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")
    return r.json()


def fal(model, **inp):                             # submit -> poll -> video url
    H = {"Authorization": "Key " + KEY}
    s = ok(requests.post(f"https://queue.fal.run/{model}", json={**inp, **EXTRA}, headers=H, timeout=120))
    for _ in range(400):
        time.sleep(3)
        if ok(requests.get(s["status_url"], headers=H, timeout=60)).get("status") == "COMPLETED": break
    else: raise RuntimeError("A clip took too long")
    return ok(requests.get(s["response_url"], headers=H, timeout=60))["video"]["url"]


def veo(prompt, image=None):                       # Google Veo through the Gemini API (8 s clips, with sound)
    H, B = {"x-goog-api-key": GKEY}, "https://generativelanguage.googleapis.com/v1beta"
    inst = {"prompt": prompt}
    if image: inst["image"] = {"bytesBase64Encoded": image.split(",", 1)[1], "mimeType": "image/jpeg"}
    op = ok(requests.post(f"{B}/models/{VEO}:predictLongRunning", json={"instances": [inst]}, headers=H, timeout=120))["name"]
    for _ in range(120):
        time.sleep(10); s = ok(requests.get(f"{B}/{op}", headers=H, timeout=60))
        if s.get("done"): break
    else: raise RuntimeError("A clip took too long")
    if "error" in s: raise RuntimeError(str(s["error"])[:200])
    try: return s["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
    except (KeyError, IndexError): raise RuntimeError("Google returned no video (the prompt may have been blocked)")


def last_frame(video, jpg):                        # last frame as a data URI for the next clip
    ff("-sseof", "-1", "-i", video, "-update", "1", "-q:v", "2", jpg)
    return "data:image/jpeg;base64," + base64.b64encode(open(jpg, "rb").read()).decode()


def img_uri(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((1280, 1280)); b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def plan(sec):
    out = []
    while sec > 0.01:
        c = min(sec, ALLOWED[-1]); out.append(next(a for a in ALLOWED if a >= c)); sec -= c
    return out


def run(j):
    d, scenes = j["dir"], [s.strip() for s in re.split(r"\n\s*\n", j["prompt"]) if s.strip()]
    plans = [plan(j["seconds"] / len(scenes)) for _ in scenes]
    total, n, parts, stop = sum(map(len, plans)), 0, [], False
    start = last_frame(j["video"], d + "/v.jpg") if j["video"] else None
    pro = lambda m: m.replace("/fast", "/pro") if j["tier"] == "pro" else m
    for si, (sc, pl) in enumerate(zip(scenes, plans)):
        if si < len(j["images"]): start = img_uri(j["images"][si])
        for ci, sec in enumerate(pl):
            if j["cancel"]: stop = True; break
            j["msg"] = f"scene {si + 1} of {len(scenes)}, clip {ci + 1} of {len(pl)}"
            if USE_G: url = veo(sc, start)
            else: url = fal(pro(I2V), prompt=sc, image_url=start, duration=sec) if start else fal(pro(T2V), prompt=sc, duration=sec)
            p = f"{d}/c{n}.mp4"
            with requests.get(url, stream=True, timeout=300, headers={"x-goog-api-key": GKEY} if USE_G else {}) as r, open(p, "wb") as f:
                for chunk in r.iter_content(1 << 20): f.write(chunk)
            parts.append(p); start = last_frame(p, f"{d}/l{n}.jpg"); n += 1; j["progress"] = n / total
        if stop: break
    if not parts: raise RuntimeError("Stopped before the first clip finished")
    open(d + "/list.txt", "w").write("".join(f"file '{p}'\n" for p in parts))
    a = ["-i", j["audio"], "-map", "0:v", "-map", "1:a", "-shortest"] if j["audio"] else []
    if a: a = ["-stream_loop", "-1"] + a
    ff("-f", "concat", "-safe", "0", "-i", d + "/list.txt", *a, "-c:v", "libx264", "-preset", "veryfast",
       "-pix_fmt", "yuv420p", "-c:a", "aac", f"{OUT}/{j['id']}.mp4")
    j["status"], j["msg"] = "done", "stopped early" if stop else ""


def job(j):
    with gate:
        try: j["status"] = "running"; run(j)
        except Exception as e: j["status"], j["msg"] = "error", str(e)[:300]


@app.post("/generate")
def generate():
    if not (KEY or GKEY): return Response("The site owner has not set FAL_KEY or GEMINI_API_KEY yet.", 500)
    f, jid = request.form, uuid.uuid4().hex[:12]
    d = f"{OUT}/work_{jid}"; os.makedirs(d)
    j = dict(id=jid, cid=f.get("cid", ""), dir=d, prompt=f.get("prompt", "").strip(), tier=f.get("tier", "fast"),
             seconds=max(1.0, float(f.get("seconds") or 10)), images=[], video=None, audio=None,
             status="queued", progress=0.0, msg="", cancel=False)
    for u in request.files.getlist("files"):
        n = secure_filename(u.filename or "")
        if not n: continue
        p = f"{d}/{n}"; u.save(p); e = n.rsplit(".", 1)[-1].lower()
        if e in EXT["img"]: j["images"].append(p)
        elif e in EXT["vid"] and not j["video"]: j["video"] = p
        elif e in EXT["aud"] and not j["audio"]: j["audio"] = p
        elif e in EXT["txt"]: j["prompt"] += "\n\n" + open(p, errors="ignore").read()
    if not j["prompt"].strip():
        if not (j["images"] or j["video"]): return Response("Write a prompt or attach a file.", 400)
        j["prompt"] = "Smooth, natural, cinematic motion continuing this scene"
    JOBS[jid] = j; threading.Thread(target=job, args=(j,), daemon=True).start()
    return jsonify(id=jid)


@app.get("/jobs")
def jobs():
    c = request.args.get("cid", "")
    return jsonify([{k: j[k] for k in ("id", "prompt", "status", "progress", "msg")} for j in JOBS.values() if j["cid"] == c])


@app.post("/cancel/<jid>")
def cancel(jid):
    if jid in JOBS: JOBS[jid]["cancel"] = True
    return "", 204


@app.get("/video/<jid>")
def video(jid):
    p = f"{OUT}/{jid}.mp4"
    if not re.fullmatch(r"[0-9a-f]{12}", jid) or not os.path.exists(p): abort(404)
    return send_file(p, mimetype="video/mp4", conditional=True)


@app.get("/")
def home(): return Response(HTML, mimetype="text/html")


HTML = r"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Ganguly Video Generator</title>
<style>
:root{--bg:#e8ecef;--ink:#0f1b26;--mut:#5a6874;--blue:#2b4a7a;--line:#c5cdd4;--amber:#d99a00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
header{background:var(--ink);color:#fff;padding:18px 28px;border-bottom:6px solid var(--amber)}
h1{margin:0;font:700 clamp(1.5rem,3vw,2.2rem)/1.1 Georgia,serif}header p{margin:4px 0 0;color:#aab6c1;font-size:.95rem}
main{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:28px;padding:28px;max-width:1200px;margin:auto}
@media(max-width:800px){main{grid-template-columns:1fr;padding:16px}}
form{display:grid;gap:12px;align-content:start}label{display:grid;gap:4px;font-weight:600;font-size:.9rem}
textarea,input,select{font:inherit;padding:9px 10px;border:1px solid var(--line);background:#fff;border-radius:3px;color:var(--ink)}
textarea{min-height:150px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}small{color:var(--mut);font-weight:400}
button{font:600 1rem system-ui;padding:11px 16px;border:0;border-radius:3px;background:var(--blue);color:#fff;cursor:pointer}
button:disabled{opacity:.5}button.s{background:none;color:var(--mut);padding:4px 0;text-decoration:underline;font-size:.85rem}
:focus-visible{outline:3px solid var(--amber);outline-offset:2px}
#jobs{display:grid;gap:16px;align-content:start}.job{border-left:6px solid var(--blue);padding:4px 0 4px 14px}
.job b{display:block;font-weight:600}.job p{margin:2px 0 6px;color:var(--mut);font-size:.9rem}
progress{width:100%;height:8px;accent-color:var(--amber)}video{width:100%;max-width:640px;background:#000;display:block;margin:6px 0}
.empty{color:var(--mut);border:2px dashed var(--line);padding:28px}
</style>
<header><h1>Ganguly Video Generator</h1><p>Describe it, attach anything that helps, and make videos of any length, as many as you like.</p></header>
<main>
<form id=f>
<label>Prompt <small>Leave a blank line between scenes. Length is shared across scenes.</small>
<textarea name=prompt placeholder="A fishing boat leaves a Kolkata ghat at dawn, mist on the Hooghly, warm light on the water."></textarea></label>
<label>Attachments <small>Images (scene start frames), a video to continue, audio, or a .txt script</small>
<input type=file name=files multiple accept="image/*,video/*,audio/*,.txt,.md"></label>
<div class=row><label>Length (seconds)<input type=number name=seconds value=10 min=1 step=any></label>
<label>Quality<select name=tier><option value=fast selected>Fast<option value=pro>Best</select></label></div>
<button id=go>Make video</button></form>
<section id=jobs><div class=empty id=none>Your videos appear here. Start as many as you like.</div></section>
</main>
<script>
const $=s=>document.querySelector(s),cid=localStorage.gvg||(localStorage.gvg=crypto.randomUUID());
$('#f').onsubmit=async e=>{e.preventDefault();const b=$('#go');b.disabled=true;const d=new FormData(e.target);d.append('cid',cid);
 const r=await fetch('/generate',{method:'POST',body:d});if(!r.ok)alert(await r.text());b.disabled=false;poll()};
async function poll(){
 for(const j of await (await fetch('/jobs?cid='+cid)).json()){
  $('#none')?.remove();let c=document.getElementById('j'+j.id);
  if(!c){c=document.createElement('div');c.id='j'+j.id;c.className='job';
   c.innerHTML='<b></b><p></p><progress max=1></progress><div></div><button type=button class=s>Stop and keep what is made</button>';
   c.querySelector('b').textContent=j.prompt.slice(0,100);
   c.querySelector('button').onclick=()=>fetch('/cancel/'+j.id,{method:'POST'});$('#jobs').prepend(c)}
  c.querySelector('p').textContent=j.status+(j.msg?', '+j.msg:'');c.querySelector('progress').value=j.progress;
  if(j.status=='done'&&!c.querySelector('video'))c.children[3].innerHTML=`<video controls src=/video/${j.id}></video><a download href=/video/${j.id}>Download</a>`;
  if(j.status=='done'||j.status=='error')c.querySelector('button')?.remove()}}
setInterval(poll,2500);poll();
</script></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 7860)), threaded=True)
