"""FieldHealth API. Run: uvicorn main:app. Analysis runs as a BackgroundTask; poll /api/surveys/{id}."""
import csv, io, json, shutil, sqlite3
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import analysis as A

DATA = Path("data"); DATA.mkdir(exist_ok=True)
db = sqlite3.connect(DATA / "app.db", check_same_thread=False); db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS fields(id INTEGER PRIMARY KEY, name, crop, area REAL, unit, lat REAL, lon REAL);
CREATE TABLE IF NOT EXISTS surveys(id INTEGER PRIMARY KEY, field_id INT, date, status, error, result);
CREATE TABLE IF NOT EXISTS settings(k PRIMARY KEY, v);""")
ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RECS = {"green": "Healthy: continue routine care.",
        "yellow": "Monitor: re-inspect within 7 days.",
        "red": "Check irrigation, pests/disease and nutrient deficiency; scout these zones now."}
app = FastAPI(title="FieldHealth")


def settings():
    s = dict(A.DEFAULTS)
    for r in db.execute("SELECT * FROM settings"):
        s[r["k"]] = json.loads(r["v"])
    return s


def gps(path):
    try:
        g = Image.open(path).getexif().get_ifd(0x8825)
        dms = lambda t, ref: (float(t[0]) + float(t[1]) / 60 + float(t[2]) / 3600) * (-1 if ref in "SW" else 1)
        return dms(g[2], g[1]), dms(g[4], g[3])
    except Exception:
        return None


@app.post("/api/fields")
def create_field(name: str = Form(...), crop: str = Form(""), area: float = Form(...),
                 unit: str = Form("acres"), lat: float = Form(None), lon: float = Form(None)):
    c = db.execute("INSERT INTO fields(name,crop,area,unit,lat,lon) VALUES(?,?,?,?,?,?)", (name, crop, area, unit, lat, lon))
    db.commit(); return {"id": c.lastrowid}


@app.get("/api/fields")
def list_fields():
    return [dict(r) for r in db.execute("SELECT * FROM fields")]


@app.get("/api/fields/{fid}/surveys")
def field_surveys(fid: int):
    out = []
    for r in db.execute("SELECT * FROM surveys WHERE field_id=? ORDER BY date", (fid,)):
        res = json.loads(r["result"] or "{}")
        out.append({"id": r["id"], "date": r["date"], "status": r["status"], "pct": res.get("pct")})
    return out  # feed to a trend chart: date vs pct.green


@app.post("/api/fields/{fid}/surveys")
async def upload(fid: int, bg: BackgroundTasks, date: str = Form(...), files: list[UploadFile] = File(...)):
    bad = [f.filename for f in files if Path(f.filename).suffix.lower() not in ALLOWED]
    if bad:
        raise HTTPException(400, f"Unsupported file type: {', '.join(bad)} (use JPG/PNG/TIFF)")
    sid = db.execute("INSERT INTO surveys(field_id,date,status) VALUES(?,?,'queued')", (fid, date)).lastrowid
    db.commit()
    d = DATA / "surveys" / str(sid); d.mkdir(parents=True)
    for i, f in enumerate(files):
        with open(d / f"{i}{Path(f.filename).suffix.lower()}", "wb") as o:
            shutil.copyfileobj(f.file, o)
    bg.add_task(run, sid, fid)
    return {"id": sid}


def run(sid, fid):
    db.execute("UPDATE surveys SET status='running' WHERE id=?", (sid,)); db.commit()
    s, d = settings(), DATA / "surveys" / str(sid)
    counts, rows, zones, warns, coords = [0, 0, 0], [], [], [], None
    try:
        for i, p in enumerate(sorted(x for x in d.iterdir() if x.stem.isdigit())):
            try:
                rgb, lab, means, pct, w = A.analyze_image(p, s)
            except A.AnalysisError as e:
                warns.append(f"{p.name}: {e}"); continue
            warns += [f"{p.name}: {x}" for x in w]
            A.render_overlay(rgb, lab, s["block"]).save(d / f"overlay_{p.stem}.png")
            for k in range(3):
                counts[k] += int((lab == k).sum())  # block counts => area-weighted combination
            zones += [{**z, "image": p.stem} for z in A.problem_zones(lab, int(s["block"]))]
            coords = coords or gps(p)
            for (r, c), v in __import__("numpy").ndenumerate(lab):
                rows.append([p.stem, r, c, c * s["block"], r * s["block"], round(float(means[r, c]), 4), ["green", "yellow", "red", "excluded"][v if v >= 0 else 3]])
        tot = sum(counts)
        if tot == 0:
            raise A.AnalysisError("; ".join(warns) or "No vegetation detected.")
        f = dict(db.execute("SELECT * FROM fields WHERE id=?", (fid,)).fetchone())
        pct = [c / tot * 100 for c in counts]
        res = {"pct": dict(zip(["green", "yellow", "red"], pct)),
               "area": {k: f["area"] * p / 100 for k, p in zip(["green", "yellow", "red"], pct)},
               "unit": f["unit"], "zones": sorted(zones, key=lambda z: -z["blocks"])[:5],
               "warnings": warns, "gps": coords, "settings": s}
        (d / "regions.csv").write_text("\n".join(",".join(map(str, r)) for r in [["image", "row", "col", "x", "y", "mean_index", "class"]] + rows))
        db.execute("UPDATE surveys SET status='done',result=? WHERE id=?", (json.dumps(res), sid))
        if coords and f["lat"] is None:
            db.execute("UPDATE fields SET lat=?,lon=? WHERE id=?", (*coords, fid))
    except A.AnalysisError as e:
        db.execute("UPDATE surveys SET status='failed',error=? WHERE id=?", (str(e), sid))
    except Exception as e:
        db.execute("UPDATE surveys SET status='failed',error=? WHERE id=?", (f"Unexpected error: {e}", sid))
    db.commit()


@app.get("/api/surveys/{sid}")
def survey(sid: int):
    r = db.execute("SELECT * FROM surveys WHERE id=?", (sid,)).fetchone()
    if not r: raise HTTPException(404)
    return {"id": sid, "status": r["status"], "error": r["error"], "result": json.loads(r["result"] or "null"), "recommendations": RECS}


@app.get("/api/surveys/{sid}/overlay")
def overlay(sid: int, i: int = 0): return FileResponse(DATA / "surveys" / str(sid) / f"overlay_{i}.png")


@app.get("/api/surveys/{sid}/original")
def original(sid: int, i: int = 0):
    return FileResponse(next((DATA / "surveys" / str(sid)).glob(f"{i}.*")))


@app.get("/api/surveys/{sid}/csv")
def csv_export(sid: int): return FileResponse(DATA / "surveys" / str(sid) / "regions.csv", filename=f"survey_{sid}.csv")


@app.get("/api/surveys/{sid}/report.pdf")
def report(sid: int):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    r = db.execute("SELECT s.*, f.name, f.crop FROM surveys s JOIN fields f ON f.id=s.field_id WHERE s.id=?", (sid,)).fetchone()
    if not r or r["status"] != "done": raise HTTPException(409, "Survey not ready")
    res, buf = json.loads(r["result"]), io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4); y = 800
    def line(t, size=11):
        nonlocal y; c.setFont("Helvetica", size); c.drawString(50, y, t); y -= size + 6
    line(f"FieldHealth report: {r['name']} ({r['crop']})", 16); line(f"Survey date: {r['date']}"); y -= 6
    for k in ("green", "yellow", "red"):
        line(f"{k.title()}: {res['pct'][k]:.1f}%  ({res['area'][k]:.2f} {res['unit']}) - {RECS[k]}")
    c.drawImage(str(DATA / "surveys" / str(sid) / "overlay_0.png"), 50, y - 300, width=300, height=300, preserveAspectRatio=True, mask="auto"); y -= 320
    line("Top problem zones:", 13)
    for z in res["zones"]: line(f"image {z['image']}: x={z['x']} y={z['y']} size {z['w']}x{z['h']}px, {z['blocks']} blocks")
    c.save()
    return Response(buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_{sid}.pdf"})


@app.get("/api/settings")
def get_settings(): return settings()


@app.put("/api/settings")
def put_settings(body: dict):
    for k in A.DEFAULTS:
        if k in body: db.execute("REPLACE INTO settings VALUES(?,?)", (k, json.dumps(body[k])))
    db.commit(); return settings()


@app.delete("/api/settings")  # reset to defaults
def reset_settings(): db.execute("DELETE FROM settings"); db.commit(); return settings()


app.mount("/", StaticFiles(directory="static", html=True))
