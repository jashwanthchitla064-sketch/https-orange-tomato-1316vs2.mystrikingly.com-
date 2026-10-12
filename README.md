# FieldHealth (MVP)
Upload drone images of a field -> Green/Yellow/Red crop-health map, percentages, problem zones, PDF + CSV.

## Run
`docker compose up --build` -> http://localhost:8000   (or `cd backend && pip install -r requirements.txt && uvicorn main:app`)
Tests: `cd backend && python -m pytest tests`

## How it works
1. **Index**: RGB -> VARI `(G-R)/(G+R-B)` (default) or ExG; 4-band TIFF (R,G,B,NIR) -> NDVI `(NIR-R)/(NIR+R)`. Clipped to [-1,1].
2. **Mask**: shadows (very dark), water (blue-dominant) and gray pixels (roads) are dropped. Bare soil is kept, so bare patches score Red. Blocks with <30% crop pixels are excluded from the percentages.
3. **Grid**: mean index per block (default 20x20 px, `block` setting).
4. **Classify**: Green >= 0.6, Yellow 0.3-0.6, Red < 0.3. Percentages are over classified blocks and sum to 100.
5. Multiple images: analyzed separately; block counts are summed (area-weighted). No stitching yet.

## Tuning
`PUT /api/settings {"green":0.5,"yellow":0.2,"block":30,"index":"ExG"}`; `DELETE /api/settings` resets. VARI on real RGB crops is typically 0.1-0.5, so lower thresholds (e.g. 0.35/0.15) often fit better; adjust per crop and camera, then re-upload.

## API
`POST /api/fields`, `GET /api/fields`, `POST /api/fields/{id}/surveys` (multipart, background job), `GET /api/surveys/{id}` (status+result), `/overlay`, `/original`, `/csv`, `/report.pdf`, `GET /api/fields/{id}/surveys` (trend data).
