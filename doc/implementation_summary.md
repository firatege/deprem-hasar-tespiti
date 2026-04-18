# Proje Ozeti (Su Ana Kadar)

Bu dosya, su ana kadar projede ne yaptigimizi tek yerde toplar.

## 1) Ne Kuruldu?

- Backend omurgasi: `FastAPI`
- Asenkron is akisi: `RQ + Redis`
- Job state machine: `queued -> running -> succeeded|failed` ve `cancel`
- Veri katmani: `SQLite index + file cache`
- Veri guvenligi: `integrity check` + `soft-lock` + opsiyonel `batch quarantine`
- Etiket politikasi: `Major+ => damaged (binary=1)`
- Konteyner runtime: `Docker + docker compose`

## 2) API Yapisi

### Health
- `GET /health`

### Jobs
- `POST /jobs/submit`
- `GET /jobs/status/{job_id}`
- `GET /jobs/result/{job_id}`
- `POST /jobs/{job_id}/cancel`

### Data
- `POST /data/integrity/run?batch_move=true|false`

## 3) Asenkron Akis (Kisa)

1. Client `POST /jobs/submit` ile is gonderir.
2. API `jobs` tablosuna kaydi `queued` olarak yazar.
3. Job, Redis queue (`tile-jobs`) uzerine enqueue edilir.
4. Worker job'i alip `running` yapar.
5. `process_tile_job` sonucu hesaplar, cache'e yazar, DB durumunu `succeeded` yapar.
6. Client `GET /jobs/status/{id}` ile takip eder, `GET /jobs/result/{id}` ile sonucu alir.

## 4) DB ve Depolama

### SQLite tablolari (aktif kullanim)
- `jobs`
- `tiles`
- `integrity_events`

### Tile alanlari (onemli)
- `tile_id`, `file_path`
- `source_version`, `pga_value`, `binary_label`
- `is_soft_locked`, `flagged_for_fix`, `lock_reason`

### Disk yapisi
- `storage/tiles`: tile dosyalari
- `storage/cache`: is sonucu json
- `storage/quarantine`: batch tasinan problemli dosyalar

## 5) Integrity (Orphan/Bozuk Veri) Mantigi

Kontrol edilenler:
- DB'de var ama diskte yok (`missing_on_disk`) -> soft-lock + event
- Diskte 0-byte dosya (`zero_byte_on_disk`) -> soft-lock + event
- Diskte var ama index'te yok (`missing_in_index`) -> rapor + aday quarantine

Batch move aciksa:
- `missing_in_index` ve `zero_byte` adaylari `quarantine` klasorune tasinir.

CLI:
- `data/integrity_check.py`

## 6) Docker / Compose Yapisi

Servisler:
- `api` (`:8000`)
- `worker`
- `redis` (`:6379`)

Kalicilik:
- `sqlite_data` -> `/app/data`
- `storage_data` -> `/app/storage`
- `redis_data` -> `/data`

Calisma modu notu:
- Compose ortaminda `SYNC_FALLBACK_WHEN_QUEUE_UNAVAILABLE=false` (strict async)

## 7) "Smoke" Nedir? (Ivir Zivir Degil, Kisa E2E Kontrol)

`smoke` burada su isi yapiyor:
- API ayakta mi?
- Job submit olabiliyor mu?
- Worker job'i tuketip `succeeded` yapabiliyor mu?
- Sonuc endpoint'i dogru donuyor mu?

Yani uzun test degil, "sistem minimum calisiyor mu" kontrolu.

Scriptler:
- `scripts/dev-compose.ps1` (`up/down/ps/logs/smoke`)
- `scripts/smoke-compose.ps1` (hazirlik + submit + poll + result)

## 8) Cozulen Kritik Sorunlar

1. Worker restart loop:
- Sebep: `rq.Connection` import uyumsuzlugu
- Cozum: `Worker(..., connection=redis_conn)` kullanimina gecildi
- Dosya: `app/workers/rq_worker.py`

2. Smoke readiness kirmasi:
- Sebep: readiness endpoint/tek-shot deneme
- Cozum: `/health` ile retry/polling eklendi
- Dosya: `scripts/smoke-compose.ps1`

3. Smoke preflight:
- Sebep: servislerden biri yokken anlamsiz timeout
- Cozum: `api/worker/redis` running kontrolu eklendi
- Dosya: `scripts/dev-compose.ps1`

4. Compose saglamlastirma:
- `version` anahtari kaldirildi
- healthcheck + restart + depends_on condition iyilestirildi
- Dosya: `docker-compose.yml`

## 9) Hangi Dosyalar Omurga?

- `app/main.py`
- `app/api/jobs.py`
- `app/api/data.py`
- `app/api/health.py`
- `app/queue.py`
- `app/workers/tasks.py`
- `app/workers/rq_worker.py`
- `app/db/repository.py`
- `app/db/schema.sql`
- `app/services/integrity.py`
- `data/integrity_check.py`
- `docker-compose.yml`
- `Dockerfile`
- `scripts/dev-compose.ps1`
- `scripts/smoke-compose.ps1`

## 10) Hemen Baslat / Kapat

Baslat:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
./scripts/dev-compose.ps1 -Action smoke
```

Kapat:

```powershell
docker compose down
```

---

Guncelleme tarihi: 2026-04-18

