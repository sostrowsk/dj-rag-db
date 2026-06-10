# dj-rag-db

Django app package `scribe` (app label, import path, DB tabellen bleiben
`scribe`). Host-Projekte pinnen dieses Repo als Poetry-git-Dependency auf
`main` — jeder Push auf main ist sofort releasebar.

## TDD-Regeln (Pflicht)

- **Test zuerst, RED bestaetigen, dann implementieren, GREEN bestaetigen.**
- Bugfix = Regressionstest, der den Bug reproduziert und VOR dem Fix failt.
- Reine Moves: Import-Smoke-Tests.
- Tests laufen aus dem Host-Projekt: `pytest --pyargs scribe.tests`
  (das Package hat keine eigene Settings-/pytest-Infrastruktur).
- LLM-/Netzwerk-/Milvus-Calls IMMER mocken (`scribe/tests/mocks.py`) —
  kein Test darf echte Provider-APIs oder einen echten Milvus treffen.

## Architektur-Regeln

- Keine Imports aus Host-Apps (users, project, leasing, ai_agents,
  data_room, ...). Host-Models/-Tasks NUR lazy ueber `scribe/conf.py`
  (`SCRIBE_PROJECT_DOCUMENT_MODEL`, `SCRIBE_CLIENT_DOCUMENT_MODEL`,
  `SCRIBE_INDEX_DOCUMENT_TASK`).
- Peer-Apps `ai_router` und `progress` duerfen direkt importiert werden —
  sie sind System-Check-gesichert (`scribe.E001`/`E002` in `scribe/apps.py`),
  aber NICHT in pyproject deklariert (nur der Host pinnt dj-* Packages).
- **Migrations-Byte-Stabilitaet:** Aenderungen duerfen keine neuen
  Migrationen im Host erzeugen (`makemigrations --check --dry-run` muss im
  Host clean bleiben). Modul-Level-Settings-FKs nicht "dynamisieren".
- Settings-Katalog im README aktuell halten, wenn neue Settings dazukommen.
