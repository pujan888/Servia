# Servia

A realtime AI customer-support voice agent.

| Folder | What it holds |
| --- | --- |
| `server/` | The Python: the API, the LiveKit agent worker, the jobs worker, tests and migrations. See `server/README.md` |
| `web/` | The frontend
| `docker/` | Image definitions for shared services (Postgres with pgvector) |
| `docker-compose.local.yml` | Local: Postgres, Redis and the server containers |
| `docker-compose.yml` | The same, for the server we deploy to |
| `Makefile` | Shortcuts; host commands run inside `server/` |

## Run it on your machine

```bash
cp server/.env.example server/.env   # fill in the LiveKit keys
make db
make install migrate seed
make api      # terminal 1
make agent    # terminal 2
make jobs     # terminal 3
```

Or everything in Docker: `make up`.
