# Servia server

The Python side of Servia: the API, the LiveKit agent worker and the jobs worker.

```
POST /v1/calls  ──▶  call row ──▶  LiveKit token with {call_id}
                                          │
browser joins ────────────────────────────┤
                                          ▼
                              worker reads call_id
                              context pack + the call's release
                              prompt compiled
                              AgentSession (STT → LLM → TTS)
                                          │
                              hang-up ──▶ jobs worker: summary, memory
```

What the agent knows and how it behaves comes from data, not code: the organization,
its skills, tools, knowledge and rules, and the caller's services and open work.
A call runs on a release, a frozen copy of that configuration.

## Run it

Commands run from the `servia/` folder, one level up.

```bash
cp server/.env.example server/.env   # fill in the LiveKit keys
make up                              # everything in docker, watches code
make down
```

Open http://localhost:8000, pick a customer, start a call. 

On the host instead:

```bash
make db
make install migrate seed
make api        # terminal 1
make agent      # terminal 2
make jobs       # terminal 3
```

Talk to the agent by typing, no audio:

```bash
cd server && uv run python -m app.text_call <user_id> "Hi" "Can someone come out on Thursday?"
```

Tests: `make test` (needs postgres on 5433 and redis on 6379).

After changing skills, tools or knowledge, build a release and promote it, or calls
keep using the old configuration. See `POST /v1/releases`.

## Layout

| | |
|---|---|
| `app/core/` | settings, enums, database, cache, container |
| `app/modules/accounts/` | organizations, customers, staff |
| `app/modules/catalog/` | services and enrollments |
| `app/modules/work/` | tasks and work items |
| `app/modules/calls/` | calls, turn records, tool journal, reports |
| `app/modules/context/` | builds the context pack before a call |
| `app/modules/config/` | skills, tool bindings, model profile, routes, switches |
| `app/modules/knowledge/` | knowledge publishing and search |
| `app/modules/memory/` | memory across calls |
| `app/modules/detection/` | caller blocks, review items, detection rules |
| `app/modules/releases/` | releases, canary, rollback |
| `app/modules/evaluation/` | scenarios, suites, annotations, analytics |
| `app/modules/jobs/` | the job queue |
| `app/agent/` | the conversation agent, prompt, skills, tools, handoff |
| `app/voice/` | LiveKit token, dispatch and the agent worker |
| `app/analysis/` | after-call pipeline and outcome analytics |
| `app/eval/` | replaying calls to test a release |
| `app/seed.py` | demo data (Northwind) |

`app/modules/context/assembler.py` is where everything the agent knows about a caller
comes together before the call starts.
