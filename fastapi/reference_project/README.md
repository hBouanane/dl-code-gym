# Reference project: Library API (hexagonal / clean architecture)

A small but complete FastAPI service with books and loans. It's structured the way real
production codebases are. Read it, then **rebuild it from an empty folder** (see the bottom of this page).

## Structure

```
reference_project/
├── app/
│   ├── domain/                  # 1. the core: entities + business rules. Imports NOTHING external
│   │   ├── models.py            #    Book entity (borrow / give_back rules)
│   │   └── errors.py            #    BookNotFound, BookAlreadyBorrowed, ...
│   ├── application/             # 2. use cases + ports (interfaces the core needs)
│   │   ├── ports.py             #    BookRepository, Clock (typing.Protocol)
│   │   └── services.py          #    LibraryService: add_book, borrow, give_back, list_books
│   ├── adapters/
│   │   ├── inbound/http/        # 3a. driving adapter: FastAPI turns HTTP into use-case calls
│   │   │   ├── schemas.py       #     Pydantic request/response models (NOT the domain models)
│   │   │   ├── routes.py        #     APIRouter, thin: parse -> call service -> return schema
│   │   │   ├── dependencies.py  #     Depends() helpers to get the service
│   │   │   └── errors.py        #     domain error -> HTTP status mapping, in ONE place
│   │   └── outbound/            # 3b. driven adapters: implementations of the ports
│   │       ├── memory.py        #     InMemoryBookRepository (tests, demos)
│   │       ├── sqlalchemy_repo.py  #  SqlAlchemyBookRepository + ORM table (production)
│   │       └── clock.py         #     SystemClock
│   ├── config.py                # pydantic-settings Settings (env vars, prefix LIB_)
│   └── main.py                  # 4. composition root: create_app() wires everything together
├── tests/
│   ├── unit/test_library_service.py   # use cases with in-memory fakes: fast, no web, no DB
│   └── integration/test_api.py        # HTTP + real SQLAlchemy (sqlite in memory)
├── Dockerfile  .dockerignore  compose.yaml
└── requirements.txt
```

## The dependency rule

```
   adapters/inbound/http  ──►  application  ──►  domain
   adapters/outbound      ──►  application (implements ports)  ──►  domain
   main.py (composition root) knows everybody and wires them
```

Arrows point **inward**. `domain/` never imports `fastapi`, `sqlalchemy` or `pydantic`.
`application/` only knows the ports (interfaces), never the concrete adapters.
That is why you can test the whole business logic in milliseconds with fakes, and why you can
swap SQLite for Postgres (or HTTP for a CLI or a LangGraph tool) without touching the core.

| Term | Here |
|---|---|
| Entity / domain model | `domain/models.py: Book` |
| Port (outbound / driven) | `application/ports.py: BookRepository, Clock` |
| Use case / inbound port | `application/services.py: LibraryService` |
| Driving adapter | `adapters/inbound/http/*` |
| Driven adapter | `adapters/outbound/*` |
| Composition root | `main.py: create_app()` |

## Run it

```bash
cd fastapi/reference_project
pip install -r requirements.txt
python -m pytest -q                      # unit + integration tests
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
LIB_DATABASE_URL=memory uvicorn app.main:app   # switch adapter through configuration only
docker compose up --build                # see ../../docker
```

## Rebuild exercise (do this after FastAPI exercises 14–19)

In a new empty folder, recreate this project **from memory**, in this order:

1. `domain/` (entity + errors), plus one unit test of the rules
2. `application/ports.py` and `services.py`, plus unit tests with an in-memory repo
3. `adapters/outbound/memory.py`, then `sqlalchemy_repo.py`
4. `adapters/inbound/http/` (schemas, routes, error mapping)
5. `main.py` with `create_app(settings)`, then the integration tests
6. `Dockerfile` and `compose.yaml`

Then copy this repo's `tests/` folder into yours and run `pytest`. When they pass, you're done.
Do it 3 times, on 3 different days.
