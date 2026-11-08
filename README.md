# dl-code-gym

Blank-file practice for PyTorch, Hugging Face, LangGraph, FastAPI (with software architecture) and Docker: the code that shows up in every project and most interviews, broken into small exercises you rewrite from memory until you can write them without notes.

```
dl-code-gym/
├── pytorch/exercises/        34 exercises
│   └── 07_linear_regression_manual/
│       ├── blank.py          <- you write here (task + self-check, no code)
│       └── solution.py       <- reference solution (same self-check)
├── huggingface/exercises/    17 exercises
├── langgraph/exercises/      13 exercises (graphs, routing, tools, memory, human-in-the-loop, multi-agent, RAG)
├── fastapi/
│   ├── exercises/            19 exercises: 01-13 framework + key libraries, 14-19 architecture
│   └── reference_project/    a complete hexagonal FastAPI service with tests and Docker, to study and rebuild
├── docker/exercises/         12 exercises (Dockerfile.blank / Dockerfile.solution + README with check commands)
├── PROGRESS.md               tick boxes: day 1 / day 2 / day 3 / no peeking
└── requirements.txt
```

## The method

1. Open `blank.py`. Read the task at the top.
2. Write the code **from memory**, imports included. Type it, don't paste it.
3. Stuck? Open `solution.py`, read **one line**, close it, and continue. Write down where you got stuck, because that's the gap you need to work on.
4. Run `python blank.py`. The self-check at the bottom prints `All checks passed!` when your code is right.
5. **Repeat the same exercise on 3 different days.** Reset it with `git checkout -- blank.py` (or delete your code). The third time it should come out without thinking.
6. 30–45 minutes a day works better than 5 hours once a week.

## Setup

```bash
pip install -r requirements.txt
```

Run each exercise from the repo root so datasets download into a single `data/` folder:

```bash
python pytorch/exercises/07_linear_regression_manual/blank.py
```

## Suggested order

| Weeks | Exercises | Goal |
|---|---|---|
| 1 | PyTorch 01–06 | tensors, shapes, autograd: predict every shape before running |
| 2 | PyTorch 07–12 | **the training loop until it's automatic** (07, 08, 12 are the core ones) |
| 3 | PyTorch 13–19 | real data, CNNs, checkpoints, transfer learning, training tricks |
| 4 | PyTorch 20–29 | from-scratch interview classics (softmax, cross-entropy, Adam, BatchNorm, conv, micrograd) |
| 5 | PyTorch 30–34 | attention → multi-head → block → tiny GPT |
| 6 | HF 01–06 | pipelines, tokenizers, models, datasets, collators |
| 7 | HF 07–09 | fine-tuning: your own loop first, then `Trainer` |
| 8 | HF 10–17 | generation, embeddings, NER alignment, LoRA, tokenizer training |

Then build your own project, for example fine-tuning a model on Darija/Arabic text classification and pushing it to the Hub.

### Engineering track (can run in parallel with the ML track)

| Weeks | Exercises | Goal |
|---|---|---|
| 1 | FastAPI 01–05 | routes, Pydantic v2, errors, CRUD with routers, `Depends` |
| 2 | FastAPI 06–13 | JWT auth, SQLAlchemy 2.0, pydantic-settings, middleware, lifespan, async httpx, pytest, logging/retries |
| 3 | FastAPI 14–19 | **architecture**: layered → repository/Protocol → hexagonal → clean architecture → unit of work → domain events |
| 4 | `fastapi/reference_project` | read it, then rebuild it from an empty folder until its tests pass |
| 5 | Docker 01–07 | Dockerfile, caching, non-root, healthcheck, multi-stage, ARG/ENV, volumes |
| 6 | Docker 08–12 | Compose with Postgres, dev hot reload, a PyTorch inference image, CLI, dockerizing the reference project |
| 7 | LangGraph 01–06 | state, reducers, routing, loops, messages, the tool-calling (ReAct) loop |
| 8 | LangGraph 07–13 | memory, human-in-the-loop, streaming, subgraphs, map-reduce, supervisor, RAG |

The LangGraph exercises use small fake "LLM" functions, so they run offline and the checks are deterministic.
Each one shows where a real model (`init_chat_model(...)`) would plug in.

Final project combining everything: a LangGraph agent using a fine-tuned HF model, served by a hexagonal FastAPI app, shipped with Docker Compose.

**Docker exercises** have no Python self-check: each folder's `README.md` lists the `docker build` / `docker run` commands and the output you should see. Start Docker Desktop first.

## The "must write blindfolded" list (most frequent in interviews)

- PyTorch 08 / 12: training and eval loop (`zero_grad → backward → step`, `model.eval()` + `no_grad`)
- PyTorch 10 / 11: `nn.Module` and `Dataset` subclasses
- PyTorch 20 / 21: stable softmax, cross-entropy from logits
- PyTorch 30 / 31: scaled dot-product attention, multi-head attention
- PyTorch 24: LayerNorm vs BatchNorm
- HF 04 / 07: tokenizer → model → logits, and fine-tuning with a manual loop
- HF 13 / 14: label alignment with `-100`, and shifting labels for causal-LM loss
- FastAPI 04 / 05 / 07: CRUD router, `Depends` + `dependency_overrides`, SQLAlchemy session dependency
- FastAPI 16: ports and adapters with a composition root (explain "why" in an interview)
- LangGraph 06 / 07 / 08: tool loop, checkpointer + `thread_id`, `interrupt` + `Command(resume=...)`
- Docker 02 / 05: layer caching order, multi-stage builds

## Sources to read before each block (then come back and rebuild)

- PyTorch "Learn the Basics": pytorch.org/tutorials
- learnpytorch.io (Zero to Mastery PyTorch)
- Andrej Karpathy, "Neural Networks: Zero to Hero" (micrograd → GPT)
- d2l.ai for more architectures to rebuild
- Hugging Face course: huggingface.co/learn
- LangGraph docs and the free "Introduction to LangGraph" course on LangChain Academy
- FastAPI official tutorial (fastapi.tiangolo.com/tutorial), SQLAlchemy 2.0 "Unified Tutorial"
- "Architecture Patterns with Python" (Percival & Gregory, free at cosmicpython.com): repository, unit of work, events
- "Clean Architecture" (Robert C. Martin), and Alistair Cockburn's article on hexagonal architecture
- Docker docs "Get started" + the Dockerfile best-practices page
