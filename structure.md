base-rest-service-template/
 ├── src/
 │   ├── api/
 │   │   ├── rest/                   # FastAPI application
 │   │   │   ├── app.py              # FastAPI app setup
 │   │   │   ├── routes/             # API routes
 │   │   │   │   ├── health.py       # Health check endpoints 
 │   │   │   │   ├── sse.py          # SSE endpoints
 │   │   │   │   └── websocket.py    # WebSocket endpoints
 │   │   │   └── dependencies.py     # Dependency injection
 │   │   └── middleware/             # FastAPI middleware
 │   │       ├── cors.py
 │   │       ├── logging.py
 │   │       ├── error_handler.py
 │   │       └── metrics.py               
 │   │
 │   ├── core/                       # Core business logic
 │   │   ├── exceptions/             # Exception hierarchy
 │   │   └── services/               # Business logic services
 │   │
 │   ├── data/                       # Data access layer
 │   │   ├── clients/                # Database clients
 │   │   ├── repositories/           # Repository pattern
 │   │   ├── models/                 # Data models
 │   │   │   ├── postgres/           # SQLAlchemy models
 │   │   │   └── mongo/              # MongoDB models
 │   │   └── migrations/             # Database migrations
 │   │
 │   ├── handlers/                   # External service handlers
 │   │   ├── grpc_clients/           # gRPC client handlers
 │   │   └── http_clients/           # HTTP client handlers
 │   │
 │   ├── control/                    # Control layer
 │   │   └── agents/                 # Intelligent agents
 │   │
 │   ├── observability/              # Observability stack
 │   │   ├── logging/                # Structured logging
 │   │   ├── metrics/                # Prometheus metrics
 │   │   └── tracing/                # Distributed tracing
 │   │
 │   ├── schemas/                    # Pydantic schemas
 │   ├── config/                     # Configuration
 │   ├── constants/                  # Constants
 │   ├── utils/                      # Utilities
 │   └── main.py                     # Application entry point
 │
 ├── examples/                       # Example implementations
 │   ├── sse_client.py              # SSE consumer example
 │   └── websocket_client.py        # WebSocket client example
 │
 ├── docs/                           # Documentation
 ├── requirements/                   # Python dependencies
 ├── docker-compose.yml              # Docker services
 ├── Dockerfile                      # Application container
 ├── Makefile                        # Development commands
 └── .env.example
