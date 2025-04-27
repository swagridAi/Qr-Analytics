src/quant_research/analytics/common/serialization/
├── __init__.py            # Public API and registry initialization
├── core.py                # Core interfaces, formats, and base functionality
├── formats/               # Format-specific implementations
│   ├── __init__.py        # Format registry and imports
│   ├── json.py            # JSON serialization
│   ├── parquet.py         # Parquet serialization
│   ├── pickle.py          # Pickle serialization
│   └── arrow.py           # Arrow serialization
├── serializers/           # Type-specific serializers
│   ├── __init__.py        # Serializer registry
│   ├── primitives.py      # Basic type serializers (datetime, numpy, etc.)
│   ├── models.py          # Domain model serializers (Signal, etc.)
│   └── dataframes.py      # DataFrame-specific serialization
└── utils.py               # Utility functions and batch operations