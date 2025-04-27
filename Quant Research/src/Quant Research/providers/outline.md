Providers Layer Outline
Overview

Purpose: Edge-layer adapters for data ingestion from various sources
Location: src/quant_research/providers/
Function: Retrieves raw data and emits validated PriceBar or custom records

Architecture

Extensible Design: New domains can be added by simply creating a new provider file
Interface Pattern: All providers implement the BaseProvider interface

Core Components
Base Provider Interface

File: base.py
Purpose: Defines the contract that all provider implementations must follow

Available Provider Implementations
Cryptocurrency Data

File: crypto_ccxt.py
Functionality: OHLCV data via CCXT library and websocket connections
Data Sources: Various cryptocurrency exchanges

Blockchain Analytics

File: onchain.py
Functionality: Blockchain metrics and on-chain data
Data Sources: Etherscan and Glassnode APIs

Sentiment Analysis

File: sentiment_twitter.py
Functionality: Social media sentiment data collection and processing
Technology: Twint for data collection + Hugging Face transformers for sentiment analysis

Equities Market Data

File: equities_yf.py
Functionality: Stock market data ingestion
Data Source: Likely Yahoo Finance

Integration Points

Providers feed data into the core models system
Used by orchestration pipelines (Prefect/Airflow DAGs)
Output is standardized to work with the analytics engine

Implementation Details

Each provider validates and transforms raw data into standardized formats
Data flows into core/models.py as PriceBar, Signal, etc. (Pydantic models)
Eventually stored as Parquet files in DuckDB