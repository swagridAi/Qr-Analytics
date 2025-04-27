flowchart TD
    subgraph "Core Components"
        A1[BaseProvider Interface]
        A2[Provider Config]
        A3[Connection Manager]
        A4[Provider Factory]
    end

    subgraph "Provider Implementations"
        B1[CCXT Provider]
        B2[Yahoo Finance Provider]
        B3[Onchain Provider]
        B4[Sentiment Provider]
    end

    subgraph "Error Handling"
        C1[Connection Errors]
        C2[Data Fetch Errors]
        C3[Rate Limit Errors]
        C4[Validation Errors]
    end

    subgraph "Data Models"
        D1[PriceBar]
        D2[Ticker]
        D3[OrderBook]
        D4[Signal]
    end

    subgraph "CLI Tools"
        E1[Provider List]
        E2[Config Validation]
        E3[Test Command]
        E4[Fetch Command]
    end

    A1 --> B1 & B2 & B3 & B4
    A2 --> B1 & B2 & B3 & B4
    A3 --> B1 & B2 & B3 & B4
    A4 --> B1 & B2 & B3 & B4

    B1 & B2 & B3 & B4 --> D1 & D2 & D3 & D4
    B1 & B2 & B3 & B4 --> C1 & C2 & C3 & C4

    A4 --> E1 & E2 & E3 & E4
    A2 --> E2
    B1 & B2 & B3 & B4 --> E3 & E4