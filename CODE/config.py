import numpy as np

# ─────────────────────────────────────────────
# UNIVERSE OF TICKERS
# ─────────────────────────────────────────────
TICKERS = {
    # ── ETFs Broad ──────────────────────────────────────────────────────────
    "etf_broad": [
        "VT", "VTI", "SPY", "QQQ", "IVV", "VOO", "EFA", "EEM", "VEA", "VWO",
        "ACWI", "URTH", "IWDA.L", "VWRD.L",
    ],
    "etf_regions": [
        "EWJ", "EWG", "EWQ", "EWU", "EWL", "EWI", "EWP", "EWD", "EWN",
        "INDA", "EWZ", "MCHI", "GXC", "AFRI",
        "VGK", "VPL", "EWA",
    ],
    "etf_smallcap": ["IWM", "VB", "IJR", "SCHA"],

    # ── ETFs Sectorial ───────────────────────────────────────────────────────
    "etf_sectorial": [
        "XLK", "VGT", "QQQ",
        "XLV", "VHT",
        "XLF", "VFH",
        "XLE", "VDE",
        "XLI", "VIS",
        "XLY", "VCR",
        "XLP", "VDC",
        "XLB", "VAW",
        "XLU", "VPU",
        "XLC", "VOX",
        "XLRE", "VNQ",
    ],

    # ── ETFs Factor ──────────────────────────────────────────────────────────
    "etf_factor": [
        "VTV", "IWD",          # Value
        "VUG", "IWF",          # Growth
        "MTUM", "MOAT",        # Momentum
        "QUAL", "JQUA",        # Quality
        "USMV", "SPLV",        # Low Vol
        "VIG", "DVY", "SCHD",  # Dividend
        "VBR", "IWN",          # SmallCap Value
    ],

    # ── ETFs Thematic ────────────────────────────────────────────────────────
    "etf_thematic": [
        "AIQ", "ROBO", "IRBO",     # AI / Robotics
        "HACK", "CIBR", "BUG",     # Cybersecurity
        "ICLN", "QCLN", "RNRG",   # Clean Energy
        "PHO", "FIW",              # Water
        "ARKG", "IBB",             # Genomics
        "UFO", "ARKX",             # Space
        "META", "METV",            # Metaverse (ETFs)
        "MOO",                     # Agriculture
        "PAVE", "IGF",             # Infrastructure
        "PSP", "PEX",              # Private Equity
        "ESGU", "ESGV", "SUSL",   # ESG
    ],

    # ── ETFs Bonds ───────────────────────────────────────────────────────────
    "etf_bonds": [
        "BWX", "BNDX",            # World Gov
        "SHY", "IEI", "VGSH",    # US Short Term
        "TLT", "VGLT", "EDV",    # US Long Term
        "IBGL.L", "IEGA.L",      # Europe Gov
        "LQD", "VCIT",            # Corp IG
        "HYG", "JNK",             # High Yield
        "EMB", "PCY",             # EM Bonds
        "TIP", "SCHP",            # TIPS
        "ICVT", "CWB",            # Convertibles
        "MUB", "VTEB",            # Municipal
        "AGG", "BND",             # Broad Bond
    ],

    # ── ETFs Commodities ─────────────────────────────────────────────────────
    "etf_commodities": [
        "GLD", "IAU", "SGOL",    # Gold
        "SLV", "SIVR",           # Silver
        "PPLT",                  # Platinum
        "USO", "BNO",            # Oil
        "UNG",                   # Natural Gas
        "CPER",                  # Copper
        "JJN",                   # Nickel/Aluminum
        "WEAT",                  # Wheat
        "CORN",                  # Corn
        "SOYB",                  # Soy
        "NIB",                   # Cocoa
        "WOOD",                  # Wood
        "DJP", "PDBC", "GSG",   # Diversified Commodities
    ],

    # ── ETFs Real Estate ─────────────────────────────────────────────────────
    "etf_realestate": [
        "RWO", "VNQI",           # World REIT
        "VNQ", "IYR", "SCHH",   # US REIT
        "IPRP.L", "EPRA.L",     # Europe REIT
        "IFAS",                  # Asia REIT
    ],

    # ── ETFs Crypto ──────────────────────────────────────────────────────────
    "etf_crypto": [
        "IBIT", "FBTC", "GBTC",  # Bitcoin
        "ETHA", "ETHE",          # Ethereum
        "BITQ", "BKCH",          # Blockchain / Diversified Crypto
    ],

    # ── Individual stocks USA ────────────────────────────────────────────────
    "stocks_usa_tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AMD", "INTC", "ORCL",
    ],
    "stocks_usa_finance": [
        "JPM", "BAC", "GS", "MS", "BRK-B", "V", "MA",
    ],
    "stocks_usa_health": [
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",
    ],
    "stocks_usa_energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
    ],
    "stocks_usa_consumer": [
        "KO", "PEP", "MCD", "WMT", "COST", "PG", "NKE",
    ],
    "stocks_usa_industry": [
        "CAT", "BA", "GE", "HON", "MMM", "UPS",
    ],
    "stocks_usa_utilities": [
        "NEE", "DUK", "SO", "AEP",
    ],
    "stocks_usa_telecom": [
        "T", "VZ", "TMUS",
    ],
    "stocks_usa_realestate": [
        "AMT", "PLD", "CCI", "EQIX",
    ],

    # ── Individual stocks Europe ─────────────────────────────────────────────
    "stocks_europe_france": [
        "MC.PA", "TTE.PA", "SAN.PA", "BNP.PA", "AIR.PA",
        "SU.PA", "AI.PA", "OR.PA", "RI.PA", "CAP.PA",
        "KER.PA", "DSY.PA", "HO.PA", "STM.PA",
    ],
    "stocks_europe_germany": [
        "SAP.DE", "SIE.DE", "ALV.DE", "BMW.DE",
        "VOW3.DE", "BAYN.DE", "MBG.DE", "DTE.DE",
    ],
    "stocks_europe_uk": [
        "SHEL.L", "HSBA.L", "AZN.L", "BP.L",
        "GSK.L", "ULVR.L", "RIO.L", "LSEG.L",
    ],
    "stocks_europe_swiss": [
        "NESN.SW", "NOVN.SW", "ZURN.SW", "ABBN.SW",
    ],
    "stocks_europe_netherlands": [
        "ASML.AS", "PHIA.AS", "HEIA.AS", "ADYEN.AS",
    ],
    "stocks_europe_italy": [
        "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI",
    ],
    "stocks_europe_spain": [
        "SAN.MC", "IBE.MC", "ITX.MC", "BBVA.MC",
    ],
    "stocks_europe_scandinavia": [
        "NOVO-B.CO", "MAERSK-B.CO", "VOLV-B.ST", "ERIC-B.ST",
    ],

    # ── Individual stocks Asia ───────────────────────────────────────────────
    "stocks_asia_japan": [
        "7203.T", "6758.T", "9984.T", "6861.T", "7974.T",
    ],
    "stocks_asia_china": [
        "9988.HK", "0700.HK", "BABA", "JD", "BIDU", "PDD", "NIO",
    ],
    "stocks_asia_india": [
        "INFY", "WIT", "HDB", "IBN",
    ],
    "stocks_asia_korea": [
        "005930.KS", "000660.KS",
    ],
    "stocks_asia_taiwan": [
        "2330.TW", "TSM",
    ],
    "stocks_asia_australia": [
        "BHP.AX", "CBA.AX", "CSL.AX",
    ],
    "stocks_asia_brazil": [
        "VALE", "PBR", "ITUB", "BBD",
    ],

    # ── Currencies ETFs ──────────────────────────────────────────────────────
    "etf_currencies": [
        "UUP", "FXY", "CYB", "FXB", "FXF", "CEW",
    ],

    # ── Volatility / Inverse (for reference, will be excluded) ───────────────
    "etf_volatility_inverse": [
        "SH", "SPXS", "PSQ", "TBF", "TBT", "TBX.PA",
        "UVXY", "VXX", "UPRO", "TQQQ", "TNA", "SSO", "QLD", "UWM", "SVOL.L",
    ],
}

# De-duplicate preserving order
ALL_TICKERS = list(dict.fromkeys([t for g in TICKERS.values() for t in g]))

# ─────────────────────────────────────────────
# MARKET BENCHMARK
# ─────────────────────────────────────────────
MARKET_TICKER = "^GSPC"

# ─────────────────────────────────────────────
# TIME PERIODS
# ─────────────────────────────────────────────
PERIODS = {
    "1_an":  ("2024-01-01", "2024-12-31"),
    "3_ans": ("2022-01-01", "2024-12-31"),
    "5_ans": ("2020-01-01", "2024-12-31"),
}

# ─────────────────────────────────────────────
# FINANCIAL PARAMETERS
# ─────────────────────────────────────────────
TAUX_LIVRET_A = 0.015                                         # Annual risk-free rate
RF_DAILY = (1 + TAUX_LIVRET_A) ** (1 / 252) - 1              # Daily rf (compound)
TRADING_DAYS = 252

# ─────────────────────────────────────────────
# DATA QUALITY THRESHOLDS
# ─────────────────────────────────────────────
MAX_NAN_PCT = 0.10       # Drop ticker if >10% missing after ffill
MAX_FFILL_DAYS = 3       # Forward-fill gaps up to 3 business days
OUTLIER_THRESHOLD = 0.15 # Flag (not remove) daily |log-return| > 15%

# ─────────────────────────────────────────────
# PORTFOLIO CONSTRAINTS
# ─────────────────────────────────────────────
N_MAX_TITRES = 13        # Max number of holdings in final portfolio

# ─────────────────────────────────────────────
# EXCLUSION LIST (leveraged / inverse / volatility products)
# ─────────────────────────────────────────────
EXCLUSIONS = [
    "SH", "SPXS", "PSQ", "TBF", "TBT", "TBX.PA",
    "UVXY", "VXX", "UPRO", "TQQQ", "TNA", "SSO", "QLD", "UWM", "SVOL.L",
]
