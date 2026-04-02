# Contributing to Agent Health Monitor

Thank you for your interest in contributing to Agent Health Monitor! This project provides x402-powered API services for Base blockchain agent wallet health analysis. This guide will help you get started.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [x402 Integration](#x402-integration)
- [Database](#database)
- [API Development](#api-development)
- [Deployment](#deployment)
- [Submitting Changes](#submitting-changes)

## About the Project

Agent Health Monitor is an x402-powered service that:
- Analyzes Base blockchain agent wallet health
- Provides risk scoring for AI agent counterparties
- Offers proactive monitoring and alerting
- Supports ERC-8004 agent discovery

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (for x402 integration)
- PostgreSQL (or SQLite for development)
- Base network access (mainnet or Sepolia)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/moonshot-cyber/agent-health-monitor.git
   cd agent-health-monitor
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize the database**
   ```bash
   python db.py
   ```

5. **Run the API server**
   ```bash
   python api.py
   ```

## Development Environment

### Environment Variables

Required variables in `.env`:
```
# Database
DATABASE_URL=postgresql://user:pass@localhost/agent_health

# Base Network
BASE_RPC_URL=https://mainnet.base.org
BASE_SEPOLIA_RPC_URL=https://sepolia.base.org

# x402 Configuration
X402_FACILITATOR_URL=https://x402.org/facilitator
X402_PAYMENT_AMOUNT=0.01

# API Keys (for external services)
NANSEN_API_KEY=your_nansen_key
ALCHEMY_API_KEY=your_alchemy_key
```

### Using Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
├── api.py                 # Main Flask/FastAPI application
├── db.py                  # Database models and initialization
├── monitor.py             # Core monitoring logic
├── discover.py            # Agent discovery (ERC-8004)
├── generate_report_card.py # Report generation
├── scripts/               # Utility scripts
│   ├── seed_history.py
│   └── ...
├── tests/                 # Test suite
├── static/                # Static assets
└── docs/                  # Documentation
```

## Contributing Guidelines

### Types of Contributions

We welcome:
- **New Scanners**: Add support for new agent discovery protocols
- **Risk Models**: Improve wallet health scoring algorithms
- **API Endpoints**: New analysis endpoints
- **Frontend**: UI improvements for report cards
- **Documentation**: Tutorials, guides, examples
- **Bug Fixes**: Report and fix issues
- **Performance**: Optimize database queries or API responses

### Before Contributing

1. **Check existing issues** for similar requests
2. **Open a discussion** for major features
3. **Fork the repository** and create a feature branch

### Contribution Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Test locally**
   ```bash
   pytest tests/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add ERC-8004 compliance scanner"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python

Follow PEP 8 with these specifics:
- Line length: 100 characters
- Use type hints for function signatures
- Docstrings for all public functions (Google style)

Example:
```python
def analyze_wallet_health(
    wallet_address: str,
    chain_id: int = 8453
) -> dict:
    """Analyze the health of an agent wallet.
    
    Args:
        wallet_address: The Ethereum address to analyze
        chain_id: Chain ID (default: 8453 for Base mainnet)
        
    Returns:
        Dictionary containing health score and risk factors
    """
    # Implementation
```

### Import Ordering

```python
# 1. Standard library
import os
import json
from datetime import datetime

# 2. Third-party
import requests
from flask import Flask
from web3 import Web3

# 3. Local modules
from db import Wallet, Transaction
from monitor import calculate_risk_score
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_monitor.py

# Run with verbose output
pytest -v
```

### Writing Tests

```python
def test_wallet_health_calculation():
    """Test that wallet health is calculated correctly."""
    # Arrange
    wallet = Wallet(address="0x123...", transactions=[])
    
    # Act
    health = calculate_health_score(wallet)
    
    # Assert
    assert 0 <= health.score <= 100
    assert isinstance(health.risk_factors, list)
```

### Test Coverage

Aim for:
- 80%+ coverage on new code
- 100% coverage on critical paths (risk scoring, API endpoints)
- Integration tests for external API calls (mocked)

## x402 Integration

### Understanding x402

x402 is a payment protocol for API access. This project uses x402 to:
- Require payment for premium analysis endpoints
- Enable per-request billing
- Support USDC payments on Base

### Adding x402 to New Endpoints

```python
from x402 import require_payment

@app.route('/api/premium-analysis')
@require_payment(amount="0.01", token="USDC")
def premium_analysis():
    """Endpoint requiring x402 payment."""
    wallet = request.args.get('wallet')
    return analyze_wallet_deep(wallet)
```

### Testing x402 Locally

Use the x402 testnet facilitator for development:
```python
X402_FACILITATOR_URL = "https://testnet.x402.org/facilitator"
```

## Database

### Schema Changes

When modifying the database schema:
1. Update `db.py` with new models
2. Create migration scripts in `scripts/migrations/`
3. Test migrations on a copy of production data
4. Document changes in PR description

### Database Models

Key models:
- `Wallet`: Agent wallet information
- `Transaction`: On-chain transactions
- `RiskScore`: Calculated risk metrics
- `Scan`: Discovery scan results

## API Development

### Adding New Endpoints

1. Define the route in `api.py`
2. Add input validation
3. Implement business logic
4. Add tests
5. Update API documentation

### API Response Format

Standard response structure:
```json
{
  "success": true,
  "data": {
    "wallet": "0x...",
    "health_score": 85,
    "risk_level": "low"
  },
  "meta": {
    "timestamp": "2025-04-02T10:00:00Z",
    "chain_id": 8453
  }
}
```

### Error Handling

```python
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Wallet not found",
        "code": "WALLET_NOT_FOUND"
    }), 404
```

## Deployment

### Railway (Recommended)

This project is configured for Railway deployment:

1. Connect your GitHub repo to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main

### Manual Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Run database migrations
python scripts/migrate.py

# Start with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api:app
```

### Cron Jobs

The project includes cron scripts for:
- Proactive scanning (`cron_acp_scan.py`)
- Batch processing (`acp_batch_scan_results.md`)

Configure via `railway.cron.json` or your hosting provider.

## Submitting Changes

### Pull Request Template

When creating a PR, include:
- **Description**: What changed and why
- **Testing**: How you tested the changes
- **Screenshots**: For UI changes
- **Checklist**:
  - [ ] Code follows style guidelines
  - [ ] Tests added/updated
  - [ ] Documentation updated
  - [ ] x402 integration tested (if applicable)

### Review Process

- All PRs require at least one review
- CI checks must pass
- Address review comments promptly
- Squash commits before merging

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## Getting Help

- Open a [GitHub Discussion](https://github.com/moonshot-cyber/agent-health-monitor/discussions)
- Check existing documentation
- Review similar PRs for examples

## Code of Conduct

Be respectful, constructive, and inclusive. We're building tools for the entire agent ecosystem on Base.

---

*Thank you for helping make agent interactions safer on Base!*
