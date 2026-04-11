"""A2A Agent Card discovery endpoint tests."""


class TestA2AAgentCard:
    """A2A Agent Card at /.well-known/agent.json."""

    def test_well_known_agent_json(self, client):
        """GET /.well-known/agent.json returns valid A2A Agent Card."""
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        data = resp.json()
        # Required A2A Agent Card fields
        assert data["name"] == "Agent Health Monitor (AHM)"
        assert "url" in data
        assert "skills" in data
        assert len(data["skills"]) > 0
