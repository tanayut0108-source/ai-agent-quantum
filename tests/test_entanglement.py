"""Tests for the EntanglementBus."""

from quantum_agent.core.entanglement import EntanglementBus


class TestEntanglementBus:
    def test_publish_and_get(self):
        bus = EntanglementBus()
        bus.publish("agent-1", "key", "value")
        assert bus.get("key") == "value"

    def test_subscribe_receives_events(self):
        bus = EntanglementBus()
        received = []
        bus.subscribe("agent-2", "status", lambda e: received.append(e))
        bus.publish("agent-1", "status", "done")
        assert len(received) == 1
        assert received[0].value == "done"

    def test_publisher_does_not_receive_own_event(self):
        bus = EntanglementBus()
        received = []
        bus.subscribe("agent-1", "key", lambda e: received.append(e))
        bus.publish("agent-1", "key", "val")
        assert len(received) == 0

    def test_snapshot(self):
        bus = EntanglementBus()
        bus.publish("a", "x", 1)
        bus.publish("a", "y", 2)
        snap = bus.snapshot()
        assert snap == {"x": 1, "y": 2}

    def test_event_log(self):
        bus = EntanglementBus()
        bus.publish("a", "k", "v1")
        bus.publish("b", "k", "v2")
        assert len(bus.event_log) == 2
