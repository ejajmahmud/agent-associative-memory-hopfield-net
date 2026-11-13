"""Unit tests for agent-associative-memory-hopfield-net."""
import unittest
from agent_associative_memory_hopfield_net.hopfield import ModernHopfieldMemory

class TestModernHopfield(unittest.TestCase):
    def setUp(self):
        self.net = ModernHopfieldMemory(beta=8.0)

    def test_exact_recall(self):
        self.net.store_pattern("P1", [1.0, 0.0, 0.0])
        self.net.store_pattern("P2", [0.0, 1.0, 0.0])
        label, vec = self.net.retrieve([1.0, 0.0, 0.0])
        self.assertEqual(label, "P1")

    def test_noisy_recovery(self):
        self.net.store_pattern("TargetA", [1.0, 1.0, 1.0])
        self.net.store_pattern("TargetB", [-1.0, -1.0, -1.0])
        label, _ = self.net.retrieve([0.8, 0.9, 0.7])
        self.assertEqual(label, "TargetA")

if __name__ == "__main__":
    unittest.main()
