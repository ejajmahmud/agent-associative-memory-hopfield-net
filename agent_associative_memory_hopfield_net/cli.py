"""CLI for agent-associative-memory-hopfield-net."""
import sys
import argparse
from .hopfield import ModernHopfieldMemory

def main():
    parser = argparse.ArgumentParser(description="Agent Associative Memory Hopfield Net CLI")
    parser.add_argument("--demo", action="store_true", help="Run modern Hopfield associative memory retrieval demo")
    args = parser.parse_args()

    if args.demo:
        net = ModernHopfieldMemory(beta=5.0)
        net.store_pattern("config_dev", [1.0, 1.0, 0.0, 0.0])
        net.store_pattern("config_prod", [0.0, 0.0, 1.0, 1.0])
        # Corrupted partial query
        label, recovered = net.retrieve([0.9, 0.7, 0.1, 0.0])
        print(f"[+] Query [0.9, 0.7, 0.1, 0.0] converged to stored pattern: '{label}'")
        print(f"[+] Reconstructed vector: {[round(x, 2) for x in recovered]}")
    else:
        parser.print_help()
    sys.exit(0)

if __name__ == "__main__":
    main()
