"""8-node mock KG E2E full chain test"""
import sys
sys.path.insert(0, "python")

def test_full_chain_with_rust():
    try:
        from tecs import tecs_rs
        engine = tecs_rs.RustEngine()
    except ImportError:
        print("  [SKIP] tecs_rs not available")
        return

    # 8-node graph: two 4-cycles connected by bridge
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(3,4)]
    topo = engine.compute_topology_from_edges(edges, 8)

    print(f"  β₀={topo['beta0']}, β₁={topo['beta1']}")
    print(f"  H₁ bars: {topo['long_h1']}")
    print(f"  max_persistence_h1: {topo['max_persistence_h1']}")

    assert topo['beta0'] == 1, f"Expected β₀=1 (connected), got {topo['beta0']}"
    assert topo['beta1'] >= 1, f"Expected β₁≥1 (cycles), got {topo['beta1']}"
    print("  8-node full chain: PASS")

def test_tree_no_cycle():
    """Tree graph should have β₁=0"""
    try:
        from tecs import tecs_rs
        engine = tecs_rs.RustEngine()
    except ImportError:
        print("  [SKIP] tecs_rs not available")
        return

    # Simple tree: 0-1, 1-2, 1-3
    edges = [(0,1),(1,2),(1,3)]
    topo = engine.compute_topology_from_edges(edges, 4)

    print(f"  Tree: β₀={topo['beta0']}, β₁={topo['beta1']}")
    assert topo['beta0'] == 1
    assert topo['beta1'] == 0
    print("  tree no cycle: PASS")

if __name__ == "__main__":
    test_full_chain_with_rust()
    test_tree_no_cycle()
    print("\nAll full chain tests passed!")
