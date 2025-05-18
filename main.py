
# main.py

import argparse
from algorithms.graph_builder import build_graph
from algorithms.astar_search   import astar
from utils.edge_mapper         import EdgeMapper
from models.gru_predictor      import GRUPredictor
from utils.flow_to_speed       import flow_to_speed

def main():
    p = argparse.ArgumentParser(
        description="TBRGS: Traffic‑Based Route Guidance System (GRU)"
    )
    p.add_argument('--source',    required=True, help='Origin site ID (e.g. 0970)')
    p.add_argument('--target',    required=True, help='Destination site ID (e.g. 3685)')
    p.add_argument('--timestamp', required=True,
                   help='Prediction timestamp (YYYY‑MM‑DD HH:MM:SS)')
    p.add_argument('--nodes',     default='data/scats_complete_average.csv',
                   help='Path to node centroids CSV')
    p.add_argument('--volumes',   default='data/traffic_model_ready.pkl',
                   help='Path to volume pickle')
    p.add_argument('--models',    default='models/gru_models',
                   help='Directory of trained GRU arm models')
    args = p.parse_args()

    # 1) Build the road‐network graph
    print(f"🔍 Building graph from {args.nodes} …")
    centroids, edges = build_graph(args.nodes)

    # 2) Instantiate the arm‐mapper & your GRU‐based predictor
    print("🗺️  Initializing edge→arm mapper & GRU predictor …")
    mapper    = EdgeMapper(args.volumes)
    predictor = GRUPredictor(
        data_pkl   = args.volumes,
        models_dir = args.models
    )

    # 3) Define how to get flow for any edge A→B
    def get_volume_at_edge(A, B):
        loc  = mapper.best_arm(A, B, centroids)
        flow = predictor.predict(A, loc, args.timestamp)
        return flow

    # 4) Run A* search under your predicted traffic
    print(f"🚦 Running A* from {args.source} → {args.target} at {args.timestamp} …")
    path, total_time = astar(
        args.source, args.target,
        centroids, edges,
        get_volume_at_edge,
        predictor,
        args.timestamp
    )

    if not path:
        print("❌ No route found.")
        return

    # 5) Sum up the total distance
    dist_map  = {(u, v): d for u, v, d in edges}
    total_dist = sum(dist_map.get((u, v), 0.0) for u, v in zip(path, path[1:]))

    # 6) Print it all out
    print("\n🛣️ Optimal route:")
    print("   " + " → ".join(path))
    print(f"\n📏 Total distance: {total_dist:.2f} km")
    print(f"⏱️ Total travel time: {total_time:.1f} minutes")

if __name__ == "__main__":
    main()
