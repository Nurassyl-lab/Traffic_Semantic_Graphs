from torch_geometric.loader import DataLoader
from src.graph_encoding.data_loaders import get_graph_dataset

def load_graph_dataloaders(shuffle=True, batch_size=4, mode='all', side_info_path=None):
    l2d_dataset = get_graph_dataset(
        root_dir='./data/graph_dataset/L2D/',
        mode=mode,
        side_information_path=side_info_path,
        node_features_to_exclude=None
    )
    nuplan_dataset = get_graph_dataset(
        root_dir='./data/graph_dataset/NuPlan/',
        mode=mode,
        node_features_to_exclude=None
    )

    l2d_train_loader = DataLoader(
        l2d_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    num_workers=2,
    pin_memory=True,
    )
    nuplan_train_loader = DataLoader(
        nuplan_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    num_workers=2,
    pin_memory=True,
    )
    return l2d_train_loader, nuplan_train_loader

if __name__ == "__main__":
    print("Loading graph dataloaders...")

    l2d_dataloader = load_graph_dataloaders(shuffle=False, batch_size=1)[0]

    for batch in l2d_dataloader:
        print("\n=== batch ===\n", batch)

        print("\n\n=== NODE TYPES ===")
        for node_type in batch.node_types:
            node_store = batch[node_type]

            if "x" in node_store:
                x = node_store["x"]
                print(f"{node_type}: {x.shape[0]} nodes, {x.shape[1]} features")
            else:
                print(f"{node_type}: has NO 'x' tensor")

        print("\n\n=== EDGE TYPES ===")
        for edge_type in batch.edge_types:
            edge_store = batch[edge_type]

            if "edge_attr" in edge_store:
                print(f"{edge_type}: edges={edge_store.edge_index.shape[1]}, edge_feat_dim={edge_store.edge_attr.shape[1]}")
            else:
                print(f"{edge_type}: edges={edge_store.edge_index.shape[1]}, NO edge features")
           
        print("\n\n=== FEATURE DIMENSIONS ===")
        for node_type in batch.node_types:
            node_store = batch[node_type]
            if "x" in node_store:
                print(f"Node type '{node_type}' has feature dimension: {node_store.x.shape[1]}")
            else:
                print(f"Node type '{node_type}' has NO features.")

        # print("\n\n=== batch.dict ===", batch.__dict__.keys())
        # print("\nbatch.dict:\n", batch.__dict__)

        # print("\n\n=== batch._node_store_dict.keys() ===\n", batch._node_store_dict.keys())

        # print("\n\n === NODES ===")
        # print("\nEGO:")
        # EGO = batch._node_store_dict["ego"].__dict__["_mapping"]
        # for key, item in EGO.items():
        #     print(f"{key}: {item}, item.shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")

        # print("\nENV:")
        # ENV = batch._node_store_dict["environment"].__dict__["_mapping"]
        # for key, item in ENV.items():
        #     print(f"{key}: {item}, item.shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")

       
        break