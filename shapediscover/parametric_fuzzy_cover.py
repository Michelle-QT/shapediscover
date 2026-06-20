import torch
import torch.nn as nn
import numpy as np


def sparsemax(z, dim=0):
    """Sparsemax (Martins and Astudillo, 2016): the Euclidean projection of ``z``
    onto the probability simplex along ``dim``.

    Like softmax it returns a partition of unity (sums to 1 along ``dim``), but
    unlike softmax it is genuinely sparse (exact zeros). Used as an alternative
    base map for the partition of unity so the induced fuzzy cover has compact
    support: zeros propagate through the p-simplex normalization, so intersections
    become genuinely empty and the nerve becomes genuinely sparse. Differentiable
    via autograd through the sort / threshold (a valid (sub)gradient on each
    support region).
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    n = z.shape[dim]
    rng = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = n
    rng = rng.view(shape)
    z_cumsum = z_sorted.cumsum(dim)
    support = (1.0 + rng * z_sorted) > z_cumsum
    k = support.to(z.dtype).sum(dim=dim, keepdim=True)
    tau = (torch.gather(z_cumsum, dim, k.long() - 1) - 1.0) / k
    return torch.clamp(z - tau, min=0.0)


def entmax15(z, dim=0):
    """Alpha=1.5 entmax (Peters, Niculae, Martins, 2019): a sparse projection onto
    the simplex that sits between softmax (alpha=1, dense) and sparsemax (alpha=2).

    It keeps strictly more support than sparsemax (a gentler sparsity), useful when
    full sparsemax over-thins the cover and destroys overlap-dependent topology
    (the tori). Differentiable via autograd through the sort / threshold.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    z = z / 2.0
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    n = z.shape[dim]
    rho = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = n
    rho = rho.view(shape)
    mean = z_sorted.cumsum(dim) / rho
    mean_sq = (z_sorted ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1.0 - ss) / rho
    delta_nz = torch.clamp(delta, min=0.0)
    tau = mean - torch.sqrt(delta_nz)
    support_size = (tau <= z_sorted).to(z.dtype).sum(dim=dim, keepdim=True)
    tau_star = torch.gather(tau, dim, support_size.long() - 1)
    return torch.clamp(z - tau_star, min=0.0) ** 2


class PartitionOfUnity(torch.nn.Module):
    def __init__(
        self,
        model,
        map_to_simplex="softmax",
        temperature=1.0,
    ):
        """
            Implements an optimizable partition of unity built on top of an optimizable function.

            Parameters
            ----------

            model : an optimizable function
            map_to_simplex : str
                The differentiable map onto the simplex: "softmax" (default, full
                support, the dense nerve) or "sparsemax" (compact support, a
                genuinely sparse cover and nerve).
            temperature : float
                Logits are divided by this before the map. For "sparsemax" it is
                the sparsity knob: larger temperature keeps more cover elements per
                point (denser, more overlap), smaller is sparser. 1.0 is plain
                sparsemax / softmax.

        """
        super().__init__()
        self._model = model
        if map_to_simplex not in ("softmax", "sparsemax", "entmax15"):
            raise ValueError(
                "map_to_simplex must be 'softmax', 'sparsemax', or 'entmax15'; "
                f"got {map_to_simplex!r}."
            )
        self._map_to_simplex = map_to_simplex
        self._temperature = temperature

    def forward(self):
        z = self._model() / self._temperature
        if self._map_to_simplex == "sparsemax":
            return sparsemax(z, dim=0)
        if self._map_to_simplex == "entmax15":
            return entmax15(z, dim=0)
        return torch.softmax(z, dim=0)


class PointCloudFunction(torch.nn.Module):
    def __init__(self, pointcloud, n_dimensions, inner_layer_widths):
        super().__init__()

        data_dimension = pointcloud.shape[1]

        self._pointcloud = torch.tensor(
            pointcloud, dtype=torch.float32, requires_grad=False
        )

        if len(inner_layer_widths) == 0:
            layers = [nn.Linear(data_dimension, n_dimensions)]
        else:
            first_width = inner_layer_widths[0]
            last_width = inner_layer_widths[-1]
            layers = [nn.Linear(data_dimension, first_width), nn.Sigmoid()]
            for i in range(len(inner_layer_widths) - 1):
                layers.append(
                    nn.Linear(inner_layer_widths[i], inner_layer_widths[i + 1])
                )
                layers.append(nn.Sigmoid())
                # layers.append(nn.Relu())
            layers.append(nn.Linear(last_width, n_dimensions))

        self._model = nn.Sequential(*layers)

    def forward(self):
        return self._model(self._pointcloud).T


class GraphFunction(nn.Module):
    def __init__(self, graph, node_attributes, n_dimensions, inner_layer_widths):
        super().__init__()

        try:
            import torch_geometric
            import torch_geometric.nn as gnn
        except ImportError as e:
            raise ImportError(
                "The 'graph_nn' model requires torch_geometric. Install it with "
                "`pip install torch_geometric` (the package's 'nn' extra)."
            ) from e

        edge_indices, edge_attributes = torch_geometric.utils.from_scipy_sparse_matrix(
            graph.adjacency_matrix()
        )
        edge_attributes = edge_attributes.to(torch.float32)
        edge_attributes.requires_grad = False

        self._node_attributes = torch.tensor(
            node_attributes, dtype=torch.float32, requires_grad=False
        )
        self._edge_index = edge_indices
        self._edge_weight = edge_attributes

        if len(inner_layer_widths) == 0:
            layers = [
                (
                    gnn.GCNConv(node_attributes.shape[1], n_dimensions),
                    "x, edge_index, edge_weight -> x",
                )
            ]
        else:
            first_width = inner_layer_widths[0]
            last_width = inner_layer_widths[-1]
            layers = [
                (
                    gnn.GCNConv(node_attributes.shape[1], first_width),
                    "x, edge_index, edge_weight -> x",
                ),
                (nn.Sigmoid(), "x -> x"),
            ]
            for i in range(len(inner_layer_widths) - 1):
                layers.append(
                    (
                        gnn.GCNConv(inner_layer_widths[i], inner_layer_widths[i + 1]),
                        "x, edge_index, edge_weight -> x",
                    )
                )
                layers.append((nn.Sigmoid(), "x -> x"))
            layers.append(
                (
                    gnn.GCNConv(last_width, n_dimensions),
                    "x, edge_index, edge_weight -> x",
                )
            )

        self._model = gnn.Sequential("x, edge_index, edge_weight", layers)

    def forward(self):
        return self._model(self._node_attributes, self._edge_index, self._edge_weight).T


class SetFunction(nn.Module):
    def __init__(self, set_size, n_dimensions, initialization=None):
        """
            Implements an optimizable function defined on the set {1, ..., set_size} and taking values in R to the n_dimensions.

            Parameters
            ----------
                set_size : int

                n_dimensions : int

                initialization : None or numpy array of shape (n_dimensions, set_size)
        """
        super().__init__()

        if initialization is None:
            initialization = np.random.random((n_dimensions, set_size))

        self._param = nn.Parameter(
            torch.tensor(initialization, dtype=torch.float32, requires_grad=True)
        )

    def forward(self):
        return self._param
